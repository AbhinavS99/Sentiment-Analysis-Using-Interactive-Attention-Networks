import numpy as np
import nltk
import joblib
import yaml
import argparse
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def return_max(orig, new, flag):
    if flag == 0:
        if new > orig:
            orig = new

    elif flag == 1:
        if new - flag > orig:
            orig = new - flag
    
    return orig

def extraction(words, filename):
    max_context_len , max_aspect_len = 0, 0
    file = open(filename, 'r')
    lines = file.readlines()

    for i in range(0, len(lines), 3):
        text = lines[i]
        text = text.lower().strip()
        text_tokens = word_tokenize(text)
        words.extend(text_tokens)
        max_context_len = return_max(max_context_len, len(text_tokens), 1)

        asp = lines[i+1]
        asp = asp.lower().strip()
        asp_tokens = word_tokenize(asp)
        words.extend(asp_tokens)
        max_aspect_len = return_max(max_aspect_len, len(asp_tokens), 0)

    return max_aspect_len, max_context_len


def data_info(args):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if args.get('category') == 'laptop':
        filename_train = config['laptop_train_dataset_txt']
        filename_test = config['laptop_test_dataset_txt']

    elif args.get('category') == 'rest':
        filename_train = config['rest_train_dataset_txt']
        filename_test = config['rest_test_dataset_txt']
 
    word2id = {}
    word2id['<pad>'] = 0

    list_of_words = []

    train_max_aspect_len, train_max_context_len = extraction(list_of_words, filename_train)
    test_max_aspect_len, test_max_context_len = extraction(list_of_words,filename_test)

    word_count = Counter(list_of_words).most_common()
    for word, freq in word_count:
        if word not in word2id:
            if ' ' not in word and '\n' not in word:
                if 'aspect_term' not in word:
                    word2id[word] = len(word2id)


    max_aspect_len = max(train_max_aspect_len, test_max_aspect_len)
    max_context_len = max(train_max_context_len, test_max_context_len)

    print("Max Context Len ",max_context_len)
    convert_txt_to_pkl(args, config, word2id, max_aspect_len, max_context_len, 'train')
    convert_txt_to_pkl(args, config, word2id, max_aspect_len, max_context_len, 'test')
    
def pad(tokens, pad_len):
    tokens = tokens + [0]*(pad_len - len(tokens))
    tokens = tokens[:pad_len]
    return tokens

def convert_polarity(val):
    if val == 'positive':
        return 2
    if val == 'negative':
        return 0
    if val == 'neutral':
        return 1
    
def item_gen(word2id, text):
    holder = []
    text_tokens = word_tokenize(text)
    for token in text_tokens:
        if token.lower() in word2id:
            holder.append(word2id[token.lower()])
    return holder

def save_file(aspects, contexts, labels, len_aspects, len_contexts, output):
    aspects = np.array(aspects)
    contexts = np.array(contexts)
    labels = np.array(labels)
    len_aspects = np.array(len_aspects)
    len_contexts = np.array(len_contexts)
    data = (aspects, contexts, labels, len_aspects, len_contexts)
    joblib.dump(data, output)
    print("[+] Data Saved")

def convert_txt_to_pkl(args, config, word2id, max_aspect_len, max_context_len, split):
    if args.get('category') == 'laptop':
        if split == 'train':
            input = config['laptop_train_dataset_txt']
            output = config['laptop_train_pkl']
        elif split == 'test':
            input = config['laptop_test_dataset_txt']
            output = config['laptop_test_pkl']

    elif args.get('category') == 'rest':
        if split == 'train':
            input = config['rest_train_dataset_txt']
            output = config['rest_train_pkl']

        elif split == 'test':
            input = config['rest_test_dataset_txt']
            output = config['rest_test_pkl']

    aspects, contexts, labels, len_aspects, len_contexts = [] ,[], [], [], []
    file = open(input, 'r')
    lines = file.readlines()

    for i in range(0, len(lines), 3):
        polarity = lines[i+2].split()[0]
        if polarity != 'conflict':
            labels.append(convert_polarity(polarity))
            
            aspect_tokens = item_gen(word2id, lines[i+1])
            aspect_tokens = pad(aspect_tokens, max_aspect_len)

            context_tokens = item_gen(word2id, lines[i])
            context_tokens = pad(context_tokens, max_context_len)

            aspects.append(aspect_tokens)
            contexts.append(context_tokens)

                
            len_aspects.append(len(aspect_tokens))
            len_contexts.append(len(context_tokens) - 1)

    save_file(aspects, contexts, labels, len_aspects, len_contexts, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='rest')
    
    config = parser.parse_args()
    config = config.__dict__

    data_info(config)
