import yaml
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

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

    for i in tqdm(range(0, len(lines), 3)):
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



def get_word_info(category):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if category == 'laptop':
        filename_train = config['laptop_train_dataset_txt']
        filename_test = config['laptop_test_dataset_txt']

    elif category == 'rest':
        filename_train = config['rest_train_dataset_txt']
        filename_test = config['rest_test_dataset_txt']

    word2id = {}
    word2id['<pad>'] = 0

    list_of_words = []

    extraction(list_of_words, filename_train)
    extraction(list_of_words,filename_test)

    word_count = Counter(list_of_words).most_common()
    for word, freq in word_count:
        if word not in word2id:
            if ' ' not in word and '\n' not in word:
#                 if 'aspect_term' not in word:
                word2id[word] = len(word2id)

    return word2id


def load_word_embeddings(dim, word2id):
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    oov = len(word2id)
    word2vec = np.random.uniform(-0.01, 0.01, [oov, dim])
    word2vec[word2id['<pad>'], :] = 0
    filename = config['glove']
    file = open(filename, 'r', encoding='utf-8')
    lines = file.readlines()

    for line in tqdm(lines):
        line = line.split(' ')
        word = line[0]
        if word in word2id:
            oov -= 1
            key = word2id[word]
            word2vec[key] = np.array([float(x) for x in line[1:]])

    return word2vec

