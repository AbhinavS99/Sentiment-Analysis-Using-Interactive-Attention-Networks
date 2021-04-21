import yaml
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

def process_dataset(args):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.get('category') == 'rest':
        if args.get('split') == 'train':
            input = config['rest_train_dataset_xml']
            output = config['rest_train_dataset_txt']
        elif args.get('split') == 'test':
            input = config['rest_test_dataset_xml']
            output = config['rest_test_dataset_txt']

    elif args.get('category') == 'laptop':
        if args.get('split') == 'train':
            input = config['laptop_train_dataset_xml']
            output = config['laptop_train_dataset_txt']
        elif args.get('split') == 'test':
            input = config['laptop_test_dataset_xml']
            output = config['laptop_test_dataset_txt']

    holder = []
    tree = ET.parse(input)
    root = tree.getroot()

    for sentence in tqdm(root.iter('sentence')):
        text = sentence.find('text').text
        for asp_terms in sentence.iter('aspectTerms'):
            for asp_term in asp_terms.findall('aspectTerm'):    
                if asp_term.get('polarity') != 'conflict' and asp_term.get('term') != 'NULL':
                    term = asp_term.get('term')
                    polarity = asp_term.get('polarity')
                    gen_text = text
                    case = (gen_text, term, polarity)
                    holder.append(case)

    with open(output, 'w') as op:
        for gen_text, term, polarity in holder:
            op.write(gen_text+"\n")
            op.write(term+"\n")
            op.write(polarity+"\n")
    
    print("[+] Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='rest')
    parser.add_argument('--split', default='train')

    config = parser.parse_args()
    config = config.__dict__
    
    process_dataset(config)
