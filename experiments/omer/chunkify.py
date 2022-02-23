import sys
from tqdm import tqdm


def main(input_path, output_path, stride):
    docs = read_paragraphs(input_path)
    save_data(output_path, docs, stride)

    
def read_paragraphs(path):
    with open(path) as fin:
        lines = [line.strip().split() for line in tqdm(fin)]
    return lines


def save_data(path, docs, stride):
    with open(path, 'w') as fout:
        for doc in tqdm(docs):
            for start in range(0, len(doc), stride):
                fout.write(' '.join(doc[start:start+stride]))
                fout.write('\n')
            fout.write('\n')


if __name__ == '__main__':
    split = sys.argv[1]
    main(split+'.bpe', split+'.bpe.chunks', 255)
