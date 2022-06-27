import argparse,os,tqdm


def tokenize_file(in_path, out_dir):
    c = 0
    with open(in_path, 'r') as orig_file:
        with open(out_dir, 'w') as target_file:
            for line in orig_file:
                if len(line.split()) > 3:
                    target_file.write(line)
                c += 1
                if c % 10000 == 0:
                    print(c)

    return out_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path",
                        default="/home/olab/adi/git/npe/data-bin/pile/pile_orig/bpe_files/pile_00_01_wikipedia.train.tokens.bpe",
                        type=str,
                        required=False)
    parser.add_argument("--out_path",
                        default="/home/olab/adi/git/npe/data-bin/pile/pile-tokenized/pile_00_01_wikipedia.train.tokens.bpe",
                        type=str,
                        required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("start")
    tokenize_file(args.in_path, args.out_path)
