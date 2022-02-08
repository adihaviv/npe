import json
import argparse
import os

DEBUG = False

def read_pile(pile_paths, filters):
    examples = []
    log_counter = 0
    for pile_path in pile_paths:
        with open(pile_path) as f:
            for line in f:
                example = json.loads(line)
                assert len(example['meta']) == 1
                meta = example['meta']['pile_set_name'].lower()
                if meta not in filters:
                    examples.append(example['text'])

                log_counter += 1
                if log_counter % 10000 == 0:
                    print("processed {} examples".format(log_counter))

                if DEBUG and log_counter > 12345:
                    break

    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths",
                        default=r"/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/01.jsonl,"
                                r"/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/00.jsonl",
                        type=str,
                        required=False)
    parser.add_argument("--filters",
                        default="github, stackexchange",
                        type=str,
                        required=False)
    parser.add_argument("--out_dir_raw",
                        default="/home/olab/adi/data/e3po/pile",
                        type=str,
                        required=False)

    parser.add_argument("--out_dir_proc",
                        default="/home/olab/adi/git/fairseq-2/data-bin/pile",
                        type=str,
                        required=False)


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # read all data, filter and split to train/val/test
    examples = read_pile(args.paths.split(","), args.filters.split(","))
    exp_cnt = len(examples)

    train = examples[:round(exp_cnt*0.8)]
    val = examples[round(exp_cnt*0.8)+1:round(exp_cnt*0.9)]
    test = examples[round(exp_cnt * 0.9)+1:]
    print("train set size:{}, validation set size:{}, test set size:{}".format(len(train), len(val), len(test)))

    out_dir = args.out_dir_raw
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    pile_prefix = "pile_00_01"

    print("writing train file")
    train_file_path = os.path.join(out_dir, pile_prefix+'.train.tokens')
    counter = len(train)
    with open(train_file_path, 'w') as f:
        for item in train:
            f.write("%s\n" % item)
            counter -= 1
            if counter % 10000 == 0:
                print("wrote {} examples in train set".format(counter))

    print("writing validation file")
    counter = len(val)
    valid_file_path = os.path.join(out_dir, pile_prefix+'.valid.tokens')
    with open(valid_file_path, 'w') as f:
        for item in val:
            f.write("%s\n" % item)
            counter -= 1
            if counter % 10000 == 0:
                print("wrote {} examples in validation set".format(counter))

    print("writing test file")
    counter = len(test)
    test_file_path = os.path.join(out_dir, pile_prefix+'.test.tokens')
    with open(test_file_path, 'w') as f:
        for item in test:
            f.write("%s\n" % item)
            counter -= 1
            if counter % 10000 == 0:
                print("wrote {} examples in test set".format(counter))

    print("all done - you can now run: \n "
          f"fairseq-preprocess --only-source --trainpref {train_file_path} --validpref {valid_file_path} "
          f"--testpref {test_file_path} --destdir {args.out_dir_proc} --workers 20")




