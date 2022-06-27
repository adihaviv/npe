import json
import argparse
import os
import random

"""
How to recreate the pile dataset
https://github.com/oriyor/scrolls_pt_data/tree/main

Create pre-training data
(1)Download and de-compress the pile chunks, download and format wikipedia
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/00.jsonl
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/01.jsonl
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/wikipedia.jsonl

(2) Preprocess the data, getting rid of specific sources, trimming outliers, replacing special chars, output tsv per source, plot stats.
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/00_filtered.jsonl # without some sources ('Wikipedia (en)', "Github", "DM Mathematics", "Ubuntu IRC")
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/01_filtered.jsonl # without some sources ('Wikipedia (en)', "Github", "DM Mathematics", "Ubuntu IRC")
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/wikipedia_filtered.jsonl # this is wikipedia twice


----------------------- this is not relevant ------------
all the per source files are in:
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/1/txts/
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/2/txts/
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/3/txts/
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/4/txts/
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/5/txts/
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/6/txts/

(3) Sample 3% of the data for learning a tokenizer
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/091821/scrolls_corpus_sample.txt
/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/091821/scrolls_corpus_sample_stats.txt

(4) Run BPE learning on 
 python preprocess.py --only-source 
 --trainpref /home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/091821/scrolls_corpus_sample.txt    
 --destdir /home/olab/adi/git/npe/data-bin/pile/dict.txt     
 --workers 20 --bpe gpt2 --task language_modeling --dict-only --fp16

------------------------------------------------------------

(3) Next encode it with the GPT-2 BPE:
cd data-bin/pile/pile_orig
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
Finally preprocess/binarize the data using the GPT-2 fairseq dictionary:

wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60




 
(5) Unify all the files, flatten them and split to train, dev, test
** Unify all files (00,01,wikipedia)
** split to train, dev(2000), test(2000)
** flatten to txt 
** outputs the command for (6)

run pile_dataset_creator.py


(6) Preprocess datasets with the created dictionary
fairseq-preprocess --only-source --trainpref /home/olab/adi/git/npe/data-bin/pile/pile-tokenized/pile_00_01_wikipedia.train.tokens.bpe --validpref /home/olab/adi/git/npe/data-bin/pile/pile-tokenized/pile_00_01_wikipedia.valid.tokens.bpe --testpref /home/olab/adi/git/npe/data-bin/pile/pile-tokenized/pile_00_01_wikipedia.test.tokens.bpe --destdir /home/olab/adi/git/npe/data-bin/pile/pile-fixed --workers 20  --srcdict /home/olab/adi/git/npe/data-bin/pile/pile-fixed/dict.txt --fp16

"""


DEBUG = False

def read_pile(pile_paths, filters):
    examples = []
    log_counter = 0
    yt_counter = 0
    for pile_path in pile_paths:
        with open(pile_path) as f:
            for line in f:
                example = json.loads(line)
                assert len(example['meta']) == 1
                meta = example['meta']['pile_set_name'].lower()
                #print(meta)
                if meta == "youtube":
                  yt_counter += 1
                  print("found one")
                if meta not in filters:
                    examples.append(example['text'])
                log_counter += 1
                if log_counter % 10000 == 0:
                    print("filtered {} youtube examples".format(yt_counter))
                    print("processed {} examples".format(log_counter))

                if DEBUG and log_counter > 12345:
                    log_counter = 0
                    break

    return examples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths",
                        default=r"/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/00_filtered.jsonl,"
                                r"/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/01_filtered.jsonl,"
                                r"/home/olab/urishaham/scrolls/scrolls_pt_data/scrolls_pt_data/scrolls_corpus/wikipedia_filtered.jsonl",
                        type=str,
                        required=False)
    parser.add_argument("--filters",
                        default="github,youtube",
                        type=str,
                        required=False)
    parser.add_argument("--out_dir",
                        default="/home/olab/adi/git/npe/data-bin/pile_no_youtube",
                        type=str,
                        required=False)

    parser.add_argument("--dev_sample_size",
                        default=2000,
                        type=int,
                        required=False)

    return parser.parse_args()


def write_filterd_file(dataset, out_dir, dataset_type):
    print("writing test file")
    counter = len(dataset)
    out_file_path = os.path.join(out_dir, pile_prefix + '.'+dataset_type+'.tokens')
    with open(out_file_path, 'w') as f:
        for item in dataset:
            f.write("%s\n" % item)
            counter -= 1
            if counter % 10000 == 0 and counter > 0:
                print("wrote {} examples in test set".format(counter))
    return out_file_path


if __name__ == '__main__':
    args = parse_args()

    # read all data, filter and split to train/val/test
    examples = read_pile(args.paths.split(","), args.filters.split(","))
    random.shuffle(examples)
    exp_cnt = len(examples)

    val = examples[:args.dev_sample_size]
    test = examples[args.dev_sample_size:2*args.dev_sample_size]
    train = examples[2*args.dev_sample_size:]
    print("train set size:{}, validation set size:{}, test set size:{}".format(len(train), len(val), len(test)))

    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    pile_prefix = "pile_00_01_wikipedia"

    print("writing train file")
    train_file_path = write_filterd_file(train, out_dir, "train")

    print("writing validation file")
    valid_file_path = write_filterd_file(val, out_dir, "valid")

    print("writing test file")
    test_file_path = write_filterd_file(test, out_dir, "test")

    print("all done - you can now run: \n "
          f"fairseq-preprocess --only-source --trainpref {train_file_path} --validpref {valid_file_path} "
          f"--testpref {test_file_path} --destdir {args.out_dir} --workers 20 --srcdict {args.out_dir}/dict.txt"
          f"--bpe gpt2 --fp16")