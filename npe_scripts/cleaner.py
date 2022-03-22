import os
import argparse

import shutil


def delete_checkpoints(path):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path)
              for f in filenames if "checkpoint" in f
                                    and "last" not in f
                                    and "best" not in f]
    for c, file in enumerate(result):
        if c % 100 == 0:
            print("{}/{}:".format(c, len(result)))

        os.remove(file)
    print("done!")


def delete_wanb(path):
    wandb_dirs = [dp for dp, dn, filenames in os.walk(path) for dir in dn if "wandb" in dp]
    for c, dir in enumerate(wandb_dirs):
        if c % 100 == 0:
            print("{}/{}:".format(c, len(wandb_dirs)))

        if os.path.exists(dir):
            shutil.rmtree(dir)
    print("done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",
                        default=r"/home/olab/adi/experiments/npe",
                        type=str,
                        required=False)

    parser.add_argument("--root_dir_wandb",
                        default=r"/home/olab/adi/experiments/npe/slurm_scripts/",
                        type=str,
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    delete_checkpoints(args.root_dir)
    delete_wanb(args.root_dir_wandb)
