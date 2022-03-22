import argparse
import boto3
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_file_path",
                        default=r"/home/olab/adi/git/npe/data-bin/pile/pile_orig",
                        type=str,
                        required=False)

    parser.add_argument("--s3_filename",
                        default=r"pile_ds.zip",
                        type=str,
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.local_file_path):
        raise ValueError("file not exist")

    s3_client = boto3.client('s3')
    s3_client.upload_file(args.local_file_path, "pile-npe", args.s3_filename, ExtraArgs={'ACL': 'public-read'})
