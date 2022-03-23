import os
import argparse
from tqdm import tqdm
import shutil
import json


def get_pe_from_name(path):
    if "-npe-" in path:
        return "npe"
    elif "-alibi-" in path:
        return "alibi"
    elif "-learned-" in path:
        return "learned"
    elif "-sinusoidal-" in path:
        return "sinusoidal"
    else:
        raise Exception("file name should include npe, alibi, learned or sinusoidal.\n{}".format(path))


def get_parameter_value(parameters_str, key):
    key = "'"+key+"':"
    a = parameters_str[parameters_str.find(key) + len(key):]
    return a[:a.find(",")]


def get_experiment_params(lines):
    first_line = lines[0]
    first_line_prefix = first_line.find("fairseq_cli.train |")
    parameters_str = first_line[first_line_prefix + len("fairseq_cli.train |"):]
    return parameters_str


def get_validation_value(key, validation_parts):
    for part in validation_parts:
        if key in part:
            return part.replace(key, "").strip()
    raise Exception("{} not found in validation stats:{}.".format(key, validation_parts))


def get_last_validation_data(lines):
    last_validation_line_ct = 1
    while "valid on 'valid' subset |" not in lines[-last_validation_line_ct] and \
            last_validation_line_ct < len(lines):
        last_validation_line_ct += 1
    validation_line = lines[-last_validation_line_ct]
    return validation_line


def get_best_validation_data(metric, lines):
    best_mad = 10000.0
    best_validation_line = lines[0]
    for line in lines:
        if "valid on 'valid' subset |" in line and metric in line:
            mad_val = get_validation_value(metric, line.split("|"))
            if float(mad_val) < best_mad:
                best_mad = float(mad_val)
                best_validation_line = line

    return best_validation_line


def get_stats(path):
    res_files = [os.path.join(path, f) for f in tqdm(os.listdir(path)) if f.endswith(".out")]

    with open(os.path.join(path, "stats_last.txt"), "w") as out_fl:
        out_fl.write("Experiment\tLayer\tMAD\tAccuracy\tPPL\n")
        for file in tqdm(res_files):
            pe = get_pe_from_name(os.path.basename(file))
            with open(file, "r", encoding="latin-1") as in_f:
                lines = in_f.readlines()
                parameters_str = get_experiment_params(lines)
                #parameters = json.load(parameters_str)
                layer_id = get_parameter_value(parameters_str, "probe_layer_idx")

                last_validation_stats = get_last_validation_data(lines).split("|")
                ppl = get_validation_value("ppl", last_validation_stats)
                acc = get_validation_value("accuracy", last_validation_stats)
                mad = get_validation_value("mad", last_validation_stats)
                out_fl.write("{}\t{}\t{}\t{}\t{}\n".format(pe, layer_id, mad, acc, ppl))

    if args.best_metric:
        with open(os.path.join(path, "stats_best.txt"), "w") as out_fb:
            out_fb.write("Experiment\tLayer\tMAD\tAccuracy\tPPL\n")
            for file in tqdm(res_files):
                pe = get_pe_from_name(os.path.basename(file))
                with open(file, "r", encoding="latin-1") as in_f:
                    lines = in_f.readlines()
                    parameters_str = get_experiment_params(lines)
                    #parameters = json.load(parameters_str)
                    layer_id = get_parameter_value(parameters_str, "probe_layer_idx")

                    best_validation_stats = get_best_validation_data(args.best_metric, lines).split("|")
                    ppl = get_validation_value("ppl", best_validation_stats)
                    acc = get_validation_value("accuracy", best_validation_stats)
                    mad = get_validation_value("mad", best_validation_stats)
                    out_fb.write("{}\t{}\t{}\t{}\t{}\n".format(pe, layer_id, mad, acc, ppl))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        default=r"/home/olab/adi/experiments/npe/slurm_scripts/dpp-alibi-baevski-wiki103-512-mad-advertising-print",
                        type=str,
                        required=False)

    parser.add_argument("--best_metric",
                        default=None,
                        type=str,
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    get_stats(args.dir)
    print("Done")