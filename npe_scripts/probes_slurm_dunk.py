from sklearn.model_selection import ParameterGrid
import namegenerator
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=r"data-bin/pile",
                        type=str,
                        required=False)

    parser.add_argument("--slurm_out_dir",
                        default=r"slurm_scripts/",
                        type=str,
                        required=False)

    parser.add_argument("--gpus",
                        default=1,
                        type=int,
                        required=False)

    parser.add_argument("--tokens_per_sample",
                        default=1024,
                        type=int,
                        required=False)

    parser.add_argument("--max_tokens",
                        default=6400,
                        type=int,
                        required=False)

    parser.add_argument("--update_freq",
                        default=8,
                        type=int,
                        required=False)

    parser.add_argument("--max_updates",
                        default=10000,
                        type=int,
                        required=False)

    parser.add_argument("--slurm_time",
                        default=120,
                        type=int,
                        required=False)

    parser.add_argument("--npe_checkpoint",
                        default=r"npe.pt",
                        type=str,
                        required=False)

    parser.add_argument("--alibi_checkpoint",
                        default=r"alibi.pt",
                        type=str,
                        required=False)

    parser.add_argument("--sinusoidal_checkpoint",
                        default=r"baseline.pt",
                        type=str,
                        required=False)

    parser.add_argument("--learned_checkpoint",
                        default=r"learned.pt",
                        type=str,
                        required=False)

    parser.add_argument("--experiment_details",
                        default=r"pile",
                        type=str,
                        required=False)

    parser.add_argument("--fairseq_train_path",
                        default=r"fairseq-train",
                        type=str,
                        required=False)

    parser.add_argument("--save_checkpoint_dir",
                        default=r"probe_checkpoints/",
                        type=str,
                        required=False)

    return parser.parse_args()


def just_do_it(args):
    num_of_gpus = args.gpus
    experiment_name = "dpp-{}-{}".format(args.experiment_details, namegenerator.gen(n=2))
    slurm_output_dir = os.path.join(args.slurm_out_dir, experiment_name)

    hyperparams = {
        'seed': [1],
        'arch': ['transformer_lmpp_gpt3_xl'], #transformer_lmpp_wiki103
        'probe-layer-idx': [i for i in range(25)],
        'lr': [0.002, 0.0002],
        'non-linear-probe': [True, False],
    }

    slurm_template = "sbatch --job-name=" + experiment_name + \
                     " --output=" + os.path.join(slurm_output_dir, "slurm_" + experiment_name + ".out") + \
                     " --error=" + os.path.join(slurm_output_dir, "slurm_" + experiment_name + ".err") + \
                     " --partition=learnlab --time=" + str(args.slurm_time) + " --signal=USR1@120 --nodes=1" + \
                     " --ntasks=1 --mem=50000 --cpus-per-task=4 --constraint=volta32gb --gres=gpu:" + str(num_of_gpus) + " " + slurm_output_dir + "/"

    python_command_template_params = args.fairseq_train_path + " " + " " + args.data_dir + " " \
                                     " --task language_modeling_position_probe --sample-break-mode none " \
                                     " --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp " \
                                     " --keep-best-checkpoints 5 --max-update "+str(args.max_updates)+" --required-batch-size-multiple 1 " \
                                     " --wandb-project npe --validate-interval-updates 100 --checkpoint-activations " \
                                     " --memory-efficient-fp16 --optimizer adam --adam-betas \"(0.9, 0.98)\" " \
                                     " --weight-decay 0.01 --clip-norm 0.0 --lr-scheduler polynomial_decay " \
                                     " --total-num-update 10000 --warmup-updates 500  " \
                                     " --criterion dpp_cross_entropy_fix --max-tokens " + str(args.max_tokens) + \
                                     " --update-freq " + str(args.update_freq) + \
                                     " --tokens-per-sample " + str(args.tokens_per_sample)


    # save checkpoint after every epoch and only best and last!
    python_command_template_params += " --no-epoch-checkpoints "

    save_dir = os.path.join(args.save_checkpoint_dir, experiment_name)
    save_dir = os.path.join(save_dir, experiment_name)

    if not os.path.exists(slurm_output_dir):
        os.makedirs(slurm_output_dir, exist_ok=True)

    all_jobs_file_path = os.path.join(slurm_output_dir, "run_all_slurm_jobs.sh")
    with open(all_jobs_file_path, "w") as all_f:
        os.chmod(all_jobs_file_path, 0o777)
        all_f.write("# !/bin/bash\n\n")

        # AliBi
        alibi_hyperparams = hyperparams
        alibi_hyperparams['no-token-positional-embeddings'] = [True]
        alibi_hyperparams['alibi'] = [True]
        alibi_hyperparams['pretrained-decoder-filename'] = [args.alibi_checkpoint]
        create_scripts(all_f, experiment_name, "alibi", alibi_hyperparams, python_command_template_params, save_dir,
                       slurm_output_dir, slurm_template)

        # NPE
        npe_hyperparams = hyperparams
        npe_hyperparams['no-token-positional-embeddings'] = [True]
        npe_hyperparams['pretrained-decoder-filename'] = [args.npe_checkpoint]
        create_scripts(all_f, experiment_name, "npe", npe_hyperparams, python_command_template_params, save_dir,
                       slurm_output_dir, slurm_template)

        # Learned
        learned_hyperparams = hyperparams
        learned_hyperparams['decoder-learned-pos'] = [True]
        learned_hyperparams['pretrained-decoder-filename'] = [args.learned_checkpoint]
        create_scripts(all_f, experiment_name, "learned", npe_hyperparams, python_command_template_params, save_dir,
                       slurm_output_dir, slurm_template)

        # Sinusoidal
        sin_hyperparams = hyperparams
        sin_hyperparams['pretrained-decoder-filename'] = [args.sinusoidal_checkpoint]
        create_scripts(all_f, experiment_name, "sinusoidal", sin_hyperparams, python_command_template_params, save_dir,
                       slurm_output_dir, slurm_template)

    return  slurm_output_dir


def create_scripts(all_f, experiment_name, pos_method, hyperparams, python_command_template_params, save_dir, slurm_output_dir,
                   slurm_template):
    param_grid = ParameterGrid(hyperparams)
    for dict_ in param_grid:
        job_command = python_command_template_params
        job_file_path = os.path.join(slurm_output_dir, experiment_name)
        slurm_command = slurm_template + experiment_name
        full_save_dir = save_dir

        slurm_out_file = "slurm_" + experiment_name
        slurm_command = slurm_command.replace(slurm_out_file, slurm_out_file + "-" + pos_method)
        slurm_out_file += "-" + pos_method

        for key in dict_:
            if key + " " in python_command_template_params:
                continue

            if type(dict_[key]) is bool:
                if dict_[key]:
                    job_command += " --" + key + " "
                    if len(hyperparams[key]) > 1:
                        full_save_dir += "-" + key
                        job_file_path += "-" + key
                        slurm_command += "-" + key
                        slurm_command = slurm_command.replace(slurm_out_file, slurm_out_file + "-" + key)
                        slurm_out_file = slurm_out_file.replace(slurm_out_file, slurm_out_file + "-" + key)
            else:
                job_command += " --" + key + " " + str(dict_[key]) + " "
                if len(hyperparams[key]) > 1:
                    short_key = key.replace("e3po-", "")
                    full_save_dir += "-" + short_key + "-" + str(dict_[key])
                    job_file_path += "-" + short_key + "-" + str(dict_[key])
                    slurm_command += "-" + short_key + "-" + str(dict_[key])
                    slurm_command = slurm_command.replace(slurm_out_file,
                                                          slurm_out_file + "-" + short_key + "-" + str(dict_[key]))
                    slurm_out_file = slurm_out_file.replace(slurm_out_file,
                                                            slurm_out_file + "-" + short_key + "-" + str(dict_[key]))

        job_command += " --save-dir " + full_save_dir

        slurm_command += ".sh"
        job_file_path = job_file_path + ".sh"

        with open(job_file_path, "w") as f:
            os.chmod(job_file_path, 0o777)
            f.write("#!/bin/bash\n\n")
            f.write(job_command + "\n")

        all_f.write(slurm_command + "\n")


if __name__ == '__main__':
    args = parse_args()
    out_dir = just_do_it(args)
    print("main output directory: {}".format(out_dir))
