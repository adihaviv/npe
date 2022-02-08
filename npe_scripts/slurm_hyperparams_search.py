import os
from sklearn.model_selection import ParameterGrid
import namegenerator
import string

# To match the performance of relative_position_bias we need to create a batch size of 9216 (max tokens) * 8 (gpus)
# (total of 73728 tokens\loss steps)
if __name__ == '__main__':
    num_of_gpus = 8
    conda_env = "e3po10"

    #experiment = "npe-" + namegenerator.gen(n=2) + "-position_probe-gpt-3090-depth-exp"
    experiment = "npe-wiki103-vanilla"
    baseline = "baseline" in experiment
    slurm_out_dir = r"/home/olab/adi/experiments/e3po/slurm_scripts/"

    if baseline:
        slurm_out_dir += "baselines/"
    slurm_output_dir = os.path.join(slurm_out_dir, experiment)

    hyperparams = {
        #'e3po-start': [0.25, 2],
        #'e3po-ratio': [2],
        'no-token-positional-embeddings': [True],
        #'tokens-per-sample': [64, 128, 256, 512, 1024, 2040],
        'tokens-per-sample': [64],
        #'arch': ['transformer_lm_gpt', 'transformer_lm_gpt2_2', 'transformer_lm_gpt2_4', 'transformer_lm_gpt2_6']

        #'arch': ['transformer_lm_gpt']
        'arch': ['transformer_lm_wiki103']
    }

    is_position_probe_exp = "position_probe" in experiment or "position-probe" in experiment
    #max_updates = 286000 if not is_position_probe_exp else 100000
    max_updates = 286000
    #time = 5000 if not is_position_probe_exp else 1440
    time = 5000

    hp_template = experiment
    slurm_template = "sbatch --job-name=" + experiment + \
                     " --output=" + os.path.join(slurm_output_dir, "slurm_" + hp_template + ".out") + \
                     " --error=" + os.path.join(slurm_output_dir, "slurm_" + hp_template + ".err") + \
                     " --partition=killable --time="+str(time)+" --signal=USR1@120 --nodes=1" +\
                     " --ntasks=1 --mem=50000 --cpus-per-task=4 "

    slurm_template += " --exclude=n-101,n-201 " #,n-301,n-350,n-351 "
    if "3090" in experiment:
        slurm_template += f' --constraint="geforce_rtx_3090" '
    elif "v100" in experiment:
        slurm_template += f' --constraint="tesla_v100" '
    elif "quadro" in experiment:
        slurm_template += f' --constraint="quadro_rtx_8000" '


    if "a100" in experiment:
        num_of_gpus = 2 if "fp32" not in experiment else 4
        slurm_template += " --nodelist=n-401 "

    gpu_loc = experiment.find("gpu")
    if gpu_loc > 0 and (experiment[gpu_loc+3: gpu_loc+4].isnumeric() or experiment[gpu_loc-1: gpu_loc].isnumeric()):
        if experiment[gpu_loc+3: gpu_loc+4].isnumeric():
            num_of_gpus = int(experiment[gpu_loc+3: gpu_loc+4])
        else:
            num_of_gpus = int(experiment[gpu_loc-1: gpu_loc])
    slurm_template += " --gpus=" + str(num_of_gpus) + " "

    slurm_template += os.path.join(slurm_output_dir, hp_template)

    save_dir = os.path.join("/home/olab/adi/experiments/e3po/")
    save_dir = os.path.join(save_dir, experiment)
    save_dir = os.path.join(save_dir, hp_template)
    # "--task language_modeling /home/olab/adi/git/e3po/fairseq_cli/data-bin/wikitext-103 " \
    python_command_template_params = "/home/olab/adi/miniconda3/envs/"+conda_env+"/bin/python " \
                                     "/home/olab/adi/git/e3po/fairseq_cli/train.py " \
                                     "--task language_modeling  /home/olab/adi/git/fairseq-2/data-bin/wikitext-103 " \
                                     "--seed 1 --sample-break-mode none --warmup-init-lr 1e-07 " \
                                     "--skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp " \
                                     "--keep-best-checkpoints 5 --max-update "+str(max_updates) +\
                                     " --required-batch-size-multiple 1 --wandb-project e3po "

    if "gpt" in experiment:
        max_tokens = 2048
        python_command_template_params += " --dropout 0.1 " \
                                          " --optimizer adam --adam-betas '(0.9, 0.98)' " \
                                          " --weight-decay 0.01 --clip-norm 0.0 --lr 0.0005 " \
                                          " --lr-scheduler inverse_sqrt --warmup-updates 4000  " \
                                          " --sample-break-mode none "
    else:
        max_tokens = 9216
        python_command_template_params += " --lr 1.0 --t-mult 2 " \
                                          " --lr-period-updates 270000 --lr-scheduler cosine " \
                                          " --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 " \
                                          " --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 "

    # set loss function
    if "position_probe" in experiment or "position-probe":
        if "gpt" in experiment:
            python_command_template_params += " --criterion position_probe_cross_entropy "
        else:
            python_command_template_params += " --criterion position_probe_adaptive_loss "
    elif "gpt" not in experiment:
        python_command_template_params += " --criterion adaptive_loss "
    else:
        python_command_template_params += " --criterion cross_entropy "

    if "a100" in experiment:
        python_command_template_params += " --update-freq 1 --max-tokens " + str((max_tokens*2)*num_of_gpus) + " "
    else:
        python_command_template_params += " --update-freq " + str(8//num_of_gpus) + " --max-tokens " + str(max_tokens)

    # if "vanilla" not in experiment and "abs_pos" not in experiment:
    #    python_command_template_params += " --no-token-positional-embeddings "

    if "alibi" in experiment:
        python_command_template_params += " --relative-bias-fn alibi --no-token-positional-embeddings"

    elif "gaussian" in experiment:
        python_command_template_params += " --relative-bias-fn gaussian --no-token-positional-embeddings"

    elif "sanity" in experiment:
        python_command_template_params += " --relative-bias-fn sanity --no-token-positional-embeddings"

    if "fp32" not in experiment:
        python_command_template_params += " --fp16 "

    if not os.path.exists(slurm_output_dir):
        os.makedirs(slurm_output_dir, exist_ok=True)

    all_jobs_file_path = os.path.join(slurm_output_dir, "run_all_slurm_jobs.sh")
    with open(all_jobs_file_path, "w") as all_f:
        os.chmod(all_jobs_file_path, 0o777)
        all_f.write("# !/bin/bash\n\n")

        param_grid = ParameterGrid(hyperparams)
        for dict_ in param_grid:
            job_command = python_command_template_params
            job_file_path = os.path.join(slurm_output_dir,hp_template)
            slurm_command = slurm_template
            full_save_dir = save_dir
            slurm_out_file = "slurm_" + hp_template
            for key in dict_:
                if "e3po" in key and "e3po" not in experiment:
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
                        slurm_command = slurm_command.replace(slurm_out_file, slurm_out_file + "-" + short_key + "-" + str(dict_[key]))
                        slurm_out_file = slurm_out_file.replace(slurm_out_file, slurm_out_file + "-" + short_key + "-" + str(dict_[key]))

            job_command += " --save-dir " + full_save_dir

            slurm_command += ".sh"
            job_file_path = job_file_path + ".sh"

            with open(job_file_path, "w") as f:
                os.chmod(job_file_path, 0o777)
                f.write("#!/bin/bash\n\n")
                f.write(job_command + "\n")

            all_f.write(slurm_command + "\n")

    print("main output folder: {}".format(slurm_output_dir))
