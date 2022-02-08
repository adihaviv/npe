import os
from sklearn.model_selection import ParameterGrid
import namegenerator
import string

# To match the performance of relative_position_bias we need to create a batch size of 9216 (max tokens) * 8 (gpus)
# (total of 73728 tokens\loss steps)
if __name__ == '__main__':
    num_of_gpus = 1
    max_seq_len = 1024
    experiment = "vanilla-e3po-baseline"
    cp_path = "/home/olab/adi/experiments/e3po/vanilla-e3po-baseline-8gpu-distribution-anybody/" \
              "vanilla-e3po-baseline-8gpu-distribution-anybody/checkpoint_best.pt"

    slurm_out_dir = r"/home/olab/adi/experiments/e3po/slurm_scripts/eval/"
    slurm_output_dir = os.path.join(slurm_out_dir, experiment)

    slurm_template = "sbatch --job-name=" + experiment + \
                     " --output=" + os.path.join(slurm_output_dir, "slurm.out") + \
                     " --error=" + os.path.join(slurm_output_dir, "slurm.err") + \
                     " --partition=killable --time=5000 --signal=USR1@120 --nodes=1" \
                     " --ntasks=1 --mem=50000 --cpus-per-task=4 --gpus=1 "

    slurm_template += " --exclude=n-101,n-201 "

    save_dir = os.path.join("/home/olab/adi/experiments/e3po/eval/")
    save_dir = os.path.join(save_dir, experiment)
    python_command_template_params = "/home/gamir/adi/miniconda3/envs/e3po/bin/python " \
                                     "/home/olab/adi/git/e3po/fairseq_cli/eval_lm.py " \
                                     "/home/olab/adi/git/e3po/fairseq_cli/data-bin/wikitext-103 " \
                                     "--path "+cp_path+" " \
                                     "--max_tokens 9216 " \
                                     "--tokens-per-sample "+str(max_seq_len)+" --wandb-project e3po "
                                     #"--context-window 400" \


    if "a100" in experiment:
        python_command_template_params += " --update-freq 1 --fp16"
    else:
        python_command_template_params += " --update-freq " + str(8//num_of_gpus) + " --max-tokens 9216 --fp16 "

    #if "vanilla" not in experiment and "abs_pos" not in experiment:
    #    python_command_template_params += " --no-token-positional-embeddings "

    if "alibi" in experiment:
        python_command_template_params += " --relative-bias-fn alibi "

    elif "gaussian" in experiment:
        python_command_template_params += " --relative-bias-fn gaussian "

    elif "sanity" in experiment:
        python_command_template_params += " --relative-bias-fn sanity "

    if not os.path.exists(slurm_output_dir):
        os.makedirs(slurm_output_dir, exist_ok=True)

    all_jobs_file_path = os.path.join(slurm_output_dir, "run_all_slurm_jobs.sh")
    with open(all_jobs_file_path, "w") as all_f:
        os.chmod(all_jobs_file_path, 0o777)
        all_f.write("# !/bin/bash\n\n")
        job_command = python_command_template_params
        job_file_path = os.path.join(slurm_output_dir,experiment+".sh")
        slurm_command = slurm_template

        with open(job_file_path, "w") as f:
            os.chmod(job_file_path, 0o777)
            f.write("#!/bin/bash\n\n")
            f.write(job_command + "\n")

        all_f.write(slurm_command + "\n")

    print("main output folder: {}".format(slurm_output_dir))
