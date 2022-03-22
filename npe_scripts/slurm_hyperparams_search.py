import os
from sklearn.model_selection import ParameterGrid
import namegenerator
import string

# To match the performance of relative_position_bias we need to create a batch size of 9216 (max tokens) * 8 (gpus)
# (total of 73728 tokens\loss steps)
if __name__ == '__main__':
    num_of_gpus = 1
    conda_env = "npe"

    #experiment = "dpp-learned-baevski-wiki103-" + namegenerator.gen(n=2)
    experiment = "dpp-npe-baevski-wiki103-64-" + namegenerator.gen(n=2) #+"-prob-layer-idx-0"
    #experiment = "lm-learned-baevski-wiki103-512"
    # experiment = "lm-" + namegenerator.gen(n=2) + "-gpt_xl-pile-alibi-1024-bm-a100-btz8"

    debug_mode = "debug" in experiment

    baseline = "baseline" in experiment
    slurm_out_dir = r"/home/olab/adi/experiments/npe/slurm_scripts/"

    if baseline:
        slurm_out_dir += "baselines/"
    slurm_output_dir = os.path.join(slurm_out_dir, experiment)

    hyperparams = {
        'no-token-positional-embeddings': [True],
        #'no-token-positional-embeddings': [False],
        'tokens-per-sample': [64],
        #'tokens-per-sample': [128],
        #'arch': ['transformer_lm_gpt', 'transformer_lm_gpt_2', 'transformer_lm_gpt_4', 'transformer_lm_gpt_6']
        #'arch': ['transformer_lm_gpt_xl'],
        'seed': [1],
        'arch': ['transformer_lmpp_wiki103'],
        #'arch': ['transformer_lm_wiki103'],
        #'probe-layer-idx': [0],
        'probe-layer-idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'lr': [0.0002],
        #'dropout': [0, 0.1, 0.2, 0.3],
        'non-linear-probe': [True]

    }

    is_position_probe_exp = "dpp-" in experiment or "-dpp" in experiment
    max_updates = 286000 if not is_position_probe_exp else 10000
    time = 1440 if not is_position_probe_exp else 60

    if "learned" in experiment:
        hyperparams['decoder_learned_pos'] = [True]
        hyperparams['no-token-positional-embeddings'] = [False]


    if is_position_probe_exp:
        hyperparams['arch'] = ['transformer_lmpp_wiki103']
        if "npe" in experiment:
            pp_model_path = "/home/olab/adi/experiments/npe/lm-baevski-wiki103-512/lm-baevski-wiki103-512-no-token-positional-embeddings/checkpoint_best.pt"
            #pp_model_path = "/home/olab/adi/experiments/e3po/video-pack-no-token-positional-embeddings/video-pack-no-token-positional-embeddings/checkpoint_best.pt"
            hyperparams['no-token-positional-embeddings'] = [True]
        elif "sinusoidal" in experiment:
            pp_model_path = "/home/olab/adi/experiments/e3po/vanilla-e3po-baseline-8gpu-distribution-anybody/vanilla-e3po-baseline-8gpu-distribution-anybody/checkpoint_best.pt"
            hyperparams['no-token-positional-embeddings'] = [False]




    hp_template = experiment
    slurm_template = "sbatch --job-name=" + experiment + \
                     " --output=" + os.path.join(slurm_output_dir, "slurm_" + hp_template + ".out") + \
                     " --error=" + os.path.join(slurm_output_dir, "slurm_" + hp_template + ".err") + \
                     " --partition=killable --time="+str(time)+" --signal=USR1@120 --nodes=1" +\
                     " --ntasks=1 --mem=50000 --cpus-per-task=4 "

    if "3090" in experiment:
        slurm_template += f' --constraint="geforce_rtx_3090" '
    elif "v100" in experiment:
        slurm_template += f' --constraint="tesla_v100" '
    elif "quadro" in experiment:
        slurm_template += f' --constraint="quadro_rtx_8000" '
    elif "a100" in experiment:
        #slurm_template += " --nodelist=n-401 "
        slurm_template += f' --constraint="a100_sxm_80gb" '
    else:
        slurm_template += " --exclude=n-401,n-101,n-201,n-351 " #,n-301,n-350, "

    gpu_loc = experiment.find("gpu")
    if gpu_loc > 0 and (experiment[gpu_loc+3: gpu_loc+4].isnumeric() or experiment[gpu_loc-1: gpu_loc].isnumeric()):
        if experiment[gpu_loc+3: gpu_loc+4].isnumeric():
            num_of_gpus = int(experiment[gpu_loc+3: gpu_loc+4])
        else:
            num_of_gpus = int(experiment[gpu_loc-1: gpu_loc])
    slurm_template += " --gpus=" + str(num_of_gpus) + " "

    slurm_template += os.path.join(slurm_output_dir, hp_template)

    save_dir = os.path.join("/home/olab/adi/experiments/npe/")
    save_dir = os.path.join(save_dir, experiment)
    save_dir = os.path.join(save_dir, hp_template)

    pile_exp = "pile" in experiment
    dataset_path = r"/home/olab/adi/git/npe/data-bin/pile/pile_gpt2_bpe " if pile_exp else r"/home/olab/adi/git/npe/data-bin/wikitext-103 "
    sample_bm = "complete_doc" if "bm-complete_doc" in experiment else "none"

    python_command_template_params = "/home/olab/adi/miniconda3/envs/"+conda_env+"/bin/python " \
                                     "/home/olab/adi/git/npe/fairseq_cli/train.py " \
                                     + dataset_path + \
                                     " --sample-break-mode "+sample_bm+" " \
                                     " --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp " \
                                     " --keep-best-checkpoints 1 --max-update "+str(max_updates) +\
                                     " --required-batch-size-multiple 1 --wandb-project npe " \
    # "--warmup-init-lr 1e-07 "

    # set hyperparams
    if not is_position_probe_exp:
        if pile_exp:
            valid_update_steps = "1000" if not debug_mode else "20"
            python_command_template_params += " --validate-interval-updates " +valid_update_steps+ " --save-interval 1000" \
                                              " --checkpoint-activations --memory-efficient-fp16" \
                                              #" --fp16-adam-stats --combine-valid-subsets --distributed-no-spawn

        if "gpt" in experiment:
            if "complete_doc" in experiment:
                max_tokens = 8192  # btz=8
                #max_tokens = 2048   # btz=2
            else:
                max_tokens = 65536  # btz=64 --> Mem error
                max_tokens = 51200  # btz=50 --> Mem error
                max_tokens = 32768  # btz=32 --> 256
                max_tokens = 16384  # btz=16
                #max_tokens = 8192   # btz=8 --> 64

            python_command_template_params += " --optimizer adam --adam-betas '(0.9, 0.98)' " \
                                              " --weight-decay 0.01 --clip-norm 0.0 " \
                                              " --lr-scheduler polynomial_decay --total-num-update 286102" \
                                              " --warmup-updates 375 --lr 0.0002 "
                                              #--lr 0.0005

        else:
            max_tokens = 9216
            python_command_template_params += " --lr 1.0 --t-mult 2 " \
                                              " --lr-period-updates 270000 --lr-scheduler cosine " \
                                              " --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 " \
                                              " --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 "

    else:
        max_tokens = 9216
        python_command_template_params += " --optimizer adam --adam-betas '(0.9, 0.98)' " \
                                          " --weight-decay 0.01 --clip-norm 0.0 " \
                                          " --lr-scheduler polynomial_decay --total-num-update 286102" \
                                          " --warmup-updates 375 "

    # set task
    if is_position_probe_exp:
        python_command_template_params += " --task language_modeling_position_probe "
    else:
        python_command_template_params += " --task language_modeling "

    # set loss function
    if is_position_probe_exp:
        python_command_template_params += " --criterion dpp_cross_entropy_fix "

        #if "gpt" in experiment:
        #    python_command_template_params += " --criterion position_probe_cross_entropy "
        #else:
        #      python_command_template_params += " --criterion position_probe_adaptive_loss "

    elif "gpt" in experiment:
        python_command_template_params += " --criterion cross_entropy "
    else:
        python_command_template_params += " --criterion adaptive_loss "


    # set checkpoint for position probe model
    if is_position_probe_exp:
        python_command_template_params += " --pretrained-decoder-filename " + pp_model_path + " "
        python_command_template_params += " --validate-interval-updates 100 "
    update_freq = int(8//num_of_gpus)
    python_command_template_params += " --update-freq " + str(update_freq) + " --max-tokens " + str(max_tokens)

    # if "vanilla" not in experiment and "abs_pos" not in experiment:
    #    python_command_template_params += " --no-token-positional-embeddings "

    if "alibi" in experiment:
        python_command_template_params += " --alibi --no-token-positional-embeddings"

    if "fp32" not in experiment:
        python_command_template_params += " --fp16 "

    #if "save-interval" not in python_command_template_params:
    #    if is_position_probe_exp:
    #        python_command_template_params += " --save-interval 2500 "
    #    else:
    #        python_command_template_params += " --save-interval 10000 "

    # save checkpoint after every epoch and only best and last!
    python_command_template_params += " --no-epoch-checkpoints "

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
                if key+" " in python_command_template_params:
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
