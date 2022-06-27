#!/bin/bash

/home/olab/adi/miniconda3/envs/npe/bin/python /home/olab/adi/git/npe/fairseq_cli/train.py  --task language_modeling /home/olab/adi/git/npe/data-bin/pile/data-bin/pile_gpt2_bpe  --sample-break-mode none  --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp  --keep-best-checkpoints 5 --max-update 286000 --required-batch-size-multiple 1 --wandb-project npe  --validate-interval-updates 1000 --save-interval-updates 1000 --checkpoint-activations --memory-efficient-fp16 --optimizer adam --adam-betas '(0.9, 0.98)'  --weight-decay 0.01 --clip-norm 0.0  --lr-scheduler polynomial_decay --total-num-update 286102 --warmup-updates 375 --lr 0.0002  --criterion cross_entropy  --update-freq 4 --max-tokens 32768 --alibi --no-token-positional-embeddings --fp16  --arch transformer_lm_gpt_xl  --seed 42  --tokens-per-sample 1024  --save-dir /home/olab/adi/experiments/npe/lm-marriage-rough-gpt_xl-pile-alibi-a100-1024-short-exp-btz32/lm-marriage-rough-gpt_xl-pile-alibi-a100-1024-short-exp-btz32