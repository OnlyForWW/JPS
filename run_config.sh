#!/bin/bash

gpus=4,5,6,7

# First, you need to carry out an attack on advbench_subset to obtain the steering_prompt and adv_image.
python run_config.py --config_dir './config/advbench_subset' --gpus attack $gpus

# Then you can reason about questions on other datasets.
python run_config.py --config_dir './config/advbench' --type inference --gpus $gpus
python run_config.py --config_dir './config/mmsafetybench' --type inference --gpus $gpus
python run_config.py --config_dir './config/harmbench' --type inference --gpus $gpus