#!/bin/bash

# 定义一个列表，其中每个文件名都占一行
list=(
    "dataset_advbench.yaml"
    "dataset_advsubset.yaml"
    "dataset_mmsafety.yaml"
    "dataset_harmbench.yaml"
    "model_internvl2.yaml"
    "model_llava_llama3.yaml"
    "model_qwen2vl.yaml"
    "model_minigpt4.yaml"
    "common.yaml"
)

# list=(
#     "base_config_modify/dataset_advbench.yaml"
#     "base_config_modify/dataset_advsubset.yaml"
#     "base_config_modify/dataset_mmsafety.yaml"
#     "base_config_modify/model_internvl2.yaml"
#     "base_config_modify/model_llava_llama3.yaml"
#     "base_config_modify/model_qwen2vl.yaml"
#     "base_config_modify/with_jb_prompt.yaml"
#     # "base_config_modify/without_jb_prompt.yaml"
#     # "common.yaml"
# )

# 遍历列表中的每个文件
for file in "${list[@]}"; do
    # 执行 Python 脚本并传递 --config 参数
    python batch_modify_yaml.py --config "$file" --add_missing
done