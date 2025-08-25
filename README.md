# JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering

[![ACM MM 2025](https://img.shields.io/badge/ACM%20MM-2025-blue)](https://2025.acmmm.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

**JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering** (ACM MM'25 Accepted)

This repository contains the official implementation of JPS, a novel jailbreak method that combines collaborative visual perturbation and textual steering to enhance the quality and specificity of responses from multimodal large language models (MLLMs). Unlike traditional jailbreak methods that only focus on bypassing safety mechanisms, JPS prioritizes response quality and adherence to user intent while providing detailed, relevant outputs.

## 🚀 Key Features

- **Collaborative Attack Strategy**: Combines visual perturbation with textual steering for enhanced effectiveness
- **Quality-Focused Approach**: Emphasizes response quality and relevance rather than simple bypassing
- **Strong Generalization**: Optimized on only 50 samples from AdvBench-subset, yet transfers effectively to other datasets
- **Multi-Agent Framework**: Utilizes revision agents to iteratively improve steering prompts
- **Comprehensive Evaluation**: Includes MIFR quality evaluation metrics
- **Batch Processing Support**: Efficient configuration management and parallel execution

## 🔄 Method Overview

![JPS Method Overview](figs/Method.png)

*Figure: Overview of the JPS (Jailbreak with Collaborative Visual Perturbation and Textual Steering) method framework. The approach combines visual perturbation optimization with multi-agent textual steering to generate high-quality responses from multimodal large language models.*

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🛠️ Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU
- At least 24GB GPU memory (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-repo/JPS.git
cd JPS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure paths in `config_batch_process/` directory:
```bash
cd config_batch_process
# Edit the YAML files to set correct paths
bash config_process.sh
```

## 🚀 Quick Start

The JPS method works in two phases: **Attack** and **Inference**.

### Phase 1: Attack (Optimization)

First, optimize steering prompts and adversarial images on AdvBench-subset:

```bash
python run_config.py --config_dir './config/advbench_subset' --type attack --gpus 4,5,6,7
```

### Phase 2: Inference (Transfer)

Then apply the optimized components to other datasets:

```bash
# Run inference on different datasets
python run_config.py --config_dir './config/advbench' --type inference --gpus 4,5,6,7
python run_config.py --config_dir './config/mmsafetybench' --type inference --gpus 4,5,6,7
python run_config.py --config_dir './config/harmbench' --type inference --gpus 4,5,6,7
```

### Alternative: Single Configuration

For single configuration runs:

```bash
# Attack phase
python main.py --config config/path/to/config.yaml --run_type attack

# Inference phase  
python main.py --config config/path/to/config.yaml --run_type inference
```

## 📁 Project Structure

```
JPS/
├── main.py                     # Main entry point
├── run_config.py              # Batch configuration runner
├── run_config.sh              # Example execution script
├── quality_evaluate.py        # MIFR quality evaluation
├── requirements.txt           # Dependencies
├── config/                    # Configuration files
│   ├── advbench/             # AdvBench dataset configs
│   ├── advbench_subset/      # AdvBench subset configs
│   ├── mmsafetybench/        # MMSafetyBench configs
│   └── harmbench/            # HarmBench configs
├── config_batch_process/      # Batch processing utilities
│   ├── config_process.sh     # Batch configuration script
│   ├── batch_modify_yaml.py  # YAML modification tool
│   └── *.yaml                # Template configurations
├── attack/                    # Attack implementations
│   ├── attack_factory.py     # Attack factory
│   ├── base_attack.py        # Base attack class
│   ├── internvl2_8b_attack.py
│   ├── minigpt4_13b_attack.py
│   └── qwen2vl_7b_attack.py
├── model/                     # Model implementations
│   ├── model_factory.py      # Model factory
│   ├── base_model.py         # Base model class
│   ├── internvl2_8b_model.py
│   ├── minigpt4_13b_model.py
│   └── qwen2vl_7b_model.py
├── dataset/                   # Dataset loaders
│   ├── data_loader_factory.py
│   ├── advbench_loader.py
│   ├── mmsafetybench_loader.py
│   └── harmbench_loader.py
├── preprocess/                # Image preprocessing
│   ├── preprocessor_factory.py
│   └── *_preprocessor.py
├── revision/                  # Multi-agent revision system
│   ├── multi_agent_group.py  # Multi-agent coordinator
│   ├── single_agent_group.py # Single-agent alternative
│   ├── revision_agent.py     # Core revision agent
│   ├── prompt_feedback_agent.py
│   ├── response_feedback_agent.py
│   └── summarize_agent.py
├── judge/                     # Evaluation judges
│   ├── judge_factory.py
│   ├── gpt4o_mini_judge.py
│   ├── quality_judge.py
│   └── harmbench_judge.py
└── results/                   # Output directory
    ├── images/               # Adversarial images
    ├── generate/             # Generated responses
    ├── eval/                 # Evaluation results
    └── qwen2vl_7b/          # Example: Qwen2-VL-7B optimization results
        ├── *_iteration0_.png    # Initial adversarial image
        ├── *_iteration1_.png    # Refined adversarial image
        ├── *_iteration1_.txt    # First steering prompt
        ├── ...                  # Progressive iterations
        ├── *_iteration5_.png    # Final adversarial image
        └── *_iteration5_.txt    # Final steering prompt
```

## ⚙️ Configuration

### Batch Configuration Management

JPS provides convenient batch configuration tools:

1. **Navigate to config directory**:
```bash
cd config_batch_process
```

2. **Modify configuration templates**:
   - Edit `model_*.yaml` for model settings
   - Edit `dataset_*.yaml` for dataset paths
   - Edit `common.yaml` for shared settings

3. **Apply configurations**:
```bash
bash config_process.sh
```

This will automatically update all configuration files with your paths and settings.

### Key Configuration Parameters

- **attack_config**: Visual perturbation settings
  - `attack_steps`: Number of optimization steps
  - `learning_rate`: Perturbation learning rate
  - `perturbation_constraint`: L∞ constraint for perturbations
  - `revision_iteration_num`: Number of revision iterations

- **model**: Target model configuration
  - `name`: Model identifier (internvl2, qwen2vl, minigpt4)
  - `model_path`: Path to model weights
  - `agent_model_path`: Path to revision agent model

- **inference_config**: Generation settings
  - `batch_size`: Inference batch size
  - `max_new_tokens`: Maximum generated tokens

## 📖 Usage

### Attack Types

JPS supports multiple perturbation strategies:

- **Full**: Full-image perturbation
- **Patch**: Patch-based perturbation
- **Grid**: Grid-based perturbation

### Multi-Agent vs Single-Agent

Choose between multi-agent and single-agent revision:

```bash
# Multi-agent (default, better performance)
python main.py --config config.yaml --multi_agent 1

# Single-agent (faster, lower memory)
python main.py --config config.yaml --multi_agent 0
```

### Revision Process

The revision process iteratively improves steering prompts through:

1. **Response Analysis**: Analyzing model responses to identify weaknesses
2. **Prompt Revision**: Using specialized agents to improve steering prompts
3. **Visual Update**: Re-optimizing adversarial images with new prompts
4. **Quality Assessment**: Evaluating improvements in response quality

## 📊 Evaluation

### Quality Evaluation

Evaluate response quality using MIFR metrics:

```bash
python quality_evaluate.py --input_file results/generate/responses.json
```

### Attack Success Rate (ASR)

ASR is automatically computed during evaluation using configured judges:

- GPT-4o Mini Judge
- LlamaGuard3 Judge  
- HarmBench Judge

## 🤖 Supported Models

Currently supported multimodal models:

- **InternVL2-8B**: Advanced vision-language model
- **Qwen2-VL-7B**: Qwen series multimodal model
- **MiniGPT4-13B**: Compact multimodal model

## 📚 Datasets

Supported evaluation datasets:

- **AdvBench**: Standard jailbreak evaluation benchmark
- **AdvBench-Subset**: 50-sample optimization subset (available at [TAP Repository](https://github.com/RICommunity/TAP/blob/main/data/advbench_subset.csv))
- **MMSafetyBench**: Multimodal safety evaluation (available at [MM-SafetyBench Repository](https://github.com/isXinLiu/MM-SafetyBench))
- **HarmBench**: Comprehensive harm evaluation (available at [HarmBench Repository](https://github.com/centerforaisafety/HarmBench))

## 📈 Results

JPS demonstrates:

- **Superior Quality**: Higher MIFR scores compared to existing methods
- **Strong Generalization**: Effective transfer from 50 training samples
- **Efficient Optimization**: Faster convergence with multi-agent revision
- **Robust Performance**: Consistent results across different models and datasets

### Example Output Files

We provide example results from Qwen2-VL-7B optimization in `results/qwen2vl_7b/`:

- **Adversarial Images**: `full_con32|255_lr1|255_revision5_iterationX_.png` - Visual perturbations for each iteration
- **Steering Prompts**: `full_con32|255_lr1|255_revision5_iterationX_.txt` - Textual steering prompts refined through multi-agent revision

The files demonstrate the iterative refinement process:
- `iteration0`: Initial adversarial image (no steering prompt yet)
- `iteration1-5`: Progressive improvement of both visual perturbations and textual steering
- Final steering prompts become increasingly sophisticated and effective

These files can be directly used for inference on new questions by:
1. Using the final adversarial image (`iteration5_.png`)
2. Prepending the steering prompt (`iteration5_.txt`) to your target questions

## 🔧 Advanced Usage

### Target Generation for Custom Datasets

We provide `goals-and-targets-detailed.json` in the main directory as an example of structured goal-target pairs. If you want to generate targets for your own dataset or questions, you can reference `goal_construction.py`.

### Custom Datasets

To add a new dataset:

1. Create a loader in `dataset/your_dataset_loader.py`
2. Register in `dataset/data_loader_factory.py`
3. Add configuration template in `config_batch_process/`

### Custom Models

To add a new model:

1. Implement model class in `model/your_model.py`
2. Create corresponding attack in `attack/your_model_attack.py`
3. Add preprocessor in `preprocess/your_model_preprocessor.py`
4. Register in respective factory files

### Logging and Monitoring

All experiments are automatically logged with timestamps:

```bash
# View logs
ls process_logs/
tail -f process_logs/latest.log
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in configuration
2. **Model Loading Errors**: Check model paths in config files
3. **Permission Errors**: Ensure write permissions for results directory

### Performance Optimization

- Use multiple GPUs with `run_config.py`
- Adjust batch sizes based on available memory
- Enable mixed precision for faster training

## 📖 Citation

If you find our work helpful, please consider citing our paper:

```bibtex
@article{chen2025jps,
  title={JPS: Jailbreak Multimodal Large Language Models with Collaborative Visual Perturbation and Textual Steering},
  author={Chen, Renmiao and Cui, Shiyao and Huang, Xuancheng and Pan, Chengwei and Huang, Victor Shea-Jay and Zhang, QingLin and Ouyang, Xuan and Zhang, Zhexin and Wang, Hongning and Huang, Minlie},
  journal={arXiv preprint arXiv:2508.05087},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This tool is intended for research purposes only. Users are responsible for ensuring ethical and legal compliance when using this software. The authors do not condone malicious use of this technology.

---

**Note**: This research contributes to understanding multimodal model vulnerabilities and developing more robust AI safety mechanisms. Please use responsibly.

