# LMM4Edit
[ACM MM 2025] LMM4Edit: Benchmarking and Evaluating Multimodal Image Editing with LMMs

# MS-Swift 3.2.0: Multimodal Fine-tuning Framework

This repository provides code for training and evaluating multimodal large models, such as Qwen2.5-VL, using the MS-Swift fine-tuning framework.

---

## 📦 Setup

1. **Unzip the framework and install dependencies**
unzip ms-swift.zip
cd ./ms-swift-3.2.0
pip install -e .
pip install torchvision qwen_vl_utils decord
unzip transformers.zip
cd ./transformers
pip install -e .

2. **Download Weights**
Download the Qwen2.5-VL pretrained weights and place them in:
./weights/qwen2_5

##  🏋️ Training
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model_type qwen2_5_vl \
  --model /path/to/qwen2_5_vl_model \
  --dataset /path/to/train_dataset.json \
  --val_dataset /path/to/val_dataset.json \
  --max_length 4096 \
  --num_train_epochs 2 \
  --save_steps 16 \
  --eval_steps 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --freeze_llm false \
  --freeze_vit false
