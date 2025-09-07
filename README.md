# LMM4Edit
[ACM MM 2025] [LMM4Edit: Benchmarking and Evaluating Multimodal Image Editing with LMMs](https://www.arxiv.org/abs/2507.16193)

## 📦 Setup

1. **Unzip the framework and install dependencies**
```bash
unzip ms-swift.zip
cd ./ms-swift-3.2.0
pip install -e .
pip install torchvision qwen_vl_utils decord
unzip transformers.zip
cd ./transformers
pip install -e .
```
2. **Download Weights**

Download the Qwen2.5-VL pretrained weights and place them in:
```bash
./weights/qwen2_5
```
## 🚀 Training
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model_type qwen2_5_vl \
  --model ./weights/qwen2_5 \
  --dataset ./data/train_v.json \
  --val_dataset ./data/test_v.json \
  --max_length 4096 \
  --num_train_epochs 2 \
  --save_steps 16 \
  --eval_steps 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --freeze_llm false \
  --freeze_vit false
```

## 🧪 Evaluation
```bash
python evaluate.py \
  --model_path ./weights/qwen2_5 \
  --ckpt_path ./weights/checkpoints/model_weights_v.pth \
  --val_dataset ./data/test_v.json \
  --output_json /path/to/output_predictions.json \
  --QA False
```

## 📁 Resources

📄 Dataset: [Link](https://pan.baidu.com/s/1x1QHFNC6Kz_-X44QyoQTsQ?pwd=kxyt)

🎯 Pretrained Weights: [Link](https://pan.baidu.com/s/1vrpqazjq3Ah8N6vlT2A-mQ?pwd=y73e)

## 🎓Citations
If you find our work useful, please cite our paper as:
```bash
@misc{xu2025lmm4editbenchmarkingevaluatingmultimodal,
      title={LMM4Edit: Benchmarking and Evaluating Multimodal Image Editing with LMMs}, 
      author={Zitong Xu and Huiyu Duan and Bingnan Liu and Guangji Ma and Jiarui Wang and Liu Yang and Shiqi Gao and Xiaoyu Wang and Jia Wang and Xiongkuo Min and Guangtao Zhai and Weisi Lin},
      year={2025},
      eprint={2507.16193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.16193}, 
}
```


