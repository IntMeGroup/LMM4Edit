# LMM4Edit
[ACM MM 2025] [LMM4Edit: Benchmarking and Evaluating Multimodal Image Editing with LMMs](https://www.arxiv.org/abs/2507.16193)

## 🔥 Updates
- [2026-04] Upgraded the backbone model from Qwen2.5-VL-8B to Qwen3-VL-8B using the same training method.
You can download the pre-trained LoRA checkpoints from the following link:
[LMM4Edit(Qwen3-VL)](https://huggingface.co/sparkling621/LMM4Edit_Qwen3/tree/main)

## 📦 Installation
```bash
pip install -r requirements.txt
```

## ⚡Quick Start
```bash
python inference.py \
    --source_image "/path/to/source.jpg" \
    --edited_image "/path/to/edited.jpg" \
    --instruction "Editing instruction" \
    --peft_dir "weights/visual" \
    --mode visual
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

## 📁 Resources

📄 Dataset: 

[Baidu Link](https://pan.baidu.com/s/1x1QHFNC6Kz_-X44QyoQTsQ?pwd=kxyt) 

[Google Link](https://drive.google.com/file/d/13Ysqp8BTuRIA5schV4QUrahJIR6Sr7cq/view?usp=drive_link)

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


