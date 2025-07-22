import torch
from typing import Optional, List, Union, Tuple  # Add Tuple here
from transformers import Qwen2_5_VLForConditionalGeneration
from torch.utils.data import DataLoader
from swift.llm.infer import SwiftInfer
from swift.llm import TrainArguments, load_dataset, LazyLLMDataset
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
import torch.nn as nn
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
import json

model_path = '/mnt/data/xzt/qwen2_5'
dataset = '/mnt/data/xzt/MM/train_v.json'
val_dataset = '/mnt/data/xzt/MM/test_v.json'
model_path = "/home/xuzitong/MM/weights/checkpoints/model_weights_full.pth"
output_json_path = "predictions.json"
QA = False

state_dict = torch.load(model_path,map_location="cpu")

class LMM4EditModel(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        self.mlpfinal = nn.Sequential(
            nn.Linear(config.hidden_size, 2048),  # 第一个线性层，调整维度
            # nn.GELU(),
            nn.Linear(2048, 1),  # 输出一个标量（单一数字）,
            # nn.ReLU(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # hidden_states = outputs[0]
        # logits = self.lm_head(hidden_states)
        #
        # loss = None
        # if labels is not None:
        #     # Upcast to float if we need to compute the loss to avoid potential precision issues
        #     logits = logits.float()
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)
        #     # 解码 logits 为预测的 token ID
        #     # predicted_ids = torch.argmax(logits, dim=-1)
        #     # prediction_list = predicted_ids[-1].tolist()
        #     # print(prediction_list)
        #     # text = self.tokenizer.decode(prediction_list[-7:-2], skip_special_tokens=True)
        #     # print(text)


        hidden_states = outputs[0]
        last_five_hidden_states = hidden_states[:, -8, :]
        input_tensor = last_five_hidden_states.view(1, -1)
        score = abs(self.mlpfinal(input_tensor))* 5e-5  # 取序列的最后一个 token 的隐状态作为输入
        logits = score
        if labels is not None:
            labels_list = labels[-1].tolist()
            selected_labels = labels_list[-6:-1]
            label = self.tokenizer.decode(selected_labels)
            label = torch.tensor(float(label)).to(device=logits.device)
            # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            from torch.nn import MSELoss
            loss_fct = MSELoss().to(device=logits.device)
            loss = loss_fct(logits, label)


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

def _encode_dataset(dataset):
    args = TrainArguments(model_type='qwen2_5_vl', model=model_path, dataset=dataset)
    _, processor = args.get_model_processor()
    template = args.get_template(processor)
    if args.task_type == 'causal_lm':
        template.set_mode('train')
    if template.use_model:
        template.model = model

    is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'
    if not is_grpo:
        if args.lazy_tokenize:
            dataset = LazyLLMDataset(
                dataset, template.encode, strict=args.strict, random_state=args.data_seed)
        else:
            preprocessor_cls = PackingPreprocessor if args.packing else EncodePreprocessor
            preprocessor = preprocessor_cls(template=template)
            dataset = preprocessor(dataset, num_proc=args.dataset_num_proc, strict=args.strict)

        inputs = dataset[0] if hasattr(dataset, '__len__') else next(iter(dataset))
        template.print_inputs(inputs, tokenizer_kwargs=inputs.pop('tokenizer_kwargs', None) or {})
    return dataset



dataset_kwargs = {'seed': 42, 'num_proc': 1, 'streaming': False, 'use_hf': False, 'hub_token': None, 'download_mode': 'reuse_dataset_if_exists', 'columns': {}, 'strict': False, 'model_name': [None, None], 'model_author': [None, None], 'remove_unused_columns': True}
_, val_dataset = load_dataset(val_dataset, split_dataset_ratio=1.0, **dataset_kwargs)
val_dataset = _encode_dataset(val_dataset)

model = LMM4EditModel.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# **自动分配模型到多个 GPU**

# **加载模型权重**d
# # **移除 "base_model.model." 前缀**
new_state_dict = {key.replace("base_model.model.", ""): value for key, value in state_dict.items()}
# # 将 LoRA 权重与基础权重融合
for key, value in new_state_dict.items():
    if "lora_A" in key or "lora_B" in key:
        # 找到对应的基础权重，例如 "self_attn.o_proj.base_layer.weight"
        base_key = key.replace("lora_A.default", "base_layer").replace("lora_B.default", "base_layer")
        if base_key in new_state_dict:
            base_weight = new_state_dict[base_key]
            if "lora_A" in key:
                lora_A = value
            elif "lora_B" in key:
                lora_B = value
                # 融合 LoRA 权重
                delta_weight = 4 * (lora_B @ lora_A)
                new_state_dict[base_key] = base_weight + delta_weight  # 使用低秩矩阵相乘进行融合d
new_state_dict = {key.replace("base_layer.", ""): value for key, value in new_state_dict.items()}

model_dict = model.state_dict()  # 获取模型当前的state_dict
for param_name in model_dict:
    if param_name not in new_state_dict:
        print(f"警告: 权重 {param_name} 在 state_dict 中找不到。")
    else:
        # 检查每层的形状是否匹配
        if new_state_dict[param_name].shape != model_dict[param_name].shape:
            print(f"警告: 权重 {param_name} 的形状不匹配，模型需要 {model_dict[param_name].shape}，但state_dict中是 {state_dict[param_name].shape}")
# for key, value in state_dict.items():
#     print(key)
# # **保存修改后的权重到新的.pth文件**
# torch.save(new_state_dict, "/home/xztxzt/model_weights_fused.pth")

# **加载修改后的权重**
model.load_state_dict(new_state_dict, strict=False)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# 创建一个空列表，用来存储所有推理得到的 text
predictions = []

# 设置模型为评估模式
model.eval()

# 遍历 val_dataset 中的每一项
for inputs in val_dataset:
    # 确保 inputs 中的每个值都是 Tensor 类型
    inputs = {key: torch.tensor(value) if isinstance(value, list) else value for key, value in inputs.items()}

    # 添加 batch 维度 & 移动到模型所在设备
    device = next(model.parameters()).device
    inputs = {
        k: (torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, list) else v.unsqueeze(0).to(device))
        for k, v in inputs.items()
    }

    # 执行推理
    with torch.no_grad():
        output = model(**inputs)
    # 获取 logits
    logits = output['logits']
    # 获取预测的 ids
    if QA:
        predicted_ids = torch.argmax(logits, dim=-1)
        # 获取最后一个预测的 token ID 列表
        prediction_list = predicted_ids[-1].tolist()
        # 解码为文本
        text = tokenizer.decode(prediction_list[-7:-2], skip_special_tokens=True)
        predictions.append(text)
        for text in predictions:
            print(text)
    else:
        score = logits
        # 将生成的 text 添加到 predictions 列表中
        predictions.append(logits)
        for score in predictions:
            print(score)

    with open(output_json_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')

    print("Evaluation results saved")



