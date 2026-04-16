import argparse
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


RELEASE_ROOT = Path(__file__).resolve().parent
APP_ROOT = RELEASE_ROOT
DEFAULT_BASE_MODEL_DIR = APP_ROOT / "models" / "base_model"
DEFAULT_RELEASE_WEIGHTS_DIR = RELEASE_ROOT / "weights"
DEFAULT_DATA_ROOT = APP_ROOT / "dataset" / "processed"
DEFAULT_MODE_TO_PEFT_DIR = {
    "visual": DEFAULT_RELEASE_WEIGHTS_DIR / "visual",
    "alignment": DEFAULT_RELEASE_WEIGHTS_DIR / "alignment",
    "preservation": DEFAULT_RELEASE_WEIGHTS_DIR / "preservation",
}

DIMENSION_CONFIG = {
    "visual": {
        "name": "visual quality",
        "focus": "realism, artifacts, texture detail, color consistency and overall fidelity",
    },
    "alignment": {
        "name": "editing instruction alignment",
        "focus": "whether the requested semantic edit is correctly applied at the right location and extent",
    },
    "preservation": {
        "name": "attribute preservation",
        "focus": "whether content unrelated to the requested edit remains unchanged",
    },
}


class ScoreRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x):
        return self.mlp(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer image editing quality with Qwen3-VL LoRA + MLP head")
    parser.add_argument("--source_image", type=str, default=None, help="Path to source image")
    parser.add_argument("--edited_image", type=str, default=None, help="Path to edited image")
    parser.add_argument("--instruction", type=str, default=None, help="Editing instruction")
    parser.add_argument("--peft_dir", type=str, default=None, help="Directory containing LoRA checkpoints")
    parser.add_argument(
        "--mode",
        type=str,
        default="preservation",
        choices=["visual", "alignment", "preservation"],
        help="Evaluation dimension",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Base model path. If omitted, local models/base_model is preferred. If missing, ModelScope auto-downloads Qwen/Qwen3-VL-8B-Instruct.",
    )
    parser.add_argument("--csv_path", type=str, default=None, help="Optional CSV to batch-evaluate a dataset split")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for CSV evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for CSV evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of CSV rows for quick checks")
    parser.add_argument("--max_pixels_per_image", type=int, default=1048576, help="Maximum pixels per image")
    parser.add_argument("--device", type=str, default=None, help="Device to run on, e.g. cuda:0 or cpu")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into the backbone before inference")
    parser.add_argument(
        "--validate_merge",
        action="store_true",
        help="Run both unmerged and merged inference and compare scores",
    )
    return parser.parse_args()


def build_query(instruction, mode):
    cfg = DIMENSION_CONFIG[mode]
    return (
        "You are scoring an image editing result. "
        "The first image is the source image and the second image is the edited image. "
        f"Editing instruction: {instruction.strip()}. "
        f"Evaluate the edited image for {cfg['name']}. "
        f"Focus on {cfg['focus']}. "
        "Return the internal assessment score."
    )


def infer_dtype(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def resolve_image(path, max_pixels_per_image):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels_per_image:
        return image

    scale = math.sqrt(max_pixels_per_image / float(pixels))
    new_width = max(16, int(round(width * scale)))
    new_height = max(16, int(round(height * scale)))
    new_width = max(16, (new_width // 16) * 16)
    new_height = max(16, (new_height // 16) * 16)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def read_csv_rows(csv_path):
    last_exc = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin1"):
        try:
            with open(csv_path, "r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to decode {csv_path}: {last_exc}")


def resolve_data_path(raw_path, csv_path):
    raw_path = str(raw_path).strip()
    raw_path_obj = Path(raw_path)
    candidates = []
    if raw_path_obj.is_absolute():
        candidates.append(raw_path_obj)
    candidates.append((Path(csv_path).parent / raw_path).resolve())
    candidates.append((APP_ROOT / raw_path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Cannot resolve path '{raw_path}' from '{csv_path}'")


def load_regression_config(peft_dir):
    config_path = Path(peft_dir) / "regression_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing regression metadata: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_download_base_model(target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope import snapshot_download
    except ImportError:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "ModelScope is required for automatic base model download. Install modelscope or provide --model_path."
            ) from exc

    snapshot_download(
        "Qwen/Qwen3-VL-8B-Instruct",
        revision="master",
        local_dir=str(target_dir),
    )
    return str(target_dir.resolve())


def resolve_peft_dir(user_peft_dir, mode):
    candidates = []
    if user_peft_dir:
        candidates.append(Path(user_peft_dir))
    candidates.append(DEFAULT_MODE_TO_PEFT_DIR[mode])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(
        f"LoRA checkpoint not found for mode '{mode}'. Provide --peft_dir or place weights under {DEFAULT_MODE_TO_PEFT_DIR[mode]}."
    )


def resolve_model_path(user_model_path, regression_config):
    candidates = []
    if user_model_path:
        candidate = Path(user_model_path)
        if candidate.exists():
            return str(candidate.resolve())
        candidates.append(candidate)

    stored_path = regression_config.get("base_model_path")
    if stored_path:
        candidate = Path(stored_path)
        if candidate.exists():
            return str(candidate.resolve())
        candidates.append(candidate)

    if DEFAULT_BASE_MODEL_DIR.exists():
        return str(DEFAULT_BASE_MODEL_DIR.resolve())

    return maybe_download_base_model(DEFAULT_BASE_MODEL_DIR)


def load_backbone(model_path, torch_dtype):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    return model.model


def make_amp_context(device, amp_dtype):
    if device.type == "cuda" and amp_dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)


def load_model_bundle(model_path, peft_dir, device, merge_lora=False):
    regression_config = load_regression_config(peft_dir)
    checkpoint_dimension = regression_config.get("dimension")
    torch_dtype = infer_dtype(device)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = load_backbone(model_path, torch_dtype=torch_dtype)
    lora_model = PeftModel.from_pretrained(
        base_model,
        peft_dir,
        is_trainable=False,
        torch_dtype=torch_dtype,
    )
    if merge_lora:
        lora_model = lora_model.merge_and_unload()

    lora_model = lora_model.to(device)
    lora_model.eval()

    score_head = ScoreRegressor(
        input_dim=int(regression_config["hidden_size"]),
        hidden_dim=int(regression_config["head_hidden_size"]),
        hidden_dim2=int(regression_config["head_hidden_size2"]),
        dropout=float(regression_config["dropout"]),
    ).to(device)
    state_dict = torch.load(Path(peft_dir) / "score_head.pth", map_location=device)
    score_head.load_state_dict(state_dict)
    score_head.eval()

    return processor, lora_model, score_head, regression_config, checkpoint_dimension


def build_processor_inputs(processor, source_image, edited_image, instruction, mode):
    query = build_query(instruction, mode)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_image},
                {"type": "image", "image": edited_image},
                {"type": "text", "text": query},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


class CSVInferenceDataset(Dataset):
    def __init__(self, csv_path, processor, mode, max_pixels_per_image, limit=None):
        self.csv_path = str(csv_path)
        self.processor = processor
        self.mode = mode
        self.max_pixels_per_image = int(max_pixels_per_image)
        self.rows = read_csv_rows(self.csv_path)
        if limit is not None:
            self.rows = self.rows[: int(limit)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        source_path = resolve_data_path(row["source"], self.csv_path)
        edited_path = resolve_data_path(row["edited"], self.csv_path)
        source = resolve_image(source_path, self.max_pixels_per_image)
        edited = resolve_image(edited_path, self.max_pixels_per_image)
        inputs = build_processor_inputs(
            processor=self.processor,
            source_image=source,
            edited_image=edited,
            instruction=str(row["instruction"]).strip(),
            mode=self.mode,
        )
        item = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "source_path": source_path,
            "edited_path": edited_path,
            "instruction": str(row["instruction"]).strip(),
        }
        if "score" in row and str(row["score"]).strip() != "":
            item["labels"] = torch.tensor(float(row["score"]), dtype=torch.float32)
        if "pixel_values" in inputs:
            item["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            item["image_grid_thw"] = inputs["image_grid_thw"]
        return item


def make_collate_fn(pad_token_id):
    def _collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        out = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "source_path": [item["source_path"] for item in batch],
            "edited_path": [item["edited_path"] for item in batch],
            "instruction": [item["instruction"] for item in batch],
        }
        if "labels" in batch[0]:
            out["labels"] = torch.stack([item["labels"] for item in batch])
        if "pixel_values" in batch[0]:
            out["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
        if "image_grid_thw" in batch[0]:
            out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        return out

    return _collate_fn


def forward_scores(model, score_head, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    amp_dtype = infer_dtype(device)
    with torch.inference_mode():
        with make_amp_context(device, amp_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
            seq_lens = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled = hidden_states[batch_indices, seq_lens]
            pooled = pooled.to(score_head.mlp[0].weight.dtype)
            pred = score_head(pooled).squeeze(-1).float()
    return pred


def average_rank(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def safe_corr_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0, 0.0
    plcc = float(np.corrcoef(x, y)[0, 1])
    srcc = float(np.corrcoef(average_rank(x), average_rank(y))[0, 1])
    if np.isnan(plcc):
        plcc = 0.0
    if np.isnan(srcc):
        srcc = 0.0
    return plcc, srcc


def compute_metrics(preds, labels):
    pred_arr = np.asarray(preds, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.float64)
    plcc, srcc = safe_corr_np(pred_arr, label_arr)
    rmse = float(np.sqrt(np.mean(np.square(pred_arr - label_arr))))
    return {
        "plcc": float(plcc),
        "srcc": float(srcc),
        "rmse": rmse,
        "score_sum": float(plcc + srcc),
    }


def compare_prediction_lists(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = np.abs(a - b)
    return {
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
    }


def resolve_default_csv(mode):
    candidate = DEFAULT_DATA_ROOT / mode / "test.csv"
    if candidate.exists():
        return str(candidate.resolve())
    return None


def resolve_default_single_inputs(source_image, edited_image, instruction, csv_path):
    if source_image and edited_image and instruction:
        return source_image, edited_image, instruction
    if not csv_path:
        raise ValueError("Missing single-image inputs and no CSV fallback is available.")
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    row = rows[0]
    source_image = source_image or resolve_data_path(row["source"], csv_path)
    edited_image = edited_image or resolve_data_path(row["edited"], csv_path)
    instruction = instruction or str(row["instruction"]).strip()
    return source_image, edited_image, instruction


def infer_single(
    source_image=None,
    edited_image=None,
    instruction=None,
    peft_dir=None,
    mode="preservation",
    model_path=None,
    max_pixels_per_image=1048576,
    device=None,
    merge_lora=False,
):
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    peft_dir = resolve_peft_dir(peft_dir, mode)
    regression_config = load_regression_config(peft_dir)
    model_path = resolve_model_path(model_path, regression_config)

    default_csv = resolve_default_csv(mode)
    source_image, edited_image, instruction = resolve_default_single_inputs(
        source_image=source_image,
        edited_image=edited_image,
        instruction=instruction,
        csv_path=default_csv,
    )
    processor, model, score_head, _, checkpoint_dimension = load_model_bundle(
        model_path=model_path,
        peft_dir=peft_dir,
        device=device,
        merge_lora=merge_lora,
    )
    if checkpoint_dimension and checkpoint_dimension != mode:
        raise ValueError(
            f"Checkpoint dimension mismatch: checkpoint is '{checkpoint_dimension}', but --mode is '{mode}'."
        )

    source = resolve_image(source_image, max_pixels_per_image=max_pixels_per_image)
    edited = resolve_image(edited_image, max_pixels_per_image=max_pixels_per_image)
    inputs = build_processor_inputs(
        processor=processor,
        source_image=source,
        edited_image=edited,
        instruction=instruction,
        mode=mode,
    )
    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    if "pixel_values" in inputs:
        batch["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        batch["image_grid_thw"] = inputs["image_grid_thw"]

    pred = forward_scores(model=model, score_head=score_head, batch=batch, device=device)
    score_output_scale = float(regression_config.get("score_output_scale", 100.0))
    score = float(pred.item() * score_output_scale)

    del batch, inputs, processor, model, score_head, pred
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score


def evaluate_csv(
    csv_path,
    peft_dir,
    mode,
    model_path=None,
    max_pixels_per_image=1048576,
    device=None,
    batch_size=1,
    num_workers=0,
    limit=None,
    merge_lora=False,
):
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    peft_dir = resolve_peft_dir(peft_dir, mode)
    regression_config = load_regression_config(peft_dir)
    model_path = resolve_model_path(model_path, regression_config)

    processor, model, score_head, regression_config, checkpoint_dimension = load_model_bundle(
        model_path=model_path,
        peft_dir=peft_dir,
        device=device,
        merge_lora=merge_lora,
    )
    if checkpoint_dimension and checkpoint_dimension != mode:
        raise ValueError(
            f"Checkpoint dimension mismatch: checkpoint is '{checkpoint_dimension}', but --mode is '{mode}'."
        )

    dataset = CSVInferenceDataset(
        csv_path=csv_path,
        processor=processor,
        mode=mode,
        max_pixels_per_image=max_pixels_per_image,
        limit=limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_collate_fn(processor.tokenizer.pad_token_id),
    )

    preds = []
    preds_raw = []
    labels = []
    labels_raw = []
    score_output_scale = float(regression_config.get("score_output_scale", 100.0))
    for batch in loader:
        pred = forward_scores(model=model, score_head=score_head, batch=batch, device=device)
        pred_np = pred.cpu().numpy()
        preds_raw.extend(pred_np.tolist())
        preds.extend((pred_np * score_output_scale).tolist())
        if "labels" in batch:
            label_np = batch["labels"].cpu().numpy()
            labels_raw.extend(label_np.tolist())
            labels.extend((label_np * score_output_scale).tolist())

    result = {
        "csv_path": str(Path(csv_path).resolve()),
        "num_samples": len(preds),
        "merge_lora": bool(merge_lora),
        "predictions": preds,
    }
    if labels_raw:
        result["metrics"] = compute_metrics(preds_raw, labels_raw)

    del loader, dataset, processor, model, score_head
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()

    if args.csv_path:
        unmerged_result = evaluate_csv(
            csv_path=args.csv_path,
            peft_dir=args.peft_dir,
            mode=args.mode,
            model_path=args.model_path,
            max_pixels_per_image=args.max_pixels_per_image,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.limit,
            merge_lora=args.merge_lora,
        )
        print(json.dumps(unmerged_result["metrics"], indent=2, ensure_ascii=True))
        if args.validate_merge and not args.merge_lora:
            merged_result = evaluate_csv(
                csv_path=args.csv_path,
                peft_dir=args.peft_dir,
                mode=args.mode,
                model_path=args.model_path,
                max_pixels_per_image=args.max_pixels_per_image,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                limit=args.limit,
                merge_lora=True,
            )
            print(
                json.dumps(
                    {
                        "unmerged_metrics": unmerged_result.get("metrics"),
                        "merged_metrics": merged_result.get("metrics"),
                        "prediction_diff": compare_prediction_lists(
                            unmerged_result["predictions"], merged_result["predictions"]
                        ),
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
        return

    score = infer_single(
        source_image=args.source_image,
        edited_image=args.edited_image,
        instruction=args.instruction,
        peft_dir=args.peft_dir,
        mode=args.mode,
        model_path=args.model_path,
        max_pixels_per_image=args.max_pixels_per_image,
        device=args.device,
        merge_lora=args.merge_lora,
    )
    print(f"Predicted Quality Score: {score:.4f}")
    if args.validate_merge and not args.merge_lora:
        merged_score = infer_single(
            source_image=args.source_image,
            edited_image=args.edited_image,
            instruction=args.instruction,
            peft_dir=args.peft_dir,
            mode=args.mode,
            model_path=args.model_path,
            max_pixels_per_image=args.max_pixels_per_image,
            device=args.device,
            merge_lora=True,
        )
        print(
            json.dumps(
                {
                    "unmerged_score": score,
                    "merged_score": merged_score,
                    "abs_diff": abs(score - merged_score),
                },
                indent=2,
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
