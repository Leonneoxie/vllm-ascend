#!/usr/bin/env python3
"""将 AMCT PTQ .pt 参数转换为 vllm-ascend fake_mx safetensors 格式。

支持两种算法和两种量化目标：
  --algo flatquant --target attn-linear  -> attn 层 FlatQuant 参数
  --algo flatquant --target mlp           -> mlp 层 FlatQuant 参数
  --algo lht --target attn-linear         -> attn 层 LHT 参数
  --algo lht --target mlp                 -> mlp 层 LHT 参数

用法:
  python convert_ptq_to_vllm.py --algo flatquant --target attn-linear \
    --ptq_dir /data/ptq_output_flatquant_amct/ptq_params/qwen3_5/attn-linear \
    --model_dir /data/models/Qwen3.5-9B \
    --output /data/flatquant_attn_params.safetensors

  python convert_ptq_to_vllm.py --algo flatquant --target mlp \
    --ptq_dir /data/ptq_output_flatquant_mlp/ptq_params/qwen3_5/mlp \
    --model_dir /data/models/Qwen3.5-9B \
    --output /data/flatquant_mlp_params.safetensors

  # 合并 attn + mlp 参数
  python convert_ptq_to_vllm.py --merge \
    /data/flatquant_attn_params.safetensors \
    /data/flatquant_mlp_params.safetensors \
    /data/flatquant_attn_mlp_params.safetensors
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

NUM_LAYERS = 32


def load_ptq_params(ptq_dir, layer_idx, target):
    """加载一个层的 PTQ 参数。"""
    result = {}
    if target == "attn-linear":
        for unit_name in ["linear_attn", "self_attn"]:
            path = os.path.join(ptq_dir, f"layer_{layer_idx}_{unit_name}.pt")
            if os.path.exists(path):
                result[unit_name] = torch.load(path, map_location="cpu")
    elif target == "mlp":
        path = os.path.join(ptq_dir, f"layer_{layer_idx}_mlp.pt")
        if os.path.exists(path):
            result["mlp"] = torch.load(path, map_location="cpu")
    return result


def load_original_weights(model_dir, target):
    """加载模型原始权重中与 target 相关的部分。"""
    from safetensors import safe_open

    weights = {}
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
    else:
        shard_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]

    filter_key = ".mlp." if target == "mlp" else (".self_attn." if target == "attn-linear" else "")
    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt") as st:
            for key in st:
                if filter_key in key and ".weight" in key:
                    weights[key] = st.get_tensor(key)
    return weights


def apply_flatquant_transform(weight, left, right, diag_scale=None):
    """对权重应用 FlatQuant 逆变换: W' = inv(left) @ W @ inv(right).T / diag_scale。"""
    original_shape = weight.shape
    in_features = original_shape[-1]
    left_dim = left.shape[0]
    right_dim = right.shape[0]
    assert left_dim * right_dim == in_features, f"{left_dim}*{right_dim} != {in_features}"

    weight_f32 = weight.to(torch.float32).reshape(-1, left_dim, right_dim)
    inv_left = torch.linalg.solve(left.to(torch.float32), torch.eye(left_dim, dtype=torch.float32))
    inv_right_t = torch.linalg.solve(right.t().to(torch.float32), torch.eye(right_dim, dtype=torch.float32))
    transformed = torch.matmul(inv_left, weight_f32)
    transformed = torch.matmul(transformed, inv_right_t)
    if diag_scale is not None:
        diag = diag_scale.to(torch.float32).reshape(left_dim, right_dim)
        transformed = transformed / diag.unsqueeze(0).clamp(min=1e-8)
    return transformed.reshape(original_shape)


def convert_flatquant_attn(ptq_dir, model_dir, output_path):
    """转换 attn 层 FlatQuant 参数。"""
    original_weights = load_original_weights(model_dir, "attn-linear")
    output_tensors = {}

    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            input_left = params["input_transform.transform.linear_left.weight"]
            input_right = params["input_transform.transform.linear_right.weight"]
            input_diag = params.get("input_transform.transform.diag_scale")
            out_left = params["out_transform.transform.linear_left.weight"]
            out_right = params["out_transform.transform.linear_right.weight"]
            out_diag = params.get("out_transform.transform.diag_scale")

            if unit_name == "self_attn":
                for proj, tf_left, tf_right, tf_diag in [
                    ("qkv_proj", input_left, input_right, input_diag),
                    ("o_proj", out_left, out_right, out_diag),
                ]:
                    _process_flatquant_proj(
                        original_weights, output_tensors, layer_idx, f"self_attn.{proj}", tf_left, tf_right, tf_diag
                    )
            elif unit_name == "linear_attn":
                for proj in ["in_proj_qkvz", "in_proj_ba"]:
                    _process_flatquant_proj(
                        original_weights,
                        output_tensors,
                        layer_idx,
                        f"linear_attn.{proj}",
                        input_left,
                        input_right,
                        input_diag,
                    )
                _process_flatquant_proj(
                    original_weights, output_tensors, layer_idx, "linear_attn.out_proj", out_left, out_right, out_diag
                )

    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_flatquant_mlp(ptq_dir, model_dir, output_path):
    """转换 mlp 层 FlatQuant 参数。"""
    original_weights = load_original_weights(model_dir, "mlp")
    output_tensors = {}

    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = ptq["mlp"]
        input_left = params["input_transform.transform.linear_left.weight"]
        input_right = params["input_transform.transform.linear_right.weight"]
        input_diag = params.get("input_transform.transform.diag_scale")
        hidden_left = params["hidden_transform.transform.linear_left.weight"]
        hidden_right = params["hidden_transform.transform.linear_right.weight"]
        hidden_diag = params.get("hidden_transform.transform.diag_scale")

        for proj, tf_left, tf_right, tf_diag in [
            (
                "gate_proj",
                input_left.clone(),
                input_right.clone(),
                input_diag.clone() if input_diag is not None else None,
            ),
            (
                "up_proj",
                input_left.clone(),
                input_right.clone(),
                input_diag.clone() if input_diag is not None else None,
            ),
            (
                "down_proj",
                hidden_left.clone(),
                hidden_right.clone(),
                hidden_diag.clone() if hidden_diag is not None else None,
            ),
        ]:
            _process_flatquant_proj(
                original_weights, output_tensors, layer_idx, f"mlp.{proj}", tf_left, tf_right, tf_diag
            )

    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def _process_flatquant_proj(original_weights, output_tensors, layer_idx, proj_path, left, right, diag):
    prefix = f"model.language_model.layers.{layer_idx}.{proj_path}"
    key = f"{prefix}.weight"
    if key not in original_weights:
        print(f"  WARNING: {key} not found")
        return
    w = original_weights[key]
    transformed_w = apply_flatquant_transform(w, left, right, diag)
    output_tensors[key] = transformed_w.to(torch.bfloat16)
    output_tensors[f"{prefix}.left_trans"] = left.to(torch.bfloat16)
    output_tensors[f"{prefix}.right_trans"] = right.to(torch.bfloat16)
    output_tensors[f"{prefix}.clip_ratio"] = torch.ones(1, dtype=torch.float32)
    if diag is not None:
        output_tensors[f"{prefix}.diag_scale"] = diag.to(torch.float32)


def convert_lht_attn(ptq_dir, output_path):
    """转换 attn 层 LHT 参数。"""
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            input_tw = params["input_transform.transform_weight"]
            out_tw = params["out_transform.transform_weight"]
            if unit_name == "self_attn":
                prefix = f"model.language_model.layers.{layer_idx}.self_attn"
                output_tensors[f"{prefix}.qkv_proj.transform_weight"] = input_tw.to(torch.bfloat16)
                output_tensors[f"{prefix}.o_proj.transform_weight"] = out_tw.to(torch.bfloat16)
            elif unit_name == "linear_attn":
                prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
                output_tensors[f"{prefix}.in_proj_qkvz.transform_weight"] = input_tw.to(torch.bfloat16)
                output_tensors[f"{prefix}.in_proj_ba.transform_weight"] = input_tw.to(torch.bfloat16)
                output_tensors[f"{prefix}.out_proj.transform_weight"] = out_tw.to(torch.bfloat16)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_lht_mlp(ptq_dir, output_path):
    """转换 mlp 层 LHT 参数。"""
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = ptq["mlp"]
        input_tw = params["input_transform.transform_weight"]
        hidden_tw = params["hidden_transform.transform_weight"]
        for proj, tw in [
            ("gate_proj", input_tw.clone()),
            ("up_proj", input_tw.clone()),
            ("down_proj", hidden_tw.clone()),
        ]:
            prefix = f"model.language_model.layers.{layer_idx}.mlp.{proj}"
            output_tensors[f"{prefix}.transform_weight"] = tw.to(torch.bfloat16)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def merge_safetensors(paths, output_path):
    """合并多个 safetensors 文件。"""
    merged = {}
    for p in paths:
        data = load_file(p)
        merged.update(data)
        print(f"  Loaded {len(data)} tensors from {p}")
    save_file(merged, output_path)
    print(f"Saved {len(merged)} tensors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert AMCT PTQ params to vllm-ascend safetensors")
    parser.add_argument("--algo", choices=["flatquant", "lht"], help="Algorithm name")
    parser.add_argument("--target", choices=["attn-linear", "mlp"], help="Quant target")
    parser.add_argument("--ptq_dir", help="PTQ .pt params directory")
    parser.add_argument("--model_dir", help="Model directory (for FlatQuant weight transform)")
    parser.add_argument("--output", help="Output safetensors path")
    parser.add_argument("--merge", nargs=3, metavar=("INPUT1", "INPUT2", "OUTPUT"), help="Merge two safetensors files")
    args = parser.parse_args()

    if args.merge:
        merge_safetensors([args.merge[0], args.merge[1]], args.merge[2])
        return

    if not all([args.algo, args.target, args.ptq_dir, args.output]):
        parser.error("--algo, --target, --ptq_dir, --output are required (unless --merge)")

    if args.algo == "flatquant":
        if not args.model_dir:
            parser.error("--model_dir is required for FlatQuant (needs original weights for transform)")
        if args.target == "attn-linear":
            convert_flatquant_attn(args.ptq_dir, args.model_dir, args.output)
        elif args.target == "mlp":
            convert_flatquant_mlp(args.ptq_dir, args.model_dir, args.output)
    elif args.algo == "lht":
        if args.target == "attn-linear":
            convert_lht_attn(args.ptq_dir, args.output)
        elif args.target == "mlp":
            convert_lht_mlp(args.ptq_dir, args.output)


if __name__ == "__main__":
    main()
