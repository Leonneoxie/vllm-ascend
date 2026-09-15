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


def _detect_num_layers(ptq_dir):
    """Auto-detect number of layers from layer_{N}_*.pt files in ptq_dir.

    Falls back to NUM_LAYERS (32) if no matching files are found.
    Works for all targets: attn-linear (layer_{N}_self_attn.pt),
    mlp (layer_{N}_mlp.pt), moe (layer_{N}_expert_{E}.pt).
    """
    import re
    pattern = re.compile(r"^layer_(\d+)_")
    max_n = -1
    try:
        for f in os.listdir(ptq_dir):
            m = pattern.match(f)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
    except OSError:
        pass
    return max_n + 1 if max_n >= 0 else NUM_LAYERS


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

    if target == "mlp":
        filter_keys = [".mlp."]
    elif target == "attn-linear":
        filter_keys = [".self_attn.", ".linear_attn."]
    else:
        filter_keys = [""]
    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt") as st:
            for key in st.keys():  # noqa: SIM118 - safe_open exposes keys(), not dict iteration.
                if any(fk in key for fk in filter_keys) and ".weight" in key:
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

    for layer_idx in range(_detect_num_layers(ptq_dir)):
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

    for layer_idx in range(_detect_num_layers(ptq_dir)):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = _flatten_ptq_params(ptq["mlp"])
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


def convert_flatquant_moe(ptq_dir, output_path, num_experts=256):
    """Convert MoE FlatQuant PTQ params to vllm-ascend sidecar safetensors.

    AMCT MoE FlatQuant produces one .pt file per (layer, expert):
    ``layer_{N}_expert_{E}.pt``.  Each file contains ``input_transform``
    (FC1) and ``hidden_transform`` (FC2), each with ``linear_left.weight``,
    ``linear_right.weight`` and ``diag_scale``.

    The vLLM sidecar key format (see ``_build_flatquant_sidecar_key`` in
    ``fake_mx.py``) is::

        layers.{N}.experts.{E}.{fc}.{comp}

    where ``fc`` is ``fc1``/``fc2`` and ``comp`` is ``left_trans``/
    ``right_trans``/``diag`` (``diag_scale`` is abbreviated to ``diag``).

    Unlike Dense FlatQuant, the runtime applies the inverse weight transform
    itself, so the sidecar only stores the L/R/diag matrices — no
    transformed weights and no ``--model_dir`` required.
    """
    import re

    # Auto-detect num_layers from layer_{N}_expert_0.pt files.
    layer_pattern = re.compile(r"^layer_(\d+)_expert_0\.pt$")
    layer_indices = sorted(
        int(layer_pattern.match(f).group(1))
        for f in os.listdir(ptq_dir)
        if layer_pattern.match(f)
    )
    if not layer_indices:
        raise FileNotFoundError(
            f"No layer_N_expert_0.pt files found in {ptq_dir}"
        )
    num_layers = layer_indices[-1] + 1
    print(f"Detected {num_layers} layers, {num_experts} experts per layer.")

    output_tensors = {}
    total = num_layers * num_experts
    done = 0
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            pt_path = os.path.join(
                ptq_dir, f"layer_{layer_idx}_expert_{expert_idx}.pt"
            )
            if not os.path.exists(pt_path):
                print(f"WARNING: missing {pt_path}, skipping (identity fallback)")
                done += 1
                continue
            params = torch.load(pt_path, map_location="cpu", weights_only=False)
            flat = _flatten_ptq_params(params)
            for fc_name, amct_prefix in [
                ("fc1", "input_transform"),
                ("fc2", "hidden_transform"),
            ]:
                for amct_suffix, comp_short in [
                    ("linear_left.weight", "left_trans"),
                    ("linear_right.weight", "right_trans"),
                    ("diag_scale", "diag"),
                ]:
                    amct_key = f"{amct_prefix}.transform.{amct_suffix}"
                    if amct_key not in flat:
                        if amct_suffix == "diag_scale":
                            continue
                        raise KeyError(f"Missing {amct_key} in {pt_path}")
                    tensor = flat[amct_key].to(torch.float32).contiguous()
                    sidecar_key = (
                        f"layers.{layer_idx}.experts.{expert_idx}."
                        f"{fc_name}.{comp_short}"
                    )
                    output_tensors[sidecar_key] = tensor
            done += 1
            if done % 1000 == 0:
                print(f"  Processed {done}/{total} files ({100*done//total}%)...")

    save_file(output_tensors, output_path)
    expected = num_layers * num_experts * 6
    print(f"Saved {len(output_tensors)} tensors to {output_path}")
    print(f"  Expected: {expected} tensors (with diag)")
    print(f"  Actual:   {len(output_tensors)}")


def _process_flatquant_proj(original_weights, output_tensors, layer_idx, proj_path, left, right, diag):
    """Output FlatQuant transform matrices in vllm-ascend fused-projection naming.

    vllm-ascend loads left_trans/right_trans/diag_scale at runtime and applies
    the weight inverse transform itself, so we do NOT output pre-transformed
    .weight tensors here.
    """
    prefix = f"model.language_model.layers.{layer_idx}.{proj_path}"
    output_tensors[f"{prefix}.left_trans"] = left.to(torch.float32).clone()
    output_tensors[f"{prefix}.right_trans"] = right.to(torch.float32).clone()
    if diag is not None:
        output_tensors[f"{prefix}.diag_scale"] = diag.to(torch.float32).clone()


def _extract_lht_transform_weight(params, transform_name="input_transform"):
    """Extract LHT transform_weight from AMCT params, supporting multiple key formats."""
    key = f"{transform_name}.transform.linear.weight"
    if key in params:
        return params[key]
    if f"{transform_name}.transform_weight" in params:
        return params[f"{transform_name}.transform_weight"]
    if transform_name in params and isinstance(params[transform_name], dict):
        sub = params[transform_name]
        if "transform_weight" in sub:
            return sub["transform_weight"]
        if "transform" in sub and isinstance(sub["transform"], dict) and "linear" in sub["transform"]:
            return sub["transform"]["linear"]["weight"]
    return None


def convert_lht_attn(ptq_dir, output_path):
    """转换 attn 层 LHT 参数。"""
    output_tensors = {}
    for layer_idx in range(_detect_num_layers(ptq_dir)):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            input_tw = _extract_lht_transform_weight(params, "input_transform")
            out_tw = _extract_lht_transform_weight(params, "out_transform")
            if input_tw is None or out_tw is None:
                continue
            if unit_name == "self_attn":
                prefix = f"model.language_model.layers.{layer_idx}.self_attn"
                output_tensors[f"{prefix}.qkv_proj.transform_weight"] = input_tw.to(torch.float32)
                output_tensors[f"{prefix}.o_proj.transform_weight"] = out_tw.to(torch.float32)
            elif unit_name == "linear_attn":
                prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
                output_tensors[f"{prefix}.in_proj_qkvz.transform_weight"] = input_tw.to(torch.float32)
                output_tensors[f"{prefix}.in_proj_ba.transform_weight"] = input_tw.to(torch.float32).clone()
                output_tensors[f"{prefix}.out_proj.transform_weight"] = out_tw.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_lht_mlp(ptq_dir, output_path):
    """转换 mlp 层 LHT 参数。"""
    output_tensors = {}
    for layer_idx in range(_detect_num_layers(ptq_dir)):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = ptq["mlp"]
        input_tw = _extract_lht_transform_weight(params, "input_transform")
        hidden_tw = _extract_lht_transform_weight(params, "hidden_transform")
        if input_tw is None or hidden_tw is None:
            continue
        for proj, tw in [
            ("gate_proj", input_tw.clone()),
            ("up_proj", input_tw.clone()),
            ("down_proj", hidden_tw.clone()),
        ]:
            prefix = f"model.language_model.layers.{layer_idx}.mlp.{proj}"
            output_tensors[f"{prefix}.transform_weight"] = tw.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_lht_moe(ptq_dir, output_path, num_experts=256):
    """Convert MoE LHT PTQ params to vllm-ascend sidecar safetensors.

    AMCT MoE LHT produces one .pt file per (layer, expert):
    ``layer_{N}_expert_{E}.pt``.  Each file contains ``input_transform``
    (w13/FC1) and ``hidden_transform`` (w2/FC2), each with
    ``transform.linear.weight`` — a K×K Hadamard learning matrix.

    Sidecar key format (see ``_build_flatquant_sidecar_key`` in
    ``fake_mx.py``)::

        layers.{N}.experts.{E}.{w13|w2}.transform_weight

    The runtime loads these matrices, applies ``inv(T).T`` to the weight
    at load time, and applies ``x @ T`` to activations at forward time.
    """
    import re

    layer_pattern = re.compile(r"^layer_(\d+)_expert_0\.pt$")
    layer_indices = sorted(
        int(layer_pattern.match(f).group(1))
        for f in os.listdir(ptq_dir)
        if layer_pattern.match(f)
    )
    if not layer_indices:
        raise FileNotFoundError(
            f"No layer_N_expert_0.pt files found in {ptq_dir}"
        )
    num_layers = layer_indices[-1] + 1
    print(f"Detected {num_layers} layers, {num_experts} experts per layer.")

    output_tensors = {}
    total = num_layers * num_experts
    done = 0
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            pt_path = os.path.join(
                ptq_dir, f"layer_{layer_idx}_expert_{expert_idx}.pt"
            )
            if not os.path.exists(pt_path):
                print(f"WARNING: missing {pt_path}, skipping (identity fallback)")
                done += 1
                continue
            params = torch.load(pt_path, map_location="cpu", weights_only=False)
            flat = _flatten_ptq_params(params)
            for fc_name, amct_prefix in [
                ("w13", "input_transform"),
                ("w2", "hidden_transform"),
            ]:
                amct_key = f"{amct_prefix}.transform.linear.weight"
                if amct_key not in flat:
                    alt_key = f"{amct_prefix}.transform_weight"
                    if alt_key in flat:
                        amct_key = alt_key
                    else:
                        raise KeyError(
                            f"Missing LHT transform key in {pt_path}: "
                            f"tried '{amct_prefix}.transform.linear.weight' "
                            f"and '{amct_prefix}.transform_weight'"
                        )
                tensor = flat[amct_key].to(torch.float32).contiguous()
                sidecar_key = (
                    f"layers.{layer_idx}.experts.{expert_idx}."
                    f"{fc_name}.transform_weight"
                )
                output_tensors[sidecar_key] = tensor
            done += 1
            if done % 1000 == 0:
                print(f"  Processed {done}/{total} files ({100*done//total}%)...")

    save_file(output_tensors, output_path)
    expected = num_layers * num_experts * 2
    print(f"Saved {len(output_tensors)} tensors to {output_path}")
    print(f"  Expected: {expected} tensors (w13 + w2 per expert)")
    print(f"  Actual:   {len(output_tensors)}")


def convert_omniquant_moe(ptq_dir, output_path, num_experts=256):
    """Convert MoE OmniQuant PTQ params to vllm-ascend sidecar safetensors.

    AMCT MoE OmniQuant produces one .pt file per (layer, expert):
    ``layer_{N}_expert_{E}.pt``.  Each file contains ``input_transform``
    (w13/FC1) and ``hidden_transform`` (w2/FC2), each with ``log_scale``
    (a per-dimension scale vector).

    Unlike FlatQuant/LHT MoE which use per-expert sidecar keys, OmniQuant
    MoE uses ``_copy_transform_param`` which loads a per-LAYER tensor
    ``[E, dim]`` at once.  So this function stacks per-expert log_scales
    into per-layer tensors.

    Sidecar key format (matches ``_layer_prefix_candidates`` in
    ``fake_mx.py``)::

        model.language_model.layers.{N}.mlp.experts.w13_log_scale  [E, hidden]
        model.language_model.layers.{N}.mlp.experts.w2_log_scale   [E, intermediate]

    Note: The OmniQuant MoE loader copies the entire per-layer tensor, so
    the sidecar must have the same expert count as the local rank.
    Works with EP=1 (all 256 experts).  For EP>1 the sidecar shape must
    match the per-rank local expert count.
    """
    import re

    layer_pattern = re.compile(r"^layer_(\d+)_expert_0\.pt$")
    layer_indices = sorted(
        int(layer_pattern.match(f).group(1))
        for f in os.listdir(ptq_dir)
        if layer_pattern.match(f)
    )
    if not layer_indices:
        raise FileNotFoundError(
            f"No layer_N_expert_0.pt files found in {ptq_dir}"
        )
    num_layers = layer_indices[-1] + 1
    print(f"Detected {num_layers} layers, {num_experts} experts per layer.")

    output_tensors = {}
    total = num_layers
    done = 0
    for layer_idx in range(num_layers):
        w13_scales = []
        w2_scales = []
        for expert_idx in range(num_experts):
            pt_path = os.path.join(
                ptq_dir, f"layer_{layer_idx}_expert_{expert_idx}.pt"
            )
            if not os.path.exists(pt_path):
                w13_scales.append(None)
                w2_scales.append(None)
                continue
            params = torch.load(pt_path, map_location="cpu", weights_only=False)
            flat = _flatten_ptq_params(params)
            w13_scales.append(flat["input_transform.log_scale"].to(torch.float32))
            w2_scales.append(flat["hidden_transform.log_scale"].to(torch.float32))

        # Fill missing experts with zeros (shape from first available).
        ref_w13 = next((s for s in w13_scales if s is not None), None)
        ref_w2 = next((s for s in w2_scales if s is not None), None)
        if ref_w13 is None:
            print(f"WARNING: no expert files for layer {layer_idx}, skipping")
            done += 1
            continue
        for i in range(len(w13_scales)):
            if w13_scales[i] is None:
                w13_scales[i] = torch.zeros_like(ref_w13)
            if w2_scales[i] is None:
                w2_scales[i] = torch.zeros_like(ref_w2)

        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
        output_tensors[f"{prefix}.w13_log_scale"] = torch.stack(w13_scales, dim=0)
        output_tensors[f"{prefix}.w2_log_scale"] = torch.stack(w2_scales, dim=0)
        done += 1
        if done % 10 == 0:
            print(f"  Processed {done}/{total} layers ({100*done//total}%)...")

    save_file(output_tensors, output_path)
    expected = num_layers * 2
    print(f"Saved {len(output_tensors)} tensors to {output_path}")
    print(f"  Expected: {expected} tensors (w13_log_scale + w2_log_scale per layer)")
    print(f"  Actual:   {len(output_tensors)}")


def merge_safetensors(paths, output_path):
    """合并多个 safetensors 文件。"""
    merged = {}
    for p in paths:
        data = load_file(p)
        merged.update(data)
        print(f"  Loaded {len(data)} tensors from {p}")
    save_file(merged, output_path)
    print(f"Saved {len(merged)} tensors to {output_path}")


def convert_omniquant_attn(ptq_dir, output_path):
    """转换 attn 层 OmniQuant 参数。输出 log_scale per projection."""
    output_tensors = {}
    for layer_idx in range(_detect_num_layers(ptq_dir)):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            input_scale = params["input_transform.log_scale"]
            out_scale = params["out_transform.log_scale"]
            if unit_name == "self_attn":
                prefix = f"model.language_model.layers.{layer_idx}.self_attn"
                output_tensors[f"{prefix}.qkv_proj.log_scale"] = input_scale.to(torch.float32)
                output_tensors[f"{prefix}.o_proj.log_scale"] = out_scale.to(torch.float32)
            elif unit_name == "linear_attn":
                prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
                output_tensors[f"{prefix}.in_proj_qkvz.log_scale"] = input_scale.to(torch.float32)
                output_tensors[f"{prefix}.in_proj_ba.log_scale"] = input_scale.to(torch.float32).clone()
                output_tensors[f"{prefix}.out_proj.log_scale"] = out_scale.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_omniquant_mlp(ptq_dir, output_path):
    """转换 mlp 层 OmniQuant 参数。"""
    output_tensors = {}
    for layer_idx in range(_detect_num_layers(ptq_dir)):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = ptq["mlp"]
        flat = _flatten_ptq_params(params)
        input_scale = flat["input_transform.log_scale"]
        hidden_scale = flat["hidden_transform.log_scale"]
        for proj, scale in [
            ("gate_proj", input_scale.clone()),
            ("up_proj", input_scale.clone()),
            ("down_proj", hidden_scale.clone()),
        ]:
            prefix = f"model.language_model.layers.{layer_idx}.mlp.{proj}"
            output_tensors[f"{prefix}.log_scale"] = scale.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def _flatten_ptq_params(params, prefix=""):
    """Recursively flatten nested dict PTQ params into dotted keys."""
    flat = {}
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_ptq_params(v, full_key))
        elif isinstance(v, torch.Tensor):
            flat[full_key] = v
    return flat


def main():
    parser = argparse.ArgumentParser(description="Convert AMCT PTQ params to vllm-ascend safetensors")
    parser.add_argument(
        "--algo",
        choices=["flatquant", "lht", "omniquant"],
        help="Algorithm name",
    )
    parser.add_argument("--target", choices=["attn-linear", "mlp", "moe"], help="Quant target")
    parser.add_argument("--ptq_dir", help="PTQ .pt params directory")
    parser.add_argument("--model_dir", help="Model directory (for FlatQuant weight transform, not needed for moe)")
    parser.add_argument("--output", help="Output safetensors path")
    parser.add_argument("--num_experts", type=int, default=256, help="Number of experts per layer (MoE target only, default 256)")
    parser.add_argument("--merge", nargs=3, metavar=("INPUT1", "INPUT2", "OUTPUT"), help="Merge two safetensors files")
    args = parser.parse_args()

    if args.merge:
        merge_safetensors([args.merge[0], args.merge[1]], args.merge[2])
        return

    if not all([args.algo, args.target, args.ptq_dir, args.output]):
        parser.error("--algo, --target, --ptq_dir, --output are required (unless --merge)")

    if args.algo == "flatquant":
        if args.target == "moe":
            convert_flatquant_moe(args.ptq_dir, args.output, args.num_experts)
        else:
            if not args.model_dir:
                parser.error("--model_dir is required for FlatQuant attn-linear/mlp (needs original weights for transform)")
            if args.target == "attn-linear":
                convert_flatquant_attn(args.ptq_dir, args.model_dir, args.output)
            elif args.target == "mlp":
                convert_flatquant_mlp(args.ptq_dir, args.model_dir, args.output)
    elif args.algo == "lht":
        if args.target == "attn-linear":
            convert_lht_attn(args.ptq_dir, args.output)
        elif args.target == "mlp":
            convert_lht_mlp(args.ptq_dir, args.output)
        elif args.target == "moe":
            convert_lht_moe(args.ptq_dir, args.output, args.num_experts)
    elif args.algo == "omniquant":
        if args.target == "attn-linear":
            convert_omniquant_attn(args.ptq_dir, args.output)
        elif args.target == "mlp":
            convert_omniquant_mlp(args.ptq_dir, args.output)
        elif args.target == "moe":
            convert_omniquant_moe(args.ptq_dir, args.output, args.num_experts)


if __name__ == "__main__":
    main()
