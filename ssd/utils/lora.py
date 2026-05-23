import json
import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm


def load_lora_config(lora_path: str) -> dict:
    """Load LoRA adapter configuration from adapter_config.json."""
    config_file = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"No adapter_config.json found at {lora_path}. "
            "Expected a PEFT/LoRA adapter directory."
        )
    with open(config_file, "r") as f:
        return json.load(f)


def load_lora_state_dict(lora_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors or bin files."""
    state_dict = {}

    # Try safetensors first
    safetensor_files = glob(os.path.join(lora_path, "*.safetensors"))
    if safetensor_files:
        for file in safetensor_files:
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        return state_dict

    # Fall back to bin files
    bin_files = glob(os.path.join(lora_path, "adapter_model.bin"))
    if not bin_files:
        bin_files = glob(os.path.join(lora_path, "pytorch_model*.bin"))
    for file in bin_files:
        loaded = torch.load(file, map_location="cpu", weights_only=True)
        state_dict.update(loaded)

    if not state_dict:
        raise FileNotFoundError(
            f"No adapter weights found at {lora_path}. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )
    return state_dict


def _parse_lora_key(key: str) -> tuple[str, str, str] | None:
    """Parse a LoRA weight key into (module_path, target_module, lora_type).

    Example key: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    Returns: ("model.layers.0.self_attn", "q_proj", "lora_A")
    """
    # Strip common PEFT prefixes
    for prefix in ["base_model.model.", ""]:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break

    # Must end with .lora_A.weight or .lora_B.weight
    if key.endswith(".lora_A.weight"):
        lora_type = "lora_A"
        key = key[: -len(".lora_A.weight")]
    elif key.endswith(".lora_B.weight"):
        lora_type = "lora_B"
        key = key[: -len(".lora_B.weight")]
    elif key.endswith(".lora_A.default.weight"):
        lora_type = "lora_A"
        key = key[: -len(".lora_A.default.weight")]
    elif key.endswith(".lora_B.default.weight"):
        lora_type = "lora_B"
        key = key[: -len(".lora_B.default.weight")]
    else:
        return None

    # Split into module path and target module name
    parts = key.rsplit(".", 1)
    if len(parts) == 2:
        module_path, target_module = parts
    else:
        module_path = ""
        target_module = parts[0]

    return module_path, target_module, lora_type


def _get_tp_info(param: nn.Parameter) -> tuple[int, int, int | None]:
    """Extract TP sharding info from a parameter's parent module.

    Returns (tp_rank, tp_size, tp_dim) where tp_dim is None for replicated.
    """
    tp_rank = getattr(param, "_tp_rank", 0)
    tp_size = getattr(param, "_tp_size", 1)
    tp_dim = getattr(param, "_tp_dim", None)
    return tp_rank, tp_size, tp_dim


def merge_lora_weights(
    model: nn.Module,
    lora_path: str,
    packed_modules_mapping: dict,
) -> None:
    """Load LoRA adapter weights and merge them into the base model parameters.

    This performs W' = W + scaling * (B @ A) for each LoRA target module,
    handling packed modules (QKV, gate_up) and tensor-parallel sharding.
    """
    print(f"[LoRA] Loading adapter from {lora_path}")
    lora_config = load_lora_config(lora_path)
    lora_state_dict = load_lora_state_dict(lora_path)

    r = lora_config["r"]
    lora_alpha = lora_config.get("lora_alpha", r)
    scaling = lora_alpha / r
    target_modules = lora_config.get("target_modules", [])

    print(f"[LoRA] r={r}, alpha={lora_alpha}, scaling={scaling:.4f}")
    print(f"[LoRA] target_modules={target_modules}")

    # Group lora_A and lora_B by (module_path, target_module)
    lora_pairs: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in lora_state_dict.items():
        parsed = _parse_lora_key(key)
        if parsed is None:
            print(f"[LoRA] Skipping non-LoRA key: {key}")
            continue
        module_path, target_module, lora_type = parsed
        pair_key = (module_path, target_module)
        if pair_key not in lora_pairs:
            lora_pairs[pair_key] = {}
        lora_pairs[pair_key][lora_type] = tensor

    merged_count = 0
    for (module_path, target_module), pair in tqdm(
        lora_pairs.items(), desc="Merging LoRA weights"
    ):
        if "lora_A" not in pair or "lora_B" not in pair:
            print(
                f"[LoRA] Warning: incomplete pair for {module_path}.{target_module}, skipping"
            )
            continue

        lora_a = pair["lora_A"]  # [r, in_features]
        lora_b = pair["lora_B"]  # [out_features, r]
        delta_w = scaling * (lora_b @ lora_a)  # [out_features, in_features]

        # Check if this target module maps to a packed module
        if target_module in packed_modules_mapping:
            packed_name, shard_id = packed_modules_mapping[target_module]
            param_name = f"{module_path}.{packed_name}.weight"
            try:
                param = model.get_parameter(param_name)
            except AttributeError:
                print(f"[LoRA] Warning: parameter {param_name} not found, skipping")
                continue

            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                # Use the existing weight_loader but add delta instead of copy
                # We need to manually handle the shard placement
                _merge_packed_lora(param, delta_w, shard_id, weight_loader)
            else:
                param.data += delta_w
        else:
            # Direct (unpacked) module
            param_name = f"{module_path}.{target_module}.weight"
            try:
                param = model.get_parameter(param_name)
            except AttributeError:
                print(f"[LoRA] Warning: parameter {param_name} not found, skipping")
                continue

            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                _merge_direct_lora(param, delta_w, weight_loader)
            else:
                param.data += delta_w

        merged_count += 1

    print(f"[LoRA] Merged {merged_count} LoRA weight pairs into model")


def _merge_packed_lora(
    param: nn.Parameter,
    delta_w: torch.Tensor,
    shard_id: int | str,
    weight_loader,
) -> None:
    """Merge a LoRA delta into a packed parameter (QKV or gate_up).

    Instead of copying, we add the delta to the existing weights at the
    correct shard position. This mimics what weight_loader does but with
    addition instead of copy.
    """
    import inspect

    sig = inspect.signature(weight_loader)
    params = list(sig.parameters.keys())

    # For QKVParallelLinear: weight_loader(param, loaded_weight, loaded_shard_id: str)
    # For MergedColumnParallelLinear: weight_loader(param, loaded_weight, loaded_shard_id: int)
    # We create a temporary copy, load the delta into it via weight_loader,
    # then add the non-zero result back to the original param

    # Save original data
    original_data = param.data.clone()
    # Zero out param data so weight_loader writes delta into clean space
    param.data.zero_()
    # Use weight_loader to place the delta at the correct shard position
    weight_loader(param, delta_w, shard_id)
    # Add the placed delta to the original weights
    param.data += original_data


def _merge_direct_lora(
    param: nn.Parameter,
    delta_w: torch.Tensor,
    weight_loader,
) -> None:
    """Merge a LoRA delta into a direct (non-packed) parameter.

    Handles TP sharding by using the same weight_loader logic but with addition.
    """
    import inspect

    sig = inspect.signature(weight_loader)
    params = list(sig.parameters.keys())

    if len(params) == 2:
        # Simple weight_loader(param, loaded_weight) — handles TP sharding internally
        original_data = param.data.clone()
        param.data.zero_()
        weight_loader(param, delta_w)
        param.data += original_data
    else:
        # Unexpected signature, fall back to direct addition
        param.data += delta_w
