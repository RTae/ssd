from transformers import AutoTokenizer


# Infer model family based on model path name
def infer_model_family(model_path: str) -> str:
        """Infer if model is Llama or Qwen based on path name or config."""
        model_path_lower = model_path.lower()
        if "llama" in model_path_lower:
            return "llama"
        elif "qwen" in model_path_lower:
            return "qwen"
        else:
            # Fall back to checking model config if path name is ambiguous
            # (e.g. fine-tuned models with custom names)
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path)
                model_type = getattr(config, "model_type", "").lower()
                if "llama" in model_type:
                    return "llama"
                elif "qwen" in model_type:
                    return "qwen"
            except Exception:
                pass
            return "unknown"


def decode_tokens(token_ids: list[int], tokenizer: AutoTokenizer) -> list[str]:
    decoded = []
    for token in token_ids:
        try:
            text = tokenizer.decode([token], skip_special_tokens=False)
            decoded.append(text)
        except Exception:
            decoded.append(f"<token_id:{token}>")
    return decoded
