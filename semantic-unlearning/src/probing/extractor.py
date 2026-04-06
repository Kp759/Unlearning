from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class HiddenStateExtractor:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        self.dtype = dtype_map.get(dtype, torch.float16)

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {model_name} ({dtype} on {device})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        print("Model loaded.")

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def extract(
        self,
        texts: List[str],
        batch_size: int = 8,
        layers: Optional[List[int]] = None,
        aggregate: str = "last",
    ) -> Dict[int, np.ndarray]:
        all_hidden: Dict[int, List[np.ndarray]] = {}

        for start in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
            batch = texts[start : start + batch_size]
            enc = self._tokenize(batch).to(self.device)
            attention_mask = enc["attention_mask"]

            with torch.no_grad():
                outputs = self.model(**enc, output_hidden_states=True)

            hidden_states = outputs.hidden_states  # tuple of (batch, seq, d_model)
            layer_indices = layers if layers is not None else list(range(len(hidden_states)))

            for layer_idx in layer_indices:
                hs = hidden_states[layer_idx].float()  # (batch, seq, d_model)

                if aggregate == "last":
                    # Index of the last real (non-padding) token per sample
                    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
                    pooled = hs[torch.arange(hs.size(0)), seq_lengths]  # (batch, d_model)
                elif aggregate == "mean":
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1)
                elif aggregate == "max":
                    mask = attention_mask.unsqueeze(-1).bool()
                    hs_masked = hs.masked_fill(~mask, float("-inf"))
                    pooled = hs_masked.max(dim=1).values
                else:
                    raise ValueError(f"Unknown aggregate mode: {aggregate}")

                arr = pooled.cpu().numpy().astype(np.float32)
                all_hidden.setdefault(layer_idx, []).append(arr)

        return {layer: np.concatenate(arrays, axis=0) for layer, arrays in all_hidden.items()}

    def extract_per_token(
        self,
        texts: List[str],
        batch_size: int = 8,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, List[np.ndarray]]:
        all_hidden: Dict[int, List[np.ndarray]] = {}

        for start in tqdm(range(0, len(texts), batch_size), desc="Extracting per-token states"):
            batch = texts[start : start + batch_size]
            enc = self._tokenize(batch).to(self.device)
            attention_mask = enc["attention_mask"]

            with torch.no_grad():
                outputs = self.model(**enc, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            layer_indices = layers if layers is not None else list(range(len(hidden_states)))

            for layer_idx in layer_indices:
                hs = hidden_states[layer_idx].float()  # (batch, seq, d_model)

                for i in range(hs.size(0)):
                    seq_len = int(attention_mask[i].sum().item())
                    token_hs = hs[i, :seq_len, :].cpu().numpy().astype(np.float32)
                    all_hidden.setdefault(layer_idx, []).append(token_hs)

        return all_hidden

    def get_token_ids(self, texts: List[str]) -> List[List[int]]:
        result = []
        for text in texts:
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
            result.append(enc["input_ids"])
        return result

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])
