import io
import os
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

class SmolVLMHeteroNoDummyCollator:
    def __init__(self, processor, text_key: str = "prompt", image_key: str = "image_path",
                 max_length: int | None = None, want_image: bool = True, want_text: bool = True):
        self.processor = processor
        self.text_key = text_key
        self.image_key = image_key
        self.max_length = max_length
        self.want_image = bool(want_image)
        self.want_text = bool(want_text)
        tok = getattr(processor, "tokenizer", None)
        self.image_token = getattr(tok, "image_token", "<image>")

    def _load_pil(self, obj):
        if obj is None: return None
        try:
            if isinstance(obj, Image.Image): return obj.convert("RGB")
            if isinstance(obj, dict):
                b, p = obj.get("bytes"), obj.get("path")
                if b is not None: return Image.open(io.BytesIO(b)).convert("RGB")
                if p and os.path.exists(p): return Image.open(p).convert("RGB")
            if isinstance(obj, (bytes, bytearray)): return Image.open(io.BytesIO(obj)).convert("RGB")
            if isinstance(obj, str) and os.path.exists(obj): return Image.open(obj).convert("RGB")
            if isinstance(obj, np.ndarray):
                if obj.ndim == 2: obj = np.stack([obj]*3, -1)
                return Image.fromarray(obj.astype("uint8")).convert("RGB")
            if torch.is_tensor(obj):
                t = obj.detach().cpu()
                if t.ndim == 2: t = t.unsqueeze(-1).repeat(1, 1, 3)
                if t.ndim == 3 and t.shape[0] in (1, 3, 4): t = t.permute(1, 2, 0)
                arr = t.numpy().astype("uint8")
                return Image.fromarray(arr).convert("RGB")
        except Exception:
            return None
        return None

    def _normalize_image_tokens(self, text: str, k: int) -> str:
        token = self.image_token
        s = text or ""
        if k == 0: return " ".join(s.replace(token, " ").split())
        out, used, i = [], 0, 0
        while True:
            j = s.find(token, i)
            if j < 0:
                out.append(s[i:]); break
            out.append(s[i:j])
            if used < k:
                out.append(token); used += 1
            i = j + len(token)
        s2 = "".join(out)
        if used < k: s2 = ((token + " ") * (k - used)) + s2
        return " ".join(s2.split())

    def _null_text(self, k: int) -> str:
        if k <= 0: return ""
        return " ".join([self.image_token] * k)

    def __call__(self, batch):
        tok = self.processor.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

        texts, labels, pil_lists = [], [], []
        for ex in batch:
            raw_text = ex.get(self.text_key, "") or ""
            img_field = ex.get(self.image_key, None)

            imgs = [] if img_field is None else (list(img_field) if isinstance(img_field, (list, tuple)) else [img_field])

            pil_list = []
            if self.want_image:
                pil_list = [p for p in [self._load_pil(x) for x in imgs] if p is not None]
            k = len(pil_list)

            if self.want_text and self.want_image:
                norm_text = self._normalize_image_tokens(raw_text, k=k)
            elif self.want_text and not self.want_image:
                norm_text = self._normalize_image_tokens(raw_text, k=0)
            elif (not self.want_text) and self.want_image:
                norm_text = self._null_text(k)
            else:
                raise ValueError("At least one of want_text or want_images must be True.")

            texts.append(norm_text)
            pil_lists.append(pil_list)
            labels.append(int(ex["class_idx"]))

        B = len(batch)
        labels = torch.tensor(labels, dtype=torch.long)

        idx_img = [i for i, lst in enumerate(pil_lists) if len(lst) > 0]
        idx_txt = [i for i, lst in enumerate(pil_lists) if len(lst) == 0]

        text_kwargs = dict(return_tensors="pt", padding=True)
        if self.max_length is not None:
            text_kwargs.update(dict(truncation=True, max_length=self.max_length))

        if idx_img:
            enc_img = self.processor(text=[texts[i] for i in idx_img], images=[pil_lists[i] for i in idx_img], **text_kwargs)
        if idx_txt:
            enc_txt = self.processor(text=[texts[i] for i in idx_txt], **text_kwargs)

        T_img = int(enc_img["input_ids"].shape[1]) if idx_img else 0
        T_txt = int(enc_txt["input_ids"].shape[1]) if idx_txt else 0
        T = max(T_img, T_txt, 1)

        if idx_img and "pixel_values" in enc_img:
            pv = enc_img["pixel_values"]
            B_img, M, C, H, W = pv.shape
        else:
            M = C = H = W = 0

        device = enc_img["input_ids"].device if idx_img else (enc_txt["input_ids"].device if idx_txt else "cpu")
        id_dtype, mask_dtype = torch.long, torch.long

        input_ids = torch.full((B, T), pad_id, dtype=id_dtype, device=device)
        attention_mask = torch.zeros((B, T), dtype=mask_dtype, device=device)

        if M > 0:
            pixel_values = torch.zeros((B, M, C, H, W), dtype=enc_img["pixel_values"].dtype, device=device)
            pixel_attention_mask = torch.zeros((B,) + enc_img["pixel_attention_mask"].shape[1:], dtype=enc_img["pixel_attention_mask"].dtype, device=device) if "pixel_attention_mask" in enc_img else None
        else:
            pixel_values = pixel_attention_mask = None

        if idx_img:
            ii, am = enc_img["input_ids"].to(dtype=id_dtype), enc_img["attention_mask"].to(dtype=mask_dtype)
            input_ids[idx_img, :ii.shape[1]] = ii
            attention_mask[idx_img, :am.shape[1]] = am
            if pixel_values is not None: pixel_values[idx_img] = enc_img["pixel_values"]
            if pixel_attention_mask is not None: pixel_attention_mask[idx_img] = enc_img["pixel_attention_mask"].to(pixel_attention_mask.dtype)

        if idx_txt:
            ii, am = enc_txt["input_ids"].to(dtype=id_dtype), enc_txt["attention_mask"].to(dtype=mask_dtype)
            input_ids[idx_txt, :ii.shape[1]] = ii
            attention_mask[idx_txt, :am.shape[1]] = am

        enc = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "has_image_flags": torch.tensor([len(p)>0 for p in pil_lists], dtype=torch.bool, device=device),
            "has_text_flags": torch.tensor([len(self._normalize_image_tokens(t, k=0)) > 0 for t in texts], dtype=torch.bool, device=device),
        }
        if pixel_values is not None: enc["pixel_values"] = pixel_values
        if pixel_attention_mask is not None: enc["pixel_attention_mask"] = pixel_attention_mask

        pos = torch.arange(input_ids.size(1), device=device)
        enc["position_ids"] = pos.unsqueeze(0).repeat(B, 1)

        return enc
    
@dataclass
class Gemma3HeteroNoDummyCollator:
    processor: Any
    text_key: str = "text"
    image_key: str = "image"
    labels_key: str = "labels"

    max_length: int = 512
    num_labels: int = 12
    pad_to_multiple_of: Optional[int] = 8

    want_image: bool = True
    want_text: bool = True

    # If a sample ends up with neither image nor text (e.g. want_image=True but image missing AND want_text=False),
    # we fallback to empty text to keep batch size stable.
    fallback_empty_text: str = ""

    def _as_image_list(self, img) -> List[Any]:
        if img is None:
            return []
        if isinstance(img, (list, tuple)):
            return [x for x in img if x is not None]
        return [img]

    def _prompt_with_boi(self, text: str, n_imgs: int, use_text: bool) -> str:
        boi = getattr(self.processor, "boi_token", None) or getattr(self.processor.tokenizer, "boi_token", None)
        if boi is None:
            raise RuntimeError("Processor/tokenizer has no boi_token; cannot build Gemma3 multimodal prompts.")

        if n_imgs <= 0:
            return text if use_text else self.fallback_empty_text

        boi_block = " ".join([boi] * n_imgs)
        if use_text and text.strip():
            # Put image placeholders first (simple + robust). You can swap order if you prefer.
            return f"{boi_block} {text.strip()}"
        return boi_block

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tok = self.processor.tokenizer
        drop_keys = {"pixel_values", "num_crops"}  # do NOT feed these to tokenizer.pad
        KEEP_KEYS = ("input_ids", "attention_mask", "token_type_ids")

        B = len(features)

        # --- collect labels (kept aligned with original batch order) ---
        labels_list = []
        for i, f in enumerate(features):
            y = f.get(self.labels_key, None)
            if y is None:
                raise ValueError(f"Missing '{self.labels_key}' in a sample.")
            if len(y) != self.num_labels:
                raise ValueError(f"Example {i} has {len(y)} labels, expected {self.num_labels}.")
            # HF datasets may give None for missing floats inside sequences -> convert to NaN
            y = [float("nan") if v is None else float(v) for v in y]

            if len(y) != self.num_labels:
                raise ValueError(f"Example {i} has {len(y)} labels, expected {self.num_labels}.")

            labels_list.append(torch.tensor(y, dtype=torch.float32))


        # --- split into image-capable vs text-only samples (preserve original order) ---
        img_map: List[int] = []
        img_texts: List[str] = []
        img_nested_images: List[List[Any]] = []  # nested list per sample

        txt_map: List[int] = []
        txt_texts: List[str] = []

        for i, f in enumerate(features):
            raw_text = (f.get(self.text_key, "") or "")
            raw_img = f.get(self.image_key, None)

            imgs = self._as_image_list(raw_img)
            has_img = len(imgs) > 0
            has_txt = bool(raw_text.strip())

            use_img = self.want_image and has_img
            use_txt = self.want_text and has_txt

            if use_img:
                prompt = self._prompt_with_boi(raw_text, n_imgs=len(imgs), use_text=use_txt)
                img_map.append(i)
                img_texts.append(prompt)
                img_nested_images.append(imgs)
            else:
                # text-only path (even if want_text=False, we must keep batch stable)
                prompt = raw_text.strip() if use_txt else self.fallback_empty_text
                txt_map.append(i)
                txt_texts.append(prompt)

        per_sample: List[Optional[Dict[str, Any]]] = [None] * B

        # --- encode text-only samples ---
        if txt_map:
            enc_txt = self.processor(
                text=txt_texts,
                images=None,
                truncation=True,
                max_length=self.max_length,
                padding="longest",   # important for gemma3 processor
                return_tensors=None,
            )
            for j, orig_i in enumerate(txt_map):
                per_sample[orig_i] = {k: enc_txt[k][j] for k in KEEP_KEYS if k in enc_txt}

        # --- encode image samples ---
        pixel_values = None
        if img_map:
            enc_img = self.processor(
                text=img_texts,
                images=img_nested_images,   # nested list per sample
                truncation=True,
                max_length=self.max_length,
                padding="longest",   # important for gemma3 processor
                return_tensors=None,
            )
            for j, orig_i in enumerate(img_map):
                per_sample[orig_i] = {k: enc_img[k][j] for k in KEEP_KEYS if k in enc_img}

            pv = enc_img.get("pixel_values", None)
            if pv is not None:
                if torch.is_tensor(pv):
                    pixel_values = pv
                else:
                    pv_t = [torch.as_tensor(x) for x in pv]
                    pixel_values = torch.stack(pv_t, dim=0)

        # Safety: should never happen because every sample goes to img_map or txt_map
        for i in range(B):
            if per_sample[i] is None:
                per_sample[i] = tok(self.fallback_empty_text, add_special_tokens=True)

        # --- pad to batch tensors ---
        batch = tok.pad(
            per_sample,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.stack(labels_list, dim=0)

        # IMPORTANT: pixel_values is FLATTENED images (N_images, C, H, W), not (B, ...)
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        return dict(batch)
