import io
import os
import torch
import numpy as np
from PIL import Image

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