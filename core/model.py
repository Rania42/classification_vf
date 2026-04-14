"""
Chargement du modèle EfficientNet+mBERT et utilitaires de classification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

from config import MODEL_PATH, DEVICE

# ── Architecture ───────────────────────────────────────
class MultimodalBankClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vision_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        visual_dim = self.vision_model.num_features
        self.text_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        text_dim = self.text_model.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + text_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        v = self.vision_model(image)
        t = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.fusion(torch.cat([v, t], dim=1))


# ── Chargement ─────────────────────────────────────────
print(f"\n[INIT] Device : {DEVICE}")
print("[INIT] Chargement du modèle...")

checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
label_enc   = checkpoint["label_encoder"]
num_classes = checkpoint["num_classes"]
img_size    = checkpoint["img_size"]
max_text    = checkpoint["max_text_len"]
val_acc     = checkpoint.get("val_acc", 0.0)

_thr_raw = checkpoint.get("confidence_threshold", 70.0)
CONFIDENCE_THRESHOLD = _thr_raw / 100.0 if _thr_raw > 1.0 else _thr_raw

# 7 classes reconnues par le modèle
DOC_CLASSES = list(label_enc.classes_)

model = MultimodalBankClassifier(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print(f"[INIT] Classes : {DOC_CLASSES}")
print(f"[INIT] Val acc : {val_acc:.4f}  |  Seuil : {CONFIDENCE_THRESHOLD:.0%}")


# ── Inférence ──────────────────────────────────────────
def classify_with_model(img_path: str, text: str):
    """
    Retourne (pred_class, confidence, all_scores).
    all_scores = liste triée [{"class": ..., "score": ...}]
    """
    from PIL import Image
    image      = Image.open(img_path).convert("RGB")
    img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)
    tok = tokenizer(
        text, max_length=max_text,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            img_tensor,
            tok["input_ids"].to(DEVICE),
            tok["attention_mask"].to(DEVICE),
        )
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = int(np.argmax(probs))
    pred_class = label_enc.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    all_scores = sorted(
        [{"class": label_enc.classes_[i], "score": float(probs[i])}
         for i in range(len(label_enc.classes_))],
        key=lambda x: x["score"], reverse=True,
    )
    return pred_class, confidence, all_scores