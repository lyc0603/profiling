"""Script to process ENS names using BERT embeddings."""

import re
import json
import gzip
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from environ.constant import PROCESSED_DATA_PATH


# Load ENS domain data
with gzip.open(PROCESSED_DATA_PATH / "ens_domain.json.gz", "rb") as f:
    ens_domain = json.load(f)


# Load multilingual BERT
model_name = "bert-base-multilingual-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()


def is_hex_string(s: str) -> bool:
    """Return True if string is pure hex (length >= 16)."""
    return bool(re.fullmatch(r"[0-9a-fA-F]{16,}", s))


def clean_ens(ens: str):
    """Clean ENS string."""
    # remove suffixes
    if ens.endswith(".eth"):
        ens = ens[:-4]

    if ens.endswith(".addr.reverse"):
        ens = ens.replace(".addr.reverse", "")

    if ens == "reverse":
        return None

    # remove square brackets
    ens = ens.strip("[]")

    # skip hex-only names
    if is_hex_string(ens):
        return None

    return ens


def get_embedding(text: str):
    """Return mean pooled embedding."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooling
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts

    return mean_pooled.squeeze().cpu()


# Generate embeddings
embeddings = {}

for ens, addr in tqdm(ens_domain.items(), desc="Processing ENS"):
    cleaned = clean_ens(ens)
    if cleaned is None:
        continue

    try:
        emb = get_embedding(cleaned)
        embeddings[ens] = emb.numpy()
    except Exception:
        continue


# # Save
# torch.save(embeddings, PROCESSED_DATA_PATH / "ens_embeddings.pt")

# print(f"Saved {len(embeddings)} ENS embeddings.")
