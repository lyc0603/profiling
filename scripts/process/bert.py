"""
Minimal working example: generate ENS embeddings with multilingual BERT.

Hardcoded:
- Example ENS names
- Model: bert-base-multilingual-cased

pip install -U transformers torch
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    last_hidden_state: (B, T, H)
    attention_mask:    (B, T)
    returns:           (B, H)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return summed / counts


if __name__ == "__main__":
    # Hardcoded ENS examples (mixing scripts/emojis to illustrate multilingual tokenization)
    ens_list = [
        # "vitalik.eth",
        # "alice.eth",
        # "東京.eth",
        # "中文名.eth",
        # "crypto🦄.eth",
        # "résumé.eth",
        # "0xdeadbeef.eth",
        "北京",
        "苹果",
        "apple",
    ]

    # Hardcoded multilingual BERT model
    model_name = "bert-base-multilingual-cased"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Tokenize
    batch = tokenizer(
        ens_list,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**batch)
        # out.last_hidden_state: (B, T, H)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])  # (B, H)
        emb = F.normalize(emb, p=2, dim=1)  # L2-normalize for cosine similarity

    print("Model:", model_name)
    print("ENS count:", len(ens_list))
    print("Embedding shape:", tuple(emb.shape))  # (B, H), H=768 for mBERT

    # Cosine similarity matrix
    sim = emb @ emb.T
    print("\nCosine similarity matrix:")
    # Pretty print
    header = ["{:>12}".format(" ")] + ["{:>12}".format(e[:10]) for e in ens_list]
    print("".join(header))
    for i, e in enumerate(ens_list):
        row = ["{:>12}".format(e[:10])]
        row += ["{:12.3f}".format(sim[i, j].item()) for j in range(len(ens_list))]
        print("".join(row))

    # Example: mapping ENS -> embedding vector (CPU numpy)
    ens2vec = {ens: emb[i].detach().cpu().numpy() for i, ens in enumerate(ens_list)}
    print("\nOne embedding example (first 8 dims):")
    first = ens_list[0]
    print(first, ens2vec[first][:8])
