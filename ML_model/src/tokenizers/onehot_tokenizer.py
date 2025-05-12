import torch
from typing import List


class FastOneHotPreprocessor:
    def __init__(self, seq_len: int = 200, vocab: str = "ACGT", padding: str = "right"):
        self.vocab = {ch: i for i, ch in enumerate(vocab)}
        self.seq_len = seq_len
        self.padding = padding  # "right" or "left"

    def __call__(self, sequences: List[str]) -> torch.Tensor:
        pad_id = len(self.vocab)  # treat pad as [UNK] index
        indices = torch.full((len(sequences), self.seq_len), fill_value=pad_id, dtype=torch.long)

        for i, seq in enumerate(sequences):
            seq = seq.upper()[:self.seq_len]
            L = len(seq)
            for j, base in enumerate(seq):
                idx = j if self.padding == "right" else self.seq_len - L + j
                indices[i, idx] = self.vocab.get(base, pad_id)

        one_hot = torch.nn.functional.one_hot(indices, num_classes=pad_id + 1).movedim(-1, 1)
        return one_hot[:, :-1, :]  # drop the [PAD]/UNK channel
