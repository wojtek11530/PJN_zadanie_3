from typing import List, Tuple, Dict

import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, BatchEncoding
import torch

from src.utils import get_label_encoder


class TextDatasetForTransformers(Dataset):
    def __init__(self, filepath: str, tokenizer: PreTrainedTokenizerBase, max_seq_length: int = 128):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.encodings, text_labels = self.get_encodings_and_labels(self.read_txt(filepath))

        self.label_encoder = get_label_encoder(text_labels)
        self.labels = self.label_encoder.transform(text_labels).astype(np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def read_txt(input_file: str) -> List[str]:
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='UTF-8') as f:
            lines = f.read().splitlines()
        return lines

    def get_encodings_and_labels(self, lines: List[str]) -> Tuple[BatchEncoding, np.ndarray]:
        texts = []
        labels = []

        for (i, line) in enumerate(lines):
            split_line = line.split('__label__')
            text = split_line[0]
            label = split_line[1]
            texts.append(text)
            labels.append(label)

        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.max_seq_length,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        return encodings, np.array(labels)
