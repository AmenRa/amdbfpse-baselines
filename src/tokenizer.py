from torch import Tensor
from transformers import BertTokenizerFast


class Tokenizer:
    def __init__(self, name: str, max_len: int = 512):
        self.tokenizer = BertTokenizerFast.from_pretrained(
            name, local_files_only=True
        )
        self.max_len = max_len

    def __call__(self, batch_text: list[str]) -> dict[str, Tensor]:
        assert type(batch_text) == list and all(
            type(x) == str for x in batch_text
        ), "Error: `batch_text` should be a list of strings."

        return self.tokenizer(
            text=batch_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
