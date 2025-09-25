from extractor import Extractor

from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import IterableDataset


def load_data(dataset_name):
    dataset = load_dataset(dataset_name, name="sample-10BT", streaming=True)
    return dataset


class StreamingDatasetWrapper(IterableDataset):
    def __init__(self, hf_streaming_dataset, config):
        self.dataset = hf_streaming_dataset["train"]
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.n_ctx = config.n_ctx
        self.config = config
        self.extractor = Extractor("gpt2", "ln_f", config.device)

    def __iter__(self):
        for sample in self.dataset:
            string = sample["text"]
            data = self.tokenizer(
                string,
                max_length=self.n_ctx,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).input_ids.to(self.config.device)
            activations = self.extractor.extract(data)
            yield activations.squeeze(0)
