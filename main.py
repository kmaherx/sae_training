from sae import SAE
from extractor import Extractor
from trainer import SAETrainer
from dataclasses import dataclass
from transformers import AutoTokenizer
from datasets import load_dataset

# Can dictionary be fixed?
EXPANSION_FACTOR = 16
D_MODEL = 768
D_SAE = D_MODEL * EXPANSION_FACTOR


@dataclass
class SAETrainerConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    sparsity_coef: float = 1e-3
    dataset_name: str = "HuggingFaceFW/fineweb"


DATA = ["Test data"] * 1000


def load_data(dataset_name):
    dataset = load_dataset(dataset_name, name="sample-10BT", streaming=True)

    return dataset


def main():
    sae = SAE(D_MODEL, D_SAE)
    extractor = Extractor("gpt2", "ln_f")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = tokenizer(DATA, return_tensors="pt").input_ids
    config = SAETrainerConfig()
    dataset = load_data(config.dataset_name)
    # print the first sample from the dataset
    print(next(iter(dataset)))
    # activations = extractor.extract(dataset_1000)
    # trainer = SAETrainer(sae, activations, config)
    # trainer.train()


if __name__ == "__main__":
    main()
