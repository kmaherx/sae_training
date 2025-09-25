from sae import SAE
from trainer import SAETrainer
from dataclasses import dataclass
from data import load_data, StreamingDatasetWrapper
from torch.utils.data import DataLoader

# Can dictionary be fixed?
EXPANSION_FACTOR = 16
D_MODEL = 768
D_SAE = D_MODEL * EXPANSION_FACTOR


@dataclass
class SAETrainerConfig:
    d_sae: int = D_SAE
    epochs: int = 1
    learning_rate: float = 1e-3
    sparsity_coef: float = 1e-3
    dataset_name: str = "HuggingFaceFW/fineweb"
    n_ctx: int = 256
    batch_size: int = 64


def main():
    sae = SAE(D_MODEL, D_SAE)
    config = SAETrainerConfig()
    dataset = load_data(config.dataset_name)
    streaming_dataset = StreamingDatasetWrapper(dataset, config)
    dataloader = DataLoader(streaming_dataset, batch_size=config.batch_size)
    trainer = SAETrainer(sae, dataloader, config)
    trainer.train()


if __name__ == "__main__":
    main()
