from sae import SAE
from trainer import SAETrainer
from data import load_data, StreamingDatasetWrapper

from transformers import AutoModel
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch


@dataclass
class SAETrainerConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "gpt2"
    d_sae: int = 12288
    epochs: int = 10
    learning_rate: float = 1e-3
    sparsity_coef: float = 1e-3
    dataset_name: str = "HuggingFaceFW/fineweb"
    n_ctx: int = 256
    batch_size: int = 512
    split: str = "test"


def main(
    device,
    model_name,
    d_sae,
    epochs,
    learning_rate,
    sparsity_coef,
    dataset_name,
    n_ctx,
    batch_size,
    split,
):
    config = SAETrainerConfig()
    config.device = device
    config.model_name = model_name
    config.d_sae = d_sae
    config.epochs = epochs
    config.learning_rate = learning_rate
    config.sparsity_coef = sparsity_coef
    config.dataset_name = dataset_name
    config.n_ctx = n_ctx
    config.batch_size = batch_size
    config.split = split

    model = AutoModel.from_pretrained(config.model_name)
    d_model = model.config.hidden_size
    sae = SAE(d_model, config)

    dataset = load_data(config.dataset_name)
    streaming_dataset = StreamingDatasetWrapper(dataset, model, config)
    dataloader = DataLoader(streaming_dataset, batch_size=config.batch_size)

    trainer = SAETrainer(sae, dataloader, config)
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--d_sae", type=int, default=12288)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sparsity_coef", type=float, default=1e-3)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--n_ctx", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    main(
        args.device,
        args.model_name,
        args.d_sae,
        args.epochs,
        args.learning_rate,
        args.sparsity_coef,
        args.dataset_name,
        args.n_ctx,
        args.batch_size,
        args.split,
    )
