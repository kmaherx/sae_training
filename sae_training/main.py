# TODO: calculate and save feature activations
# TODO: add type hints, some docstrings, refactor config to separate file

import os

from transformers import AutoModel
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch

from sae import SAE
from trainer import SAETrainer
from data import load_data, StreamingDatasetWrapper


@dataclass
class SAETrainerConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "gpt2"
    d_sae: int = 12_288
    n_epochs: int = 10
    n_samples: int = 10_000
    learning_rate: float = 1e-3
    sparsity_coef: float = 1e-3
    dataset_name: str = "HuggingFaceFW/fineweb"
    hook_point: str = "ln_f"
    n_ctx: int = 256
    batch_size: int = 512
    split: str = "train"
    data_config: str = "sample-10BT"


def main(
    device,
    model_name,
    d_sae,
    n_epochs,
    n_samples,
    learning_rate,
    sparsity_coef,
    dataset_name,
    hook_point,
    n_ctx,
    batch_size,
    split,
    data_config,
    folder_name,
) -> None:
    config = SAETrainerConfig()
    config.device = device
    config.model_name = model_name
    config.d_sae = d_sae
    config.n_epochs = n_epochs
    config.n_samples = n_samples
    config.learning_rate = learning_rate
    config.sparsity_coef = sparsity_coef
    config.dataset_name = dataset_name
    config.hook_point = hook_point
    config.n_ctx = n_ctx
    config.batch_size = batch_size
    config.split = split
    config.data_config = data_config

    model = AutoModel.from_pretrained(config.model_name)
    d_model = model.config.hidden_size
    sae = SAE(d_model, config)

    dataset = load_data(config.dataset_name, config.split, config.data_config)
    streaming_dataset = StreamingDatasetWrapper(dataset, model, config)
    dataloader = DataLoader(streaming_dataset, batch_size=config.batch_size)

    trainer = SAETrainer(sae, dataloader, config)
    losses = trainer.train()

    file_name = f"losses_{config.model_name.replace('/', '_')}_" + \
                f"d{config.d_sae}_e{config.n_epochs}_n{config.n_samples}_" + \
                f"lr{config.learning_rate}_s{config.sparsity_coef}_".replace(".", "-") + \
                f"{config.hook_point}_ctx{config.n_ctx}_" + \
                f"bs{config.batch_size}_{config.split}_" + \
                f"{config.dataset_name.replace('/', '_')}_" + \
                f"{config.data_config.replace('/', '_')}" + \
                ".pt"
    os.makedirs(folder_name, exist_ok=True)
    write_path = os.path.join(folder_name, file_name)
    torch.save(losses, write_path)
    print(f"Losses saved to {write_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--d_sae", type=int, default=12_288)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sparsity_coef", type=float, default=1e-3)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--hook_point", type=str, default="ln_f")
    parser.add_argument("--n_ctx", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data_config", type=str, default="sample-10BT")
    parser.add_argument("--folder_name", type=str, default="./data/")
    args = parser.parse_args()

    main(
        args.device,
        args.model_name,
        args.d_sae,
        args.n_epochs,
        args.n_samples,
        args.learning_rate,
        args.sparsity_coef,
        args.dataset_name,
        args.hook_point,
        args.n_ctx,
        args.batch_size,
        args.split,
        args.data_config,
        args.folder_name,
    )

    print("Done")