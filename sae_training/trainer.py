from torch.optim import AdamW
import torch.nn.functional as F
import torch
from tqdm import tqdm


class SAETrainer:
    def __init__(self, sae, dataloader, config):
        self.sae = sae
        self.dataloader = dataloader
        self.config = config

    # Should mean and abs be torch functions?
    def compute_loss(self, inputs, outputs, hidden):
        mse_loss = F.mse_loss(inputs, outputs)
        sparsity_loss = self.config.sparsity_coef * torch.mean(torch.abs(hidden))
        return mse_loss + sparsity_loss

    def train(self):
        optimizer = AdamW(self.sae.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epochs):
            for i, activations in enumerate(tqdm(self.dataloader)):
                activations = activations.to(self.config.device)
                optimizer.zero_grad()
                sae_output, sae_hidden = self.sae(activations)
                loss = self.compute_loss(activations, sae_output, sae_hidden)
                loss.backward()
                optimizer.step()
                tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")
