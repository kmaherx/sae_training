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
    
    def step(self, batch):
        activations = batch.to(self.config.device)
        sae_output, sae_hidden = self.sae(activations)
        loss = self.compute_loss(activations, sae_output, sae_hidden)
        return loss

    def train(self):
        """Train loop. Returns training losses concatenated across epochs."""
        optimizer = AdamW(self.sae.parameters(), lr=self.config.learning_rate)
        losses = []

        for epoch in range(self.config.n_epochs):

            pbar = tqdm(self.dataloader, desc=f"Training for {self.config.n_samples} samples")

            for i, batch in enumerate(pbar):

                optimizer.zero_grad()
                loss = self.step(batch)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_postfix({"batch": i + 1, "loss": loss.item()})

                if i * self.config.batch_size >= self.config.n_samples:
                    break

        return torch.tensor(losses)
