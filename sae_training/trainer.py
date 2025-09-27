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
        """
        Train loop. Returns training losses concatenated across epochs.
        Rounds n_samples to nearest multiple of batch_size.
        """
        optimizer = AdamW(self.sae.parameters(), lr=self.config.learning_rate)
        losses = []

        for epoch in range(self.config.n_epochs):

            n_batches_rounded = self.config.n_samples // self.config.batch_size
            n_samples_rounded = n_batches_rounded * self.config.batch_size
            pbar = tqdm(range(n_samples_rounded), desc=f"Training for {n_samples_rounded} samples")

            for i in range(n_batches_rounded):

                batch = next(iter(self.dataloader))
                loss = self.step(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.update(self.config.batch_size)
                pbar.set_postfix({"samples": (i + 1) * self.config.batch_size, "loss": loss.item()})
            pbar.close()

        return torch.tensor(losses)
