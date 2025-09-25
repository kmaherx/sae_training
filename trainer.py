from torch.optim import AdamW
import torch.nn.functional as F
import torch
from tqdm import tqdm


class SAETrainer:
    def __init__(self, sae, activations, config):
        self.sae = sae
        self.model_activations = activations
        self.config = config

    # Should mean and abs be torch functions?
    def compute_loss(self, inputs, outputs, hidden):
        mse_loss = F.mse_loss(inputs, outputs)
        sparsity_loss = self.config.sparsity_coef * torch.mean(torch.abs(hidden))
        return mse_loss + sparsity_loss

    def train(self):
        optimizer = AdamW(self.sae.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epochs):
            for sae_input in tqdm(self.model_activations):
                optimizer.zero_grad()
                sae_output, sae_hidden = self.sae(sae_input)
                loss = self.compute_loss(sae_input, sae_output, sae_hidden)
                loss.backward()
                optimizer.step()
                tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")
