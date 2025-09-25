from torch import nn


class SAE(nn.Module):
    def __init__(self, d_model, config):
        super().__init__()
        self.d_model = d_model
        self.d_sae = config.d_sae
        self.encoder = nn.Linear(d_model, self.d_sae, bias=True)  # does this auto handle batch dim?
        self.decoder = nn.Linear(self.d_sae, d_model, bias=True)
        self.relu = nn.ReLU()
        self.to(config.device)

    def forward(self, x):
        x = self.encoder(x)
        hidden = self.relu(x)
        output = self.decoder(hidden)
        return output, hidden
