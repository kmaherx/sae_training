from torch import nn


class SAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        hidden = self.relu(x)
        output = self.decoder(hidden)
        return output, hidden
