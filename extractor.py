# Is it easier with TransformerLens?
from transformers import AutoModel
import torch


class Extractor:
    def __init__(self, model, layer_name):
        self.model = AutoModel.from_pretrained(model)
        self.layer_name = layer_name
        self.activations = None
        self.hook = None
        self.model.eval()

    def hook_fn(self, module, input, output):
        self.activations = output.detach()

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.hook = module.register_forward_hook(self.hook_fn)

    def clear_hook(self):
        if self.hook:
            self.hook.remove()

    def extract(self, data):
        self.register_hook()

        with torch.no_grad():
            self.model(data)

        self.clear_hook()
        return self.activations
