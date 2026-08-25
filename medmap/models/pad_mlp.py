"""PAD-UFES-20 classifier head: a one-hidden-layer MLP on top of the frozen
EfficientNet-B3 image features concatenated with the one-hot clinical
metadata. Kaiming-normal init, zero biases.

To swap in a different model, any ``nn.Module`` mapping the combined feature
vector to ``num_classes`` logits works — the training loops in
``medmap.pad.privacy`` compute per-sample gradients functionally
(``torch.func``) over all trainable parameters, so no per-layer hooks are
required.
"""

import torch.nn as nn

NUM_CLASSES = 6


class MLP1Layer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
