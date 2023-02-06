from torch import nn

def init_glorot(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)