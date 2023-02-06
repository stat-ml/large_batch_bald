from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

def init_glorot(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)