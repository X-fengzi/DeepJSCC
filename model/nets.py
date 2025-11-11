import torch
import torch.nn as nn

class ToyNet(nn.Module):
    def __init__(self, dims= []):
        super(ToyNet, self).__init__()
        self.input_dim, self.output_dim = dims
        self.init_layers()

    def init_layers(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(), nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, self.output_dim)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)

    def forward(self, X):
        # X = X.view(X.shape[0],-1)
        return self.layers(X)

class LinearModel(nn.Module):
    def __init__(self, dims= []):
        super(LinearModel, self).__init__()
        self.input_dim, self.output_dim = dims
        self.init_layers()

    def init_layers(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)

    def forward(self, X):
        return self.layers(X)

class Classifier(nn.Module):
    def __init__(self, dims= []):
        super(Classifier, self).__init__()
        self.input_dim, self.output_dim = dims
        self.init_layers()

    def init_layers(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, self.output_dim))
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)

    def forward(self, X):
        return self.layers(X)

class ThreeLayerMLP(nn.Module):
    def __init__(self, dims= []):
        super(ThreeLayerMLP, self).__init__()
        self.input_dim, self.output_dim = dims
        self.hidden_dim  = 4*self.input_dim
        self.init_layers()

    def init_layers(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)

    def forward(self, X):
        # X = X.view(X.shape[0],-1)
        return self.layers(X)

class TwoLayerMLP(nn.Module):
    def __init__(self, dims= []):
        super(TwoLayerMLP, self).__init__()
        self.input_dim, self.output_dim = dims
        self.init_layers()

    def init_layers(self):
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*4),
            nn.ReLU(True),
            nn.Linear(self.input_dim*4, self.output_dim),

        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None: nn.init.zeros_(m.bias.data)

    def forward(self, X):
        return self.layers(X)



#[TODO] +++ any other architectures. +++

