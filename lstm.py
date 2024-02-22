import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.autonotebook import trange, tqdm


class Model(nn.Module):

    def __init__(self, in_features, out_features, hidden_size=10, num_layers=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_linear = nn.Linear(in_features, hidden_size)
        self.activation = nn.Tanh()
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False
        )
        self.linear = nn.Linear(hidden_size, out_features)
    
    def forward(self, x, hidden: tuple=None):
        x = x.view(len(x), -1, self.in_features)
        assert x.shape[-1] == self.in_features, f"Expected {self.num_features} features, got {x.shape[-1]}"
        if hidden is not None:
            c, h = hidden
            assert c.shape == h.shape, "Expected hidden states to have the same shape"
            assert c.shape == (self.num_layers, x.shape[1], self.hidden_size), f"Expected hidden state shape {(self.num_layers, x.shape[1], self.hidden_size)}, got {c.shape}"  
        # x = [1,2,3,4,5,6,.....]
        x = self.init_linear(x)
        x = self.activation(x)
        x, hidden = self.lstm(x, hidden)
        # x = [[1,2,3,...10],
        #      [1,2,3,...10],
        #      [1,2,3,...10],
        #]``
        x = self.linear(x)
        # x = [1,2,3,4,5,6,.....]
        return x, hidden


    def predict(self, x, hidden: tuple=None):
        x = torch.from_numpy(x).float()
        y = np.zeros((len(x), self.out_features))
        for i in range(len(x)):
            y_, hidden = self.forward(x[i:i+1], hidden)
            y[i] = y_.detach().numpy().reshape(-1, self.out_features)[-1, :]
        return y


def train_lstm(
    model: Model, data: pd.DataFrame, epochs=1, window_length=100, sliding_step=1, lr=0.001, **kwargs
):
    """
    Train an LSTM model on the given data.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    if isinstance(data, pd.Series):
        data = [pd.DataFrame(data)]
    elif isinstance(data, pd.DataFrame):
        data = [data]
    for epoch in trange(epochs, leave=False, desc='Training'):
        l = []
        for d in data:
            for i in range(0, len(d)-window_length-1, sliding_step):
                batch = torch.from_numpy(d.iloc[i:i+window_length].values).float().reshape(window_length, 1, model.in_features)
                optimizer.zero_grad()
                output, _ = model(batch)
                loss = criterion(output, torch.from_numpy(d.iloc[i+1:i+1+window_length].values).float().reshape(window_length, 1, model.out_features))
                loss.backward()
                optimizer.step()
                l.append(loss.item())
        # losses.append(sum(l)/len(l))
        losses.extend(l)
    return losses
