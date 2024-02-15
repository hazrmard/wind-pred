import pandas as pd
import torch
import torch.nn as nn
from tqdm.autonotebook import trange, tqdm


def train_lstm(
    model: nn.Module, data: pd.DataFrame, epochs=1, batch_size=100, lr=0.001, **kwargs
):
    """
    Train an LSTM model on the given data.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in trange(epochs, leave=False, desc='Training'):
        l = []
        for i in range(0, len(data)-batch_size, batch_size):
            batch = torch.from_numpy(data.iloc[i:i+batch_size].values).float().reshape(batch_size, 1, 1)
            optimizer.zero_grad()
            output= model(batch)
            loss = criterion(output, torch.from_numpy(data.iloc[i+1:i+batch_size+1].values).float().reshape(batch_size, 1))
            loss.backward()
            optimizer.step()
            l.append(loss.item())
        losses.append(sum(l)/len(l))
    return losses



class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=10, num_layers=2, batch_first=False)
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        # x = [1,2,3,4,5,6,.....]
        x, _ = self.lstm(x.view(len(x), -1, 1))
        # x = [[1,2,3,...10],
        #      [1,2,3,...10],
        #      [1,2,3,...10],
        #]``
        x = self.linear(x).reshape(len(x), -1)
        # x = [1,2,3,4,5,6,.....]
        return x
