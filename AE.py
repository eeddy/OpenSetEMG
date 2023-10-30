import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn

"""
LSTM Autoencoder - Adapted from https://discuss.pytorch.org/t/lstm-autoencoders-in-pytorch/139727.
"""

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dimension=64):
        super(Encoder, self).__init__()

        self.layer1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dimension,
            num_layers=1,
            batch_first=True
        )
        self.layer2 = nn.LSTM(
            input_size=hidden_dimension,
            hidden_size=int(hidden_dimension/2),
            num_layers=1,
            batch_first=True
        )
        self.layer3 = nn.LSTM(
            input_size=int(hidden_dimension/2),
            hidden_size=int(hidden_dimension/4),
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)
        x, _ = self.layer3(x)
        return x 
    
    
class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dimension):
        super(Decoder, self).__init__()

        self.layer1 = nn.LSTM(
            input_size=int(hidden_dimension/4),
            hidden_size=int(hidden_dimension/4),
            num_layers=1,
            batch_first=True
        )
        self.layer2 = nn.LSTM(
            input_size=int(hidden_dimension/4),
            hidden_size=int(hidden_dimension/2),
            num_layers=1,
            batch_first=True
        )
        self.layer3 = nn.LSTM(
            input_size=int(hidden_dimension/2),
            hidden_size=hidden_dimension,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dimension, n_features)
        
    def forward(self, x):
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)
        x, _ = self.layer3(x)
        return self.output_layer(x)


class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden_dim=32):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(n_features, hidden_dim).to(self.device)
        self.decoder = Decoder(n_features, hidden_dim).to(self.device)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def fit(self, tr_dl, learning_rate=1e-5, num_epochs=100, verbose=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()
        tr_loss = []

        # now start the training
        for epoch in range(num_epochs):
            print("Epoch: " + str(epoch) + " ...")
            #training set
            self.train()
            for data, labels in tr_dl:
                optimizer.zero_grad()
                data = data.to(device)
                output = self.forward(data)
                loss = loss_function(output, data)
                loss.backward()
                optimizer.step()
                tr_loss += [loss.item()]
            print("Loss: " + str(loss.item()))
        print("HERE")
    
    def predict(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        y = self.forward(x)
        errors = []
        for i,v in enumerate(y):
            errors.append(metrics.mean_squared_error(v.detach().numpy(),x[i]))
        return np.array(errors)

    