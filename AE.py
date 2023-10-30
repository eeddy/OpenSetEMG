import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn

from soft_dtw_cuda import SoftDTW

"""
LSTM Autoencoder - Adapted from https://discuss.pytorch.org/t/lstm-autoencoders-in-pytorch/139727.
"""

class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dimension=64):
        super(Encoder, self).__init__()

        self.layer1 = nn.LSTM(
            input_size=hidden_dimension,
            hidden_size=int(hidden_dimension),
            num_layers=1,
            batch_first=True
        )
        self.layer2 = nn.LSTM(
            input_size=int(hidden_dimension),
            hidden_size=int(7),
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)
        return x 
    
    
class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dimension):
        super(Decoder, self).__init__()

        self.layer1 = nn.LSTM(
            input_size=int(7),
            hidden_size=int(7),
            num_layers=1,
            batch_first=True
        )
        self.layer2 = nn.LSTM(
            input_size=int(7),
            hidden_size=int(hidden_dimension),
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dimension, n_features)
        
    def forward(self, x):
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)
        return self.output_layer(x)


class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden_dim=8):
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
        scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
        loss_function = nn.MSELoss() #SoftDTW(use_cuda=False, gamma = 1e-3)

        # now start the training
        for epoch in range(num_epochs):
            epoch_loss = []
            print("Epoch: " + str(epoch) + " ...")
            #training set
            self.train()
            mean_losses = []
            
            for data, labels in tr_dl:
                optimizer.zero_grad()
                data = data.to(device)
                output = self.forward(data)
                loss = loss_function(output, data)
                mean_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            print("Mean Losses: " + str(np.mean(mean_losses)))
            #scheduler.step()
            # print("Loss: " + str(loss.item()))
            #avg_loss = sum(epoch_loss)/len(epoch_loss)
            #print("Loss: " + str(avg_loss))
            #epochs_loss_by_batch += [avg_loss]
    
    def predict(self, x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        y = self.forward(x.to(device))
        errors = []
        for i,v in enumerate(y):
            errors.append(metrics.mean_squared_error(v.cpu().detach().numpy(),x[i].cpu()))
        return np.array(errors)
    
    def predict_data(self, x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        return self.forward(x.to(device))