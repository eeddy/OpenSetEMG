import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class AE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.encoder1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.encoder2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, tr_dl, learning_rate=1e-3, num_epochs=100, verbose=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

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
                print("Loss: " + str(loss.item()))
                # acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
                # # log it
                # self.log["training_loss"] += [(epoch, loss.item())]
                # self.log["training_accuracy"] += [(epoch, acc)]
            self.eval()

    