from unicodedata import bidirectional
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class LSTM(nn.Module):
    def __init__(self, n_classes, n_features, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=3, batch_first=True, bidirectional=True)
        self.linear_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward_nosm(self, x):
        output, _ = self.lstm(out)
        out = output[:, -1, :]
        out = self.relu(out)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out 

    def forward(self, x):
        out = self.forward_nosm(x)
        return self.softmax(out)

    def fit(self, tr_dl, learning_rate=1e-3, num_epochs=100, verbose=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        # Logger:
        self.log = {"training_loss":[],
                    "validation_loss": [],
                    "training_accuracy": [],
                    "validation_accuracy": []} 
        # now start the training
        for epoch in range(num_epochs):
            #training set
            self.train()
            for data, labels in tr_dl:
                optimizer.zero_grad()
                data = data.to(device)
                labels = labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
                # log it
                self.log["training_loss"] += [(epoch, loss.item())]
                self.log["training_accuracy"] += [(epoch, acc)]
            self.eval()
            if verbose:
                    epoch_trloss = np.mean([i[1] for i in self.log['training_loss'] if i[0]==epoch])
                    epoch_tracc  = np.mean([i[1] for i in self.log['training_accuracy'] if i[0]==epoch])
                    print(f"{epoch}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}")
        self.eval()