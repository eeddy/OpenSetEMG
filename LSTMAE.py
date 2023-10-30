# Third Party
import torch
import torch.nn as nn
import torch.optim as optim


############
# COMPONENTS
############


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        x = x.unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.mm(x.squeeze(0), self.dense_matrix)


######
# MAIN
######


class LSTMAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(LSTMAE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x
    
    def train_model(self, train_set, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.MSELoss() #SoftDTW(use_cuda=False, gamma = 1e-3)

    
    #     for epoch in range(1, epochs + 1):
    #         model.train()

    #         # # Reduces learning rate every 50 epochs
    #         # if not epoch % 50:
    #         #     for param_group in optimizer.param_groups:
    #         #         param_group["lr"] = lr * (0.993 ** epoch)

    #         losses = []
    #         for x in train_set:
    #             x = x.to(device)
    #             optimizer.zero_grad()

    #             # Forward pass
    #             x_prime = model(x)

    #             loss = criterion(x_prime, x)

    #             # Backward pass
    #             loss.backward()

    #             # Gradient clipping on norm
    #             if clip_value is not None:
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    #             optimizer.step()

    #             losses.append(loss.item())

    #         mean_loss = mean(losses)
    #         mean_losses.append(mean_loss)

    #         if verbose:
    #             print(f"Epoch: {epoch}, Loss: {mean_loss}")

    # return mean_losses
    
    # def predict(self, x):
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     if type(x) != torch.Tensor:
    #         x = torch.tensor(x, dtype=torch.float32)
    #     y = self.forward(x.to(device))
    #     errors = []
    #     for i,v in enumerate(y):
    #         errors.append(metrics.mean_squared_error(v.cpu().detach().numpy(),x[i].cpu()))
    #     return np.array(errors)