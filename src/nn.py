import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # internally, input_size will be multiplied by 4 to fit the 4 different directions
        self.fc1 = nn.Linear(input_size * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.step = 1

    def forward(self, x):
        # take the x and rotate it 4 times
        # then concatenate them together
        # then feed it to the network
        x = torch.cat([x, x.rot90(1, [1, 2]), x.rot90(2, [1, 2]), x.rot90(3, [1, 2])], dim=1)
        # flatten the input
        x = x.view(-1, x.shape[1] * x.shape[2]).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def train(self, x, y, optimizer, criterion, epochs=100, verbose=False):
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            self.step += 1
            # print once every 100 epochs
            if verbose and epoch % 100 == 0:
                print("[INFO] Epoch {} Loss: {}".format(epoch, loss.item()))