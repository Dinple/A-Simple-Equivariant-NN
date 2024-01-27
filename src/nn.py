import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # internally, input_size will be multiplied by 4 to fit the 4 different directions
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.step = 1

    def forward(self, x):
        # flatten the input
        x = x.reshape(-1, 28 * 28).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def train(self, x, y, optimizer, criterion, epochs=100, verbose=False):
        for epoch in range(epochs):
            optimizer.zero_grad()
            for i in range(4):
                x = torch.rot90(x, 1, [1, 2]) # rotate the input 4 times
                output = self.forward(x)
                loss = criterion(output, y)
                loss.backward()
            optimizer.step()
            self.step += 1
            # print once every 100 epochs
            if verbose and epoch % 100 == 0:
                print("[INFO] Epoch {} Loss: {}".format(epoch, loss.item()))