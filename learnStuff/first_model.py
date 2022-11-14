import torch.nn as nn
import torch
import random
from learnStuff.linear_regression import LinearRegression
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
PyTorch pipeline:
1. Design model (input, output, forward pass, diffrent layers)
2. Construct loss and optimizer
3. Training loop:
    Forward = compute prediction & loss
    Backward = compute gradients
    Update weights
"""
# Linear regression
# f = w * x + b
# here : f = 2 * x + 1

# Input output, model

x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [[3], [5], [7], [9], [11], [13], [15], [17], [19], [21]]
X = torch.tensor(x, dtype=torch.float32, device=device)
Y = torch.tensor(y, dtype=torch.float32, device=device)

n_samples, n_features = X.shape
print(f'n_samples: {n_samples}, n_features: {n_features}')


input_size, output_size = n_features, n_features


model = LinearRegression(input_size, output_size, device=device)

# Define loss and optimizer
learing_rate = 0.01
epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)

New_test = torch.tensor([15], device=device, dtype=torch.float32)
X_test = torch.tensor([random.randint(1, 1000)], device=device, dtype=torch.float32)


def linear_regression_model(X_test=X_test, learing_rate=learing_rate, epochs=epochs, printEpochs=False):
    print(
        f'Prediction before training: f({X_test}) = {model(X_test).item():.3f}')
    for epoch in range(epochs):
        # predict
        y_pred = model(X)
        # loss
        l = loss(Y, y_pred)

        # calculate gradients
        l.backward()

        # update weights
        optimizer.step()

        # zeroing gradients
        optimizer.zero_grad()
        if (printEpochs):
            if (epoch+1) % 10 == 0:
                w, b = model.parameters()
                print(
                    f'Epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.3f}')
    print(
        f'Prediction after training: f({X_test}) = {model(X_test).item():.3f}')


linear_regression_model(X_test=X_test)
linear_regression_model(X_test=New_test)
