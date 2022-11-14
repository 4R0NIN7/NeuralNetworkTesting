import torch.nn as nn
import torch
from learnStuff.models.linear_regression import LinearRegression
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Linear regression
# f = w * x + b
# here : f = 2 * x + 1
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
X = torch.tensor(x, dtype=torch.float16, device=device)
Y = torch.tensor(y, dtype=torch.float32, device=device)
w = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)

# model output


def forward(x):
    return w * x + 1

# loss = MSE


def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# Testing function


def linear_regression_custom_model(X_test, learing_rate, epochs, w=w):
    print(
        f'Prediction before training: f({X_test}) = {forward(X_test).item():.3f}')
    for epoch in range(epochs):
        # predict
        y_pred = forward(X)
        # loss
        l = loss(Y, y_pred)

        # calculate gradients
        l.backward()

        # update weights
        with torch.no_grad():
            w -= learing_rate * w.grad

        # zeroing gradients
        w.grad.zero_()

        if (epoch+1) % 10 == 0:
            print(
                f'Epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.3f}')

        print(
            f'Prediction after training: f({X_test}) = {forward(X_test).item():.3f}')


linear_regression_custom_model(X_test=-123, learing_rate=0.01, epochs=10)
