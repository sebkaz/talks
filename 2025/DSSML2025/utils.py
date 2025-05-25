# parametry
import torch
from torch.nn import Linear
import matplotlib.pyplot as plt
from IPython.display import clear_output

def parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"liczba trenowalnych parametrów: {num_params}")

def warotosci_parametrow(model):
    for layer in model.layers:
        if isinstance(layer, Linear):
            print(f"weight: {layer.state_dict()['weight']}")
            print(f"bias: {layer.state_dict()['bias']}")


def train(model, x, y, criterion, optimizer,  epochs = 1000):
    for epoch in range(epochs):
        #forward feed
        output_train = model(x)
        #calculate the loss
        loss = criterion(output_train, y)
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()
        #backward propagation: calculate gradients
        loss.backward()
        #update the weights
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

def mse(y_true, y_pred) -> torch.Tensor:
    return torch.mean((y_true - y_pred)**2)

def trainqnn(X, Y, model, optimiser, iteration, lossfn, callback = None):
    """ Dodatkowa funkcja pozwalająca wykonać trenowanie naszej sieci neuronowej"""
    for i in range(iteration):
        optimiser.zero_grad()
        prediction = model(X)
        loss = lossfn(Y, prediction)
        loss.backward()
        optimiser.step()
        if callback is not None: 
            callback(model, loss)

losses = []

def callback(model, loss):
    losses.append(loss.item())
    x = torch.linspace(0,10,500).view(-1,1)
    clear_output(wait=True)
    prediction = model(x).detach()
    plt.figure(figsize=(6,2.5))
    plt.plot(x[:,0].detach(), torch.sin(x)[:,0].detach(), label="Exact solution", color="tab:grey", alpha=0.6)
    plt.plot(x[:,0].detach(), prediction[:,0], label="QML solution", color="tab:green")
    plt.title(f"Training step {len(losses)}")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,2.5))
    plt.title('Lossfn Visualised')
    plt.plot(losses)
    plt.show()