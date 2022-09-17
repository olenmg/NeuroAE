import torch
import torch.nn as nn
from tqdm import tqdm

from plot import plot_loss

def eval_ae(ae, test_x, device):
    """ Evaluate the autoencoder model with given test data 

    Params:
        ae: torch.nn, autoencoder mdoel
        test_x: np.array([# of test data, dim])
    """

    criterion = nn.MSELoss()

    ae = ae.to(device)
    ae.eval()

    with torch.no_grad():
        x = torch.tensor(test_x).float()
        y = x.clone()
        x, y = x.to(device), y.to(device)

        pred = ae(x)
        loss = criterion(pred, y).item()

    return loss

def train_ae(ae, train_x, test_x, device, epochs=10, **kwargs):
    """ Train the autoencoder model with a given train data 

    Params:
        ae: torch.nn, autoencoder mdoel
        train_x: np.array([# of train data, dim])
        test_x: np.array([# of test data, dim])
    """

    optimizer = torch.optim.Adam(ae.parameters(), **kwargs)
    criterion = nn.MSELoss()
    
    timelines = []
    losses = []
    eval_losses = []

    ae = ae.to(device)
    for i in tqdm(range(epochs)):
        ae.train()
        x = torch.tensor(train_x).float()
        y = x.clone()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = ae(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            timelines.append(i + 1)
            losses.append(loss.item())
            eval_loss = eval_ae(ae, test_x, device)
            eval_losses.append(eval_loss)
            if (i + 1) % 50 == 0:
                print(f"[Epoch {i + 1}/{epochs}] Train loss: {loss.item():.3f} | Eval loss: {eval_loss:.3f} ")

    plot_loss(timelines, losses, eval_losses)
