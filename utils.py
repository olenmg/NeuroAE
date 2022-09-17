import numpy as np
import torch
from sklearn.linear_model import LinearRegression

def load_data(data_path):
    """ Load npy data 

    Params:
        data_path: string
    """

    data = np.load(data_path)
    data = data[:, ~np.isnan(data).any(axis=0)]
    print(data.shape)
    
    return data

# Train & Test Split
def train_test_split(data, r=0.8, seed=0):
    """ Split data into random train and test subsets 
        for unsupervised learning (not require labels)

    Params:
        data: np.array([# of whole neurons, dim])
        r: float, ratio of train data in whole dataset
    """

    np.random.seed(seed)
    idx = np.arange(data.shape[0])
    train_idx = np.random.choice(idx, int(len(idx) * r), replace=False)
    test_idx = np.setdiff1d(idx, train_idx)

    train_x = data[train_idx, :]
    test_x = data[test_idx, :]
    
    return train_x, test_x, train_idx, test_idx

def get_latent_vector(ae, data, device):
    """ Get latent vectors from data using given autoencoder model

    Params:
        ae: torch.nn, autoencoder mdoel
        data: np.array([# of whole data, dim])
    """

    ae.eval()
    with torch.no_grad():
        pred = ae.get_latent(torch.tensor(data).float().to(device)).cpu().numpy()
    return pred

def get_lr_prediction(data, train_x):
    """ Fitting the linear regression model with 
        `train_x`, and predict the last column of `data`
    
    Params:
        data: np.array([# of whole neurons, dim])
        train_x: np.array([# of train data, dim])
    """

    x, y = train_x[:, :-1], train_x[:, -1]

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(x, y)
    pred = lr.predict(data[:, :-1])
    
    return lr, pred