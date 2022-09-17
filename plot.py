import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import get_lr_prediction

palette = sns.color_palette("Set2", 10)

def plot_heatmap(data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()
    sns.heatmap(data, ax=ax, cmap="YlGnBu")
    ax.set_xlabel("time")
    ax.set_ylabel("# of neuron")
    ax.set_title("Raw data")

def plot_2d_regression(ax, data, train_idx, test_idx=None, title="None"):
    """ Plot 2D data with linear regression model
    
    Params:
        data: np.array([# of whole neurons, 2])
        train_idx: List, indexes of train data
    """

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplots()

    train_x = data[train_idx, :]
    _, train_lr_pred = get_lr_prediction(data, train_x)
    ax.set_xlabel('C1', fontsize=15)
    ax.set_ylabel('C2', fontsize=15)
    ax.set_title(title, fontsize=20)
    ax.scatter(
        train_x[:, 0], train_x[:, 1],
        label='train' if test_idx is not None else None, color=palette[0]
    )
    ax.plot(data[:, 0], train_lr_pred, color='g', lw=2, label="Train LR")
    
    if test_idx is not None:
        test_x = data[test_idx, :]
        ax.scatter(test_x[:, 0], test_x[:, 1], label='test', color=palette[1])

    ax.legend()
    ax.grid()

def plot_3d_regression(ax, data, train_idx, test_idx=None, alpha=0.3, title="None"):
    """ Plot 3D data with linear multidimensional regression model
    
    Params:
        data: np.array([# of whole neurons, 3])
        train_idx: List, indexes of train data
    """

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    train_x = data[train_idx, :]
    lr, _ = get_lr_prediction(data, train_x)

    coefs = lr.coef_
    intercept = lr.intercept_

    minx, miny = int(np.rint(np.min(train_x[:, 0]))), int(np.rint(np.min(train_x[:, 1])))
    maxx, maxy = int(np.rint(np.max(train_x[:, 0]))), int(np.rint(np.max(train_x[:, 1])))
    xs = np.tile(np.arange(minx, maxx, 1), (maxy - miny, 1))
    ys = np.tile(np.arange(miny, maxy), (maxx - minx, 1)).T
    zs = xs * coefs[0] + ys * coefs[1] + intercept
    print(f"{title} Equation: y = {intercept:.2f} + {coefs[0]:.2f}x1 + {coefs[1]:.2f}x2")
    
    ax.set_xlabel('C1', fontsize=15)
    ax.set_ylabel('C2', fontsize=15)
    ax.set_zlabel('C3', fontsize=15)

    ax.set_title(title, fontsize=20)
    ax.scatter(
        train_x[:, 0], train_x[:, 1], train_x[:, 2],
        label='train' if test_idx is not None else None, color=palette[0]
    )
    if test_idx is not None:
        test_x = data[test_idx, :]
        ax.scatter(test_x[:, 0], test_x[:, 1], test_x[:, 2], label='test', color=palette[1])

    surf = ax.plot_surface(xs, ys, zs, alpha=alpha, color=palette[7], label="Train LR")
    ### These lines are for error fixing
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ###
    
    ax.legend()
    ax.grid()

def pca_and_tsne(data, train_idx, test_idx, n_components=2, title=""):
    """ Plot the results of both PCA and t-SNE with given number of components 

    Params:
        data: np.array([# of whole neurons, dim])
        train_idx: List, indexes of train data
        test_idx: List, indexes of test data
        n_components: int, number of components for PCA and t-SNE
    """

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(data)
    df_pca = pd.DataFrame(data=projected, columns=[f"C{i + 1}" for i in range(n_components)])
    np_pca = df_pca.to_numpy()
    
    tsne = TSNE(n_components=n_components)
    projected = tsne.fit_transform(data)
    df_tsne = pd.DataFrame(data=projected, columns=[f"C{i + 1}" for i in range(n_components)])
    np_tsne = df_tsne.to_numpy()
    
    if n_components == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        plot_2d_regression(ax1, np_pca, train_idx, test_idx, title=f"{title} / 2D / PCA")
        plot_2d_regression(ax2, np_tsne, train_idx, test_idx, title=f"{title} / 2D / t-SNE")
    elif n_components == 3:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        plot_3d_regression(ax1, np_pca, train_idx, test_idx, title=f"{title} / 3D / PCA")
        plot_3d_regression(ax2, np_tsne, train_idx, test_idx, title=f"{title} / 3D / t-SNE")
    
    print(f"{title} / {n_components}D PCA-explained variance ratio: {pca.explained_variance_ratio_}")
    return df_pca, df_tsne, pca.explained_variance_ratio_

def plot_reconstruct(ae, data, device, title, noise_std=None):
    """ Reconstruct the data with a given autoencoder model

    Params:
        ae: torch.nn, autoencoder mdoel
        data: np.array([# of whole data, dim])
        noise_std: std. of noise to give to reconstructed data
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ae = ae.to(device)
    ae.eval()
    with torch.no_grad():
        pred = ae(torch.tensor(data).float().to(device)).cpu().numpy()

    # Original
    fig.suptitle(title, fontsize=24)
    fig.subplots_adjust(top=0.8)        
    sns.heatmap(data, cmap="YlGnBu", ax=ax1, vmin=0, vmax=1)
    ax1.set_title("Original data", fontsize=16)
    ax1.set_xlabel("time")
    ax1.set_ylabel("# of neuron")
    
    # Reconstructed
    if noise_std is not None:
        pred += np.random.normal(0, noise_std, size=pred.shape)
    
    sns.heatmap(pred, cmap="YlGnBu", ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Reconstructed", fontsize=16)
    ax2.set_xlabel("time")
    ax2.set_ylabel("# of neuron")

def plot_loss(timelines, losses, eval_losses):
    fig, ax = plt.subplots()
    ax.plot(timelines, losses, label="Train loss", c='b')
    ax.plot(timelines, eval_losses, label="Eval loss", c='r')
    ax.set_title("Loss value")
    ax.set_xlabel("# of Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.legend()
