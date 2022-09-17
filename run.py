def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

import torch
import matplotlib.pyplot as plt

from train import train_ae
from plot import plot_heatmap, plot_3d_regression, pca_and_tsne, plot_reconstruct
from utils import load_data, get_latent_vector, train_test_split
from model import AutoEncoder

def main(data_path, model_class, latent_dim, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(data_path=data_path)
    train_x, test_x, train_idx, test_idx = train_test_split(data)
    optim_params = kwargs
    epochs = kwargs.pop("epochs")
    plot_heatmap(data)

    model = model_class(in_dim=data.shape[1], out_dim=latent_dim, device=device)
    plot_reconstruct(model, data, device, title="Before training", noise_std=0.)
    train_ae(model, train_x, test_x, device, epochs=epochs, **optim_params)

    plot_reconstruct(model, train_x, device, title="After training: Train", noise_std=0.)
    plot_reconstruct(model, train_x, device, title="After training: Test", noise_std=0.)
    plot_reconstruct(model, data, device, title="After training: Total", noise_std=0.)
    plot_reconstruct(model, data, device, title="After training: Total + Noise", noise_std=0.05)
    
    latent = get_latent_vector(model, data, device)
    if latent.shape[1] == 3:
        plot_3d_regression(None, latent, train_idx, test_idx, title="Latent vectors from Autoencoder")
    else:
        print(f"Dimension of latent vector: {latent.shape}, cannot visualize directly.")
    
    pca_and_tsne(data, train_idx, test_idx, n_components=2, title="Raw data")
    pca_and_tsne(latent, train_idx, test_idx, n_components=2, title="Latent")
    pca_and_tsne(data, train_idx, test_idx, n_components=3, title="Raw data")
    pca_and_tsne(latent, train_idx, test_idx, n_components=3, title="Latent")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-path', '-P', type=str,
        help="Data path"
    )
    parser.add_argument(
        '--latent-dim', '-D', type=int, default=12,
        help="Dimension of latent vector"
    )
    args = parser.parse_args()
    
    params = {
        'epochs': 300,
        'lr': 3e-4,
    #     'weight_decay': 1e-5
    }

    model_class = AutoEncoder
    main(data_path=args.data_path, model_class=model_class, latent_dim=args.latent_dim, **params)
