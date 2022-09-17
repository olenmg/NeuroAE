# NeuroAE
Simple autoencoder that can generate latent vectors of time-scale firing rate data from multiple neurons.  

## Requirements
- Python 3.7+

## Quick start
```
pip3 install -r requirements.txt
python3 run.py --data-path ./data/sample.npy --latent-dim 12
```

**Note** If you want to utilize GPU, please refer the instruction of [PyTorch documentation](https://pytorch.org/get-started/locally/) and install the appropriate version of CUDA toolkit.

## Data
You should prepare your own data with `npy` format.  
If you want to use `mat` format, please refer the [link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html), convert and save your `mat` format data as Numpy array.  
The shape of the data must be `(# of neurons, rate of each timestep)`.

## Modeling
You can customize your own model at `model.py`.
