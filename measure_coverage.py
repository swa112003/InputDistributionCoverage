import argparse
import os
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import datasets, transforms
from disvae.utils.modelIO import load_model, load_metadata
from utils.helpers import get_device, set_seed, get_config_section, FormatterNoDuplicate
from helper import generate_array
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
from timeit import default_timer as timer
import imageio

#Global variable declarations
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
      
#Calculate mean and log variance vectors output by the encoder for the test inputs
def evaluate(model, testloader, logger):
    initialize = True
    for data, _ in tqdm(testloader, leave=False, disable=not default_config['no_progress_bar']):
        data = data.to(device)
        recon_batch, latent_dist, latent_sample = model(data)
        if initialize:
            mu = latent_dist[0]
            sd = torch.exp(0.5 * latent_dist[1])
            initialize = False
        else:
            mu = torch.cat((mu, latent_dist[0]), 0)
            sd_temp = torch.exp(0.5 * latent_dist[1])
            sd = torch.cat((sd, sd_temp), 0)
      
    mu = mu.to('cpu')
    mu_np = mu.detach().numpy()
    
    sd = sd.to('cpu')
    sd_np = sd.detach().numpy()
    
    logger.info("latent_dist mu_np shape for the mnist test dataset {}".format(mu_np.shape))
    return mu_np, sd_np

                     
default_config = get_config_section([CONFIG_FILE], "Custom")
description = 'Measure coverage over disentangled representations'
parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)
parser.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")  
parser.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')            
parser.add_argument('-b', '--no_bins', type=int, default=20,
                         help='no of bins.') 
parser.add_argument('-ways', '--ways', type=int, default=3,
                         help='ways')
parser.add_argument('-density', '--density', type=float, default=0.9999,
                         help='density')                         
parser.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')  
parser.add_argument('--batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')      
parser.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS) 
parser.add_argument('--dataset', type=str,
                            default="mnist", choices=["mnist", "cifar10", "fmnist"],
                            help='dataset')                         
                         
args = parser.parse_args()

formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(args.log_level.upper())
stream = logging.StreamHandler()
stream.setLevel(args.log_level.upper())
stream.setFormatter(formatter)
logger.addHandler(stream)
set_seed(args.seed)
device = get_device(is_gpu=not args.no_cuda)
exp_dir = os.path.join(RES_DIR, args.name)

   
model = load_model(exp_dir, is_gpu=not args.no_cuda)
metadata = load_metadata(exp_dir)
                         
logger.info("Testing Device: {}".format(device))
print(f"VAE {args.name}")
if args.dataset == "mnist":
    testset = dset.MNIST(root="./data", train=False, download=True)
    dataset_mnist = dset.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(dataset_mnist, batch_size=args.batchsize, shuffle=False)
    
elif args.dataset == "cifar10":
    testset = dset.CIFAR10(root="./data", train = False, download=True)
    dataset_cifar = dset.CIFAR10(root="./data", download=True, train = False, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(dataset_cifar, batch_size=args.batchsize, shuffle=False)
    
else:
    testset = dset.FashionMNIST(root="./data", train=False, download=True)
    dataset_fmnist = dset.FashionMNIST(root="./data", train=False, download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(dataset_fmnist, batch_size=args.batchsize, shuffle=False)

mu_test, sd_test = evaluate(model, test_loader, logger)

#calculate the KL-divergence for the testdata set
kl_div = 1 + np.log(np.square(sd_test)) - np.square(mu_test) - np.square(sd_test)
kl_div *= -0.5
kl_div = np.mean(kl_div, axis=0)

#delete the dimensions with close to zero KL-divergence values
noise = []
for l in range(mu_test.shape[1]):
    if abs(kl_div[l]) <= 0.01:
        noise.append(l)

#no of dimensions with information
info_dims = mu_test.shape[1] - len(noise)

mu_test = np.delete(mu_test, noise, 1)
print(f"deleting columns {noise} with KL {kl_div[noise]} from the latent mean vector")

#create acts file for measuring total t-way coverage
acts = create_acts(info_dims, args.no_bins)

#generate feasible feature vectors
feasible_vectors, valid_samples, _ = generate_array(mu_test, args.density, args.no_bins)
print(f"full test set valid samples {valid_samples} \n\n")

coverage = measure_coverage(feasible_vectors, acts, ways=args.ways, timeout=15)
print(f"total {args.ways}-way coverage wrt covering array is {coverage}")

