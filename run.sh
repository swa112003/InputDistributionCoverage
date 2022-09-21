#!/bin/bash

help()
{
    echo "Usage: run.sh [dataset: mnist/fmnist/cifar10] [vae: btcvae/factor] [latent_dim] [intervals] [ways] [target_density: range[0.1]]"
	echo "E.g. run.sh mnist btcvae 8 20 3 0.9999"
    exit 2
}

if [ $# -eq 0 ]; then
  help
fi

dataset=$1
vae=$2
latent_dim=$3
intervals=$4
ways=$5
target_density=$6

python measure_coverage.py $vae"_"$dataset"_"$latent_dim --dataset $dataset --no_bins $intervals --ways $ways --density $target_density 
