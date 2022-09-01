import sys, pdb
sys.path.append("modules")
sys.path.append('CausalMBRL')
sys.path.append('CausalMBRL/envs')

import gym
from typing import OrderedDict
from tqdm import tqdm
from matplotlib import pyplot as plt
from exps.chem_datagen import generate_colors, generate_samples

import numpy as onp
from jax import numpy as jnp
from modules.ColorGen import LinearGaussianColor

def generate_data(n, obs_data, d, exp_edges, noise_sigma, data_seed,
                n_interv_sets,
                sem_type='linear-gauss', low=-8., high=8., 
                interv_low=-5., interv_high=5.):

    chem_data = LinearGaussianColor(
                    n=n,
                    obs_data=obs_data,
                    d=d,
                    graph_type="erdos-renyi",
                    degree=2 * exp_edges,
                    sem_type=sem_type,
                    dataset_type="linear",
                    noise_scale=noise_sigma,
                    data_seed=data_seed,
                    low=low, high=high
                )
    gt_W = chem_data.W
    gt_P = chem_data.P
    gt_L = chem_data.P.T @ chem_data.W.T @ chem_data.P
    print(gt_W)

    # ? generate linear gaussian colors
    z, interv_targets, interv_values = generate_colors(n, d, obs_data, n_interv_sets, chem_data, low, high, interv_low, interv_high, noise_sigma)
    normalized_z = 255. * ((z / (2 * high)) + 0.5)

    # ? Use above colors to generate images
    images = generate_chem_image_dataset(n, d, interv_values, interv_targets, z)
    onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/interv_values-seed{data_seed}_d{d}_ee{int(exp_edges)}.npy', onp.array(interv_values))
    onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/interv_targets-seed{data_seed}_d{d}_ee{int(exp_edges)}.npy', onp.array(interv_targets))
    onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/z-seed{data_seed}_d{d}_ee{int(exp_edges)}.npy', onp.array(z))
    onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/images-seed{data_seed}_d{d}_ee{int(exp_edges)}.npy', onp.array(images))
    onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/W-seed{data_seed}_d{d}_ee{int(exp_edges)}.npy', onp.array(gt_W))
    onp.save(f'/home/mila/j/jithendaraa.subramanian/scratch/P-seed{data_seed}_d{d}_ee{int(exp_edges)}.npy', onp.array(gt_P))
    print(gt_W)
    print()

    max_cols = jnp.max(interv_targets.sum(1))
    data_idx_array = jnp.array([jnp.arange(d + 1)] * n)
    interv_nodes = onp.split(data_idx_array[interv_targets], interv_targets.sum(1).cumsum()[:-1])
    interv_nodes = jnp.array([jnp.concatenate((interv_nodes[i], jnp.array([d] * (max_cols - len(interv_nodes[i])))))
        for i in range(n)]).astype(int)

    return z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L

def generate_chem_image_dataset(n, d, interv_values, interv_targets, z):
    images = None
    env = gym.make(f'LinGaussColorCubesRL-{d}-{d}-Static-10-v0')

    for i in tqdm(range(n)):
        action = OrderedDict()
        action['nodes'] = onp.where(interv_targets[i])
        action['values'] = interv_values[i]
        ob, _, _, _ = env.step(action, z[i])
        
        if i == 0:
            images = ob[1][jnp.newaxis, :]
        else:
            images = onp.concatenate((images, ob[1][jnp.newaxis, :]), axis=0)

    return images