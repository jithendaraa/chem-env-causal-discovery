import sys, pdb
from os.path import join

sys.path.append('CausalMBRL')
sys.path.append('CausalMBRL/envs')

from exps.chem_datagen import *
from utils import *
from tqdm import tqdm
import envs

n = 200
obs_data = 100
d = 5
degree = 1.0
l_dim = d * (d - 1) // 2
noise_sigma = 1.0
exp_edges = 1.0
data_seed = 0
n_interv_sets = 10

low = -8.
high = 8.
interv_low = -5.
interv_high = 5.
proj_dims = (1, 50, 50)

z, interv_nodes, interv_values, images, gt_W, gt_P, gt_L = generate_data(n, obs_data, d, exp_edges, noise_sigma, data_seed, n_interv_sets, 
                                                                        low=low, high=high, interv_low=interv_low, interv_high=interv_high)