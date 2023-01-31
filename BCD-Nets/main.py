import numpy as onp
import jax.numpy as jnp
from typing import Tuple, Optional, cast
import itertools
import warnings
import sys, string, pdb, os
import pickle as pkl
from tqdm import tqdm
import pandas as pd


# python main.py -s 0 --dim 20 --num_steps 5000 --do_ev_noise --sem_type linear-gauss --batch_size 1000 --use_wandb
warnings.simplefilter(action="ignore", category=FutureWarning)

from doubly_stochastic import GumbelSinkhorn
import jax.random as rnd
from jax import vmap, grad, jit, lax, pmap, partial, value_and_grad

from utils import (
    lower,
    eval_W_non_ev,
    eval_W_ev,
    ff2,
    num_params,
    save_params,
    get_variance,
    get_variances,
    from_W,
    rk,
    get_double_tree_variance,
    auroc,
)
from jax.tree_util import tree_map, tree_multimap

import jax
from tensorflow_probability.substrates.jax.distributions import (
    Normal,
    Horseshoe,
)

import matplotlib.pyplot as plt
import matplotlib as mpl
from jax import config
import haiku as hk
from models import (
    get_model,
    get_model_arrays,
)
import time
from jax.flatten_util import ravel_pytree
import optax
from PIL import Image
# from flows import get_flow_CIF

# from golem_utils import solve_golem_cv, bootstrapped_golem_cv
from argparse import ArgumentParser
# from baselines import run_all_baselines, eval_W_samples
from metrics import intervention_distance, ensemble_intervention_distance
from _types import PParamType, LStateType

print("finished imports")

config.update("jax_enable_x64", True)
import jax


mpl.rcParams["figure.dpi"] = 300


PRNGKey = jnp.ndarray
QParams = Tuple[jnp.ndarray, hk.Params]

# For running sweeps
parser = ArgumentParser()

parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("--eval_eid", action="store_true")
parser.add_argument("--run_baselines", action="store_true")
parser.add_argument("--dim", type=int, default=8)
parser.add_argument("--use_sachs", action="store_true")
parser.add_argument("--do_ev_noise", action="store_true")
parser.add_argument("--factorized", action="store_true")
parser.add_argument("--do_bootstrap_golem", action="store_true")
parser.add_argument("--print_golem_solution", action="store_true")
parser.add_argument("--use_flow", action="store_true")
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--subsample", action="store_true")
parser.add_argument("--n_data", type=int, default=100)
parser.add_argument("--only_baselines", action="store_true")
parser.add_argument("--num_steps", type=int, default=20_000)
parser.add_argument("--golem_steps", type=int, default=200_000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--logit_constraint", type=float, default=10)
parser.add_argument("--fixed_tau", type=float, default=0.2)
parser.add_argument("--max_deviation", type=float, default=0.01)
parser.add_argument("--n_baseline_seeds", type=int, default=5)
parser.add_argument("--fast_baselines", action="store_true")
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--use_alternative_horseshoe_tau", action="store_true")
parser.add_argument("--n_sumu_iters", type=int, default=50_000)

parser.add_argument("--use_my_data", type=bool, default=True)
parser.add_argument("--data_path", type=str, default='er2-lingauss-d020:v1')
parser.add_argument("--root_path", type=str, default='/home/mila/j/jithendaraa.subramanian/scratch/artifacts/')
parser.add_argument("--learn_noise", type=bool, default=False)

parser.add_argument(
    "--sem_type",
    type=str,
    choices=["linear-gauss", "linear-gumbel"],
    default="linear-gauss",
)

args = parser.parse_args()
dim = args.dim
random_seed = args.seed
run_baselines = args.run_baselines
eval_eid = args.eval_eid
use_sachs = args.use_sachs
do_ev_noise = args.do_ev_noise
factorized = args.factorized
batch_size = args.batch_size
print_golem_solution = args.print_golem_solution
degree = args.degree
use_flow = args.use_flow
subsample = args.subsample
n_data = args.n_data
sem_type = args.sem_type
only_baselines = args.only_baselines
num_steps = args.num_steps
golem_steps = args.golem_steps
lr = args.lr
logit_constraint = args.logit_constraint
max_deviation = args.max_deviation
do_bootstrap_golem = args.do_bootstrap_golem
n_baseline_seeds = args.n_baseline_seeds
fast_baselines = args.fast_baselines
use_wandb = args.use_wandb
use_alternative_horseshoe_tau = args.use_alternative_horseshoe_tau
n_sumu_iters = args.n_sumu_iters

data_path = args.root_path + args.data_path 
if args.seed >= 10:
    seed_folder = str(args.seed)
else:
    seed_folder = '0' + str(args.seed)

train_data_path = data_path + '/' + seed_folder + '/train_data.csv'
gt_graph_path = data_path + '/' + seed_folder + '/adjacency.npy'
print(train_data_path, gt_graph_path)

let_num = {letter: index for index, letter in enumerate(string.ascii_uppercase, start=0)} 

def graph_to_matrix(graph, num_nodes):
    matrix = onp.zeros((num_nodes, num_nodes))
    for child in graph.adjacency():
        # adj shows child of node
        for n, parent in enumerate(graph.get_parents(child[0])):        
            theta = graph.get_cpds(child[0]).mean[n+1]
            matrix[let_num[parent]][let_num[child[0]]] = theta
    return matrix

if use_sachs:
    dim = 11

override_to_cpu = False
if override_to_cpu:
    jax.config.update("jax_platform_name", "cpu")

onp.random.seed(random_seed)
from dag_utils import (
    SyntheticDataset,
    process_sachs,
    get_sachs_ground_truth,
)


num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")
if "gpu" not in str(jax.devices()).lower():
    print("NO GPU FOUND")
    # exit


lr_P = lr
lr_L = lr
num_flow_layers = 2
num_perm_layers = 2
hidden_size = 128
fixed_tau = args.fixed_tau

num_mixture_components = 4
num_outer = 1
fix_L_params = False
log_stds_max: Optional[float] = 10.0
L_dist = Normal
log_sigma_l = 0


init_std = 0.00
use_grad_global_norm_clipping = False
P_norm = 100
L_norm = 100

flow_threshold = -1e3
L_init_scale = 0.0
s_init_scale = 0.0
init_flow_std = 0.1
s_prior_std = 3.0
calc_shd_c = False
pretrain_flow = False
if factorized:
    method = "factorized"
else:
    method = "both"
rng_key = rk(random_seed)

if do_ev_noise:
    log_sigma_W = jnp.zeros(dim)
else:
    log_sigma_W = onp.random.uniform(low=0, high=jnp.log(2), size=(args.dim,))


if not use_flow:
    flow_type = None

if use_sachs:
    dim = 11
    # Note that for our training, we centered based on n=853 but then gave 100
    Xs = process_sachs(center=True, normalize=True, n_data=n_data, rng_key=rng_key)
    test_Xs = process_sachs(center=True, normalize=True, rng_key=rng_key)
    ground_truth_W = get_sachs_ground_truth()
    n_data = len(Xs)
    ground_truth_sigmas = jnp.ones(dim)
    print(jnp.sum(ground_truth_W != 0))

else:
    if args.use_my_data:
        Xs_csv = pd.read_csv(train_data_path).to_numpy()[:, 1:]
        Xs_csv = Xs_csv.astype(onp.float)
        Xs = cast(jnp.ndarray, Xs_csv)
        n_data, dim = Xs.shape
        args.n_data = n_data
        args.dim = dim
        ground_truth_W = onp.load(gt_graph_path)
        log_noise_sigma = 0.5 * onp.log(0.1)
    
    else:
        print("SD", jnp.exp(log_sigma_W) )
        sd = SyntheticDataset(
            n=n_data,
            d=args.dim,
            graph_type="erdos-renyi",
            degree=2 * degree,
            sem_type=sem_type,
            dataset_type="linear",
        )
        ground_truth_W = sd.W

        Xs = sd.simulate_sem(
            ground_truth_W,
            n_data,
            sd.sem_type,
            w_range=None,
            noise_scale=None,
            dataset_type="linear",
            W_2=None,
            sigmas=jnp.exp(log_sigma_W),
        )

        test_Xs = sd.simulate_sem(
            ground_truth_W,
            sd.n,
            sd.sem_type,
            sd.w_range,
            sd.noise_scale,
            sd.dataset_type,
            sd.W_2,
        )

Xs = jnp.array(Xs)

if do_ev_noise:
    noise_dim = 1
    # Generate noises same as GOLEM/Notears github
    log_sigma_W = jnp.zeros(dim)
else:
    noise_dim = dim
    log_sigma_W = onp.random.uniform(low=0, high=jnp.log(2), size=(args.dim,))


# This may be preferred from 'The horseshoe estimator: Posterior concentration around nearly black vectors'
# van der Pas et al
if args.use_alternative_horseshoe_tau:
    p_n_over_n = 2 * degree / (dim - 1)
    if p_n_over_n > 1:
        p_n_over_n = 1
    horseshoe_tau = p_n_over_n * jnp.sqrt(jnp.log(1.0 / p_n_over_n))
else:
    horseshoe_tau = (1 / onp.sqrt(n_data)) * (2 * degree / ((dim - 1) - 2 * degree))
if horseshoe_tau < 0:  # can happen for very small graphs
    horseshoe_tau = 1 / (2 * dim)
print(f"Horseshoe tau is {horseshoe_tau}")

ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=max_deviation)

l_dim = dim * (dim - 1) // 2

ground_truth_sigmas = jnp.exp(log_sigma_W)
print("\n")


wandb = None
if use_wandb:
    import wandb

    wandb.init(project="Learning DAGs")
    configuration = {
        "dim": dim,
        "lr_P": lr_P,
        "lr_L": lr_L,
        "n_data": n_data,
        "max_deviation": max_deviation,
        "num_devices": num_devices,
        "batch_size": batch_size,
        "num_inner": 1,
        "model_type": "MLP",
        "hidden_size": hidden_size,
        "num_flow_layers": num_flow_layers,
        "num_mixture_components": num_mixture_components,
        "num_perm_layers": num_perm_layers,
        "use_grad_global_norm_clipping": use_grad_global_norm_clipping,
        "P_norm": P_norm,
        "L_norm": L_norm,
        "flow_type": "CIF",
        "fixed_tau": fixed_tau,
        "do_ev_noise": do_ev_noise,
        "L_init_scale": L_init_scale,
        "use_flow": use_flow,
        "method": method,
        "init_flow_std": init_flow_std,
        "s_prior_std": s_prior_std,
        "print_golem_solution": print_golem_solution,
        "horseshoe_tau": horseshoe_tau,
        "l_init_offset": 0.0,
        "degree": degree,
        "sem_type": sem_type,
        "random_seed": random_seed,
        "logit_constraint": logit_constraint,
        "use_sachs": use_sachs,
        "eval_eid": eval_eid,
        "bootstrap_golem": do_bootstrap_golem,
        "run_baselines": run_baselines,
        "n_baseline_seeds": n_baseline_seeds,
        "use_alternative_horseshoe_tau": use_alternative_horseshoe_tau,
        "n_sumu_iters": n_sumu_iters,
        "data_path": args.data_path,
        'model': 'BCDNets',
        'learn_noise': args.learn_noise
    }
    wandb.config.update(configuration)
    print(configuration)
    wandb_name = wandb.run.name
    wandb_str = wandb_name.split("-")[0]  # type: ignore
    wandb_string = (
        f"{sem_type.split('-')[1]}_d_{degree}_s_{random_seed}_{wandb_str[:4]}"
    )
    wandb.run.name = wandb_string


plt.imshow(ground_truth_W)
plt.savefig(f"./tmp{dim}_{args.seed}.png")

if wandb is not None:
    wandb.log(
        {
            "Ground Truth": [wandb.Image(Image.open(f"./tmp{dim}_{args.seed}.png"), caption="W sample")],
            # "Data": Xs_csv 
        },
        step=0,
    )
    plt.close()

L_layers = []
P_layers = []
if use_grad_global_norm_clipping:
    L_layers += [optax.clip_by_global_norm(L_norm)]
    P_layers += [optax.clip_by_global_norm(P_norm)]
P_layers += [optax.scale_by_belief(eps=1e-8), optax.scale(-lr_P)]
L_layers += [optax.scale_by_belief(eps=1e-8), optax.scale(-lr_L)]
opt_P = optax.chain(*P_layers)
opt_L = optax.chain(*L_layers)
opt_joint = None
time0 = time.time()


def init_parallel_params(rng_key: PRNGKey):
    
    def init_params(rng_key: PRNGKey):
        if args.learn_noise:
            L_params = jnp.concatenate(
                (
                    jnp.zeros(l_dim + noise_dim),
                    jnp.zeros(l_dim + noise_dim) - 1,
                )
            )
        else:
            L_params = jnp.concatenate(
                (
                    jnp.zeros(l_dim),
                    jnp.zeros(l_dim) - 1,
                )
            )

        # Would be nice to put none here, but need to pmap well
        L_states = jnp.array([0.0])
        P_params = get_model_arrays(
            dim,
            batch_size,
            num_perm_layers,
            rng_key,
            hidden_size=hidden_size,
            do_ev_noise=do_ev_noise,
        )
        if factorized:
            P_params = jnp.zeros((dim, dim))
        P_opt_params = opt_P.init(P_params)
        L_opt_params = opt_L.init(L_params)
        return (
            P_params,
            L_params,
            L_states,
            P_opt_params,
            L_opt_params,
        )

    output = init_params(rng_key)
    return output

_, p_model = get_model(
    dim, batch_size, num_perm_layers, hidden_size=hidden_size, do_ev_noise=do_ev_noise,
)

P_params, L_params, L_states, P_opt_params, L_opt_params = init_parallel_params(rng_key)

print(f"L model has {ff2(num_params(L_params))} parameters")
print(f"P model has {ff2(num_params(P_params))} parameters")


@jit
def get_P_logits(
    P_params: PParamType, L_samples: jnp.ndarray, rng_key: PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    if factorized:
        # We ignore L when giving the P parameters
        assert type(P_params) is jnp.ndarray
        p_logits = jnp.tile(P_params.reshape((1, dim, dim)), (len(L_samples), 1, 1))
    else:
        P_params = cast(hk.Params, P_params)
        p_logits = p_model(P_params, rng_key, L_samples)  # type:ignore

    if logit_constraint is not None:
        # Want to map -inf to -logit_constraint, inf to +logit_constraint
        p_logits = jnp.tanh(p_logits / logit_constraint) * logit_constraint

    return p_logits.reshape((-1, dim, dim))


@jit
def sample_L(
    L_params: PParamType, L_state: LStateType, rng_key: PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, LStateType]:

    L_params = cast(jnp.ndarray, L_params)
    if args.learn_noise: 
        means, log_stds = L_params[: l_dim + noise_dim], L_params[l_dim + noise_dim :]
    else:
        means, log_stds = L_params[:l_dim], L_params[l_dim:]
    if log_stds_max is not None:
        # Do a soft-clip here to stop instability
        log_stds = jnp.tanh(log_stds / log_stds_max) * log_stds_max
    l_distribution = L_dist(loc=means, scale=jnp.exp(log_stds))
    
    full_l_batch = l_distribution.sample(
        seed=rng_key, sample_shape=(batch_size,)
    )
    full_l_batch = cast(jnp.ndarray, full_l_batch)
    
    full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)
    full_log_prob_l = cast(jnp.ndarray, full_log_prob_l)

    out_L_states = None

    if args.learn_noise is False:
        batched_lognoise_sigma = jnp.ones((args.batch_size, noise_dim)) * log_noise_sigma
        full_l_batch = jnp.concatenate((full_l_batch, batched_lognoise_sigma), axis=1)
        
    return full_l_batch, full_log_prob_l, out_L_states

@jit
def log_prob_x(Xs, log_sigmas, P, L, rng_key):
    """Calculates log P(X|Z) for latent Zs
    X|Z is Gaussian so easy to calculate
    Args:
        Xs: an (n x dim)-dimensional array of observations
        log_sigmas: A (dim)-dimension vector of log standard deviations
        P: A (dim x dim)-dimensional permutation matrix
        L: A (dim x dim)-dimensional strictly lower triangular matrix
    Returns:
        log_prob: Log probability of observing Xs given P, L
    """
    n, dim = Xs.shape
    W = (P @ L @ P.T).T
    precision = (
        (jnp.eye(dim) - W) @ (jnp.diag(jnp.exp(-2 * log_sigmas))) @ (jnp.eye(dim) - W).T
    )
    eye_minus_W_logdet = 0
    log_det_precision = -2 * jnp.sum(log_sigmas) + 2 * eye_minus_W_logdet

    def datapoint_exponent(x):
        return -0.5 * x.T @ precision @ x

    log_exponent = vmap(datapoint_exponent)(Xs)

    return (0.5 * n * (log_det_precision - dim * jnp.log(2 * jnp.pi))
        + jnp.sum(log_exponent)
    )

@jit
def hard_elbo(
    P_params: PParamType,
    L_params: hk.Params,
    L_states: LStateType,
    Xs: jnp.ndarray,
    rng_key: PRNGKey,
    tau: float
) -> Tuple[jnp.ndarray, LStateType]:
    """Computes ELBO estimate from parameters.
    Computes ELBO(P_params, L_params), given by
    E_{e1}[E_{e2}[log p(x|L, P)] - D_KL(q_L(P), p(L)) -  log q_P(P)],
    where L = g_L(L_params, e2) and P = g_P(P_params, e1).
    The derivative of this corresponds to the pathwise gradient estimator
    Args:
        P_params: inputs to sampling path functions
        L_params: inputs parameterising function giving L|P distribution
        Xs: (n x dim)-dimension array of inputs
        rng_key: jax prngkey object
        log_sigma_W: (dim)-dimensional array of log standard deviations
        log_sigma_l: scalar prior log standard deviation on (Laplace) prior on l.
    Returns:
        ELBO: Estimate of the ELBO
    """
    num_bethe_iters = 20
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)

    def outer_loop(rng_key: PRNGKey):
        """Computes a term of the outer expectation, averaging over batch size"""
        rng_key, rng_key_1 = rnd.split(rng_key, 2)
        full_l_batch, full_log_prob_l, out_L_states = sample_L(
            L_params, L_states, rng_key
        )
        w_noise = full_l_batch[:, -noise_dim:]
        l_batch = full_l_batch[:, :-noise_dim]
        batched_noises = jnp.ones((batch_size, dim)) * w_noise.reshape(
            (batch_size, noise_dim)
        )
        batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, dim)
        batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key_1)
        
        batched_P_samples = ds.sample_hard_batched_logits(
            batched_P_logits, tau, rng_key,
        )
        
        likelihoods = vmap(log_prob_x, in_axes=(None, 0, 0, 0, None))(
            Xs, batched_noises, batched_P_samples, batched_lower_samples, rng_key,
        )
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :l_dim], axis=1)
        s_prior_probs = jnp.sum(
            full_l_batch[:, l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1
        )
        KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
        logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(
            batched_P_samples, batched_P_logits, num_bethe_iters
        )
        log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
        final_term = likelihoods - KL_term_L - logprob_P + log_P_prior

        return jnp.mean(final_term), out_L_states

    elbos, out_L_states = outer_loop(rng_key)
    elbo_estimate = jnp.mean(elbos)
    return elbo_estimate, tree_map(lambda x: x[-1], out_L_states)

@jit
def soft_elbo(
    P_params: PParamType,
    L_params: hk.Params,
    L_states: LStateType,
    Xs: jnp.ndarray,
    rng_key: PRNGKey,
    tau: float
) -> Tuple[jnp.ndarray, LStateType]:
    """Computes ELBO estimate from parameters.
    Computes ELBO(P_params, L_params), given by
    E_{e1}[E_{e2}[log p(x|L, P)] - D_KL(q_L(P), p(L)) -  log q_P(P)],
    where L = g_L(L_params, e2) and P = g_P(P_params, e1).
    The derivative of this corresponds to the pathwise gradient estimator
    Args:
        P_params: inputs to sampling path functions
        L_params: inputs parameterising function giving L|P distribution
        Xs: (n x dim)-dimension array of inputs
        rng_key: jax prngkey object
        log_sigma_W: (dim)-dimensional array of log standard deviations
        log_sigma_l: scalar prior log standard deviation on (Laplace) prior on l.
    Returns:
        ELBO: Estimate of the ELBO
    """
    num_bethe_iters = 20
    l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)

    def outer_loop(rng_key: PRNGKey):
        """Computes a term of the outer expectation, averaging over batch size"""
        rng_key, rng_key_1 = rnd.split(rng_key, 2)
        full_l_batch, full_log_prob_l, out_L_states = sample_L(
            L_params, L_states, rng_key
        )
        w_noise = full_l_batch[:, -noise_dim:]
        l_batch = full_l_batch[:, :-noise_dim]
        batched_noises = jnp.ones((batch_size, dim)) * w_noise.reshape(
            (batch_size, noise_dim)
        )
        batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, dim)
        batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key_1)
        batched_P_samples = ds.sample_soft_batched_logits(batched_P_logits, tau, rng_key)
        likelihoods = vmap(log_prob_x, in_axes=(None, 0, 0, 0, None))(
            Xs, batched_noises, batched_P_samples, batched_lower_samples, rng_key,
        )
        l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :l_dim], axis=1)
        s_prior_probs = jnp.sum(
            full_l_batch[:, l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1
        )
        KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
        logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(
            batched_P_samples, batched_P_logits, num_bethe_iters
        )
        log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
        final_term = likelihoods - KL_term_L - logprob_P + log_P_prior

        return jnp.mean(final_term), out_L_states

    elbos, out_L_states = outer_loop(rng_key)
    elbo_estimate = jnp.mean(elbos)
    return elbo_estimate, tree_map(lambda x: x[-1], out_L_states)

if do_ev_noise:
    eval_W_fn = eval_W_ev
else:
    eval_W_fn = eval_W_non_ev

def eval_mean(
    P_params, L_params, L_states, Xs, rng_key=rk(0), do_shd_c=calc_shd_c, tau=1,
):
    """Computes mean error statistics for P, L parameters and data"""
    _, dim = Xs.shape
    x_prec = onp.linalg.inv(jnp.cov(Xs.T))
    full_l_batch, _, _ = sample_L(L_params, L_states, rng_key)
    w_noise = full_l_batch[:, -noise_dim:]
    l_batch = full_l_batch[:, :-noise_dim]
    batched_lower_samples = jit(vmap(lower, in_axes=(0, None)), static_argnums=(1,))(
        l_batch, dim
    )
    batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key)
    batched_P_samples = jit(ds.sample_hard_batched_logits)(
        batched_P_logits, tau, rng_key
    )

    def sample_W(L, P):
        return (P @ L @ P.T).T

    Ws = jit(vmap(sample_W))(batched_lower_samples, batched_P_samples)

    def sample_stats(W, noise):
        stats = eval_W_fn(
            W,
            ground_truth_W,
            ground_truth_sigmas,
            0.3,
            Xs,
            jnp.ones(dim) * jnp.exp(noise),
            provided_x_prec=x_prec,
            do_shd_c=do_shd_c,
            do_sid=do_shd_c,
        )
        return stats

    stats = sample_stats(Ws[0], w_noise[0])
    stats = {key: [stats[key]] for key in stats}
    for i, W in enumerate(Ws[1:]):
        new_stats = sample_stats(W, w_noise[i])
        for key in new_stats:
            stats[key] = stats[key] + [new_stats[key]]

    # stats = vmap(sample_stats)(rng_keys)
    out_stats = {key: onp.mean(stats[key]) for key in stats}
    out_stats["auroc"] = auroc(Ws, ground_truth_W, 0.3)
    return out_stats


def get_num_sinkhorn_steps(P_params, L_params, L_states, rng_key):

    full_l_batch, _, _ = sample_L(L_params, L_states, rng_key)
    batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key)
    _, errors = jit(ds.sample_hard_batched_logits_debug)(
        batched_P_logits, tau, rng_key,
    )
    first_converged = jnp.where(jnp.sum(errors, axis=0) == -batch_size)[0]
    if len(first_converged) == 0:
        converged_idx = -1
    else:
        converged_idx = first_converged[0]
    return converged_idx


def eval_ID(P_params, L_params, L_states, Xs, rng_key, tau):
    """Computes mean error statistics for P, L parameters and data"""

    _, dim = Xs.shape
    full_l_batch, _, _ = jit(sample_L, static_argnums=3)(L_params, L_states, rng_key)
    w_noise = full_l_batch[:, -noise_dim:]
    l_batch = full_l_batch[:, :-noise_dim]
    batched_lower_samples = jit(vmap(lower, in_axes=(0, None)), static_argnums=(1,))(
        l_batch, dim
    )
    batched_P_logits = jit(get_P_logits)(P_params, full_l_batch, rng_key)
    batched_P_samples = jit(ds.sample_hard_batched_logits)(
        batched_P_logits, tau, rng_key,
    )

    def sample_W(L, P):
        return (P @ L @ P.T).T

    Ws = jit(vmap(sample_W))(batched_lower_samples, batched_P_samples)
    eid = ensemble_intervention_distance(
        ground_truth_W,
        Ws,
        onp.exp(log_sigma_W),
        onp.exp(w_noise) * onp.ones(dim),
        sem_type,
    )
    return eid

@jit
def parallel_soft_elbo_estimate(P_params, L_params, L_states, Xs, rng_keys, tau):
    elbos, _ = soft_elbo(P_params, L_params, L_states, Xs, rng_keys, tau)
    return jnp.mean(elbos)

@jit
def parallel_hard_elbo_estimate(P_params, L_params, L_states, Xs, rng_keys, tau):
    elbos, _ = hard_elbo(P_params, L_params, L_states, Xs, rng_keys, tau)
    return jnp.mean(elbos)

@jit
def parallel_gradient_step(P_params, L_params, L_states, Xs, P_opt_state, L_opt_state, rng_key, tau,):
    rng_key, rng_key_2 = rnd.split(rng_key, 2)
    tau_scaling_factor = 1.0 / tau

    (_, L_states), grads = value_and_grad(hard_elbo, argnums=(0, 1), has_aux=True)(P_params, L_params, L_states, Xs, rng_key, tau)
    elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling_factor * x, grads)

    l2_elbo_grad_P = grad(
        lambda p: 0.5 * sum(jnp.sum(jnp.square(param)) for param in jax.tree_leaves(p))
    )(P_params)
    elbo_grad_P = tree_multimap(lambda x, y: x + y, elbo_grad_P, l2_elbo_grad_P)

    P_updates, P_opt_state = opt_P.update(elbo_grad_P, P_opt_state, P_params)
    P_params = optax.apply_updates(P_params, P_updates)
    L_updates, L_opt_state = opt_L.update(elbo_grad_L, L_opt_state, L_params)
    L_params = optax.apply_updates(L_params, L_updates)
    return (P_params, L_params, L_states, P_opt_state, L_opt_state, rng_key_2)


@jit
def compute_grad_variance(
    P_params, L_params, L_states, Xs, rng_key, tau,
):
    
    (_, L_states), grads = value_and_grad(hard_elbo, argnums=(0, 1), has_aux=True)(
        P_params, L_params, L_states, Xs, rng_key, tau
    )

    return get_double_tree_variance(*grads)


def tau_schedule(i):
    boundaries = jnp.array([5_000, 10_000, 20_000, 60_000, 100_000])
    values = jnp.array([30.0, 10.0, 1.0, 1.0, 0.5, 0.25])
    index = jnp.sum(boundaries < i)
    return jnp.take(values, index)


t0 = time.time()
t_prev_batch = t0
if fixed_tau is not None:
    tau = fixed_tau
else:
    tau = tau_schedule(0)


soft_elbo_ = parallel_soft_elbo_estimate(P_params, L_params, L_states, Xs, rng_key, tau)
steps_t0 = time.time()
best_elbo = -jnp.inf
mean_dict = {}
t00 = 0.0

for i in range(num_steps):
    (
        P_params,
        new_L_params,
        L_states,
        P_opt_params,
        new_L_opt_params,
        rng_key,
    ) = parallel_gradient_step(
        P_params, L_params, L_states, Xs, P_opt_params, L_opt_params, rng_key, tau,
    )
    if jnp.any(jnp.isnan(ravel_pytree(new_L_params)[0])):   print("Got NaNs in L params")
    L_params = new_L_params
    L_opt_params = new_L_opt_params

    if i == 0:
        print(f"Compiled gradient step after {time.time() - t0}s")
        t00 = time.time()
    
    if i % 10 == 0: print(f'Step {i}/{num_steps}')

    if i % 100 == 0:
        if fixed_tau is None: tau = tau_schedule(i)
        t000 = time.time()
        current_elbo = parallel_hard_elbo_estimate(P_params, L_params, L_states, Xs, rng_key, tau)
        soft_elbo_ = parallel_soft_elbo_estimate(P_params, L_params, L_states, Xs, rng_key, tau)
        num_steps_to_converge = get_num_sinkhorn_steps(P_params, L_params, L_states, rng_key)
        if i == 1:    print(f"Compiled estimates after {time.time() - t00}s")
        
        print(f"After {i} iters, hard elbo is {ff2(current_elbo)}, soft elbo is {ff2(soft_elbo_)}")

        out_dict = {
            "ELBO": onp.array(current_elbo),
            "soft ELBO": onp.array(soft_elbo_),
            "tau": onp.array(tau),
            "Wall Time": onp.array(time.time() - t0),
            "Sinkhorn steps": onp.array(num_steps_to_converge),
        }

        if wandb is not None:
            wandb.log(out_dict, step=i)

        t_prev_batch = time.time()

        if (i % 500 == 0) and (((time.time() - steps_t0) > 30) or (i % 1_000 == 0)):
            # Log evalutation metrics at most once every two minutes
            
            _do_shd_c = False
            elbo_grad_std = compute_grad_variance(P_params, L_params, L_states, Xs, rng_key, tau,)

            try:
                mean_dict = eval_mean(P_params, L_params, L_states, Xs, rk(i), _do_shd_c,)
                train_mean_dict = eval_mean(P_params, L_params, L_states, Xs, rk(i), _do_shd_c,)
                
            except:
                print("Error occured in evaluating test statistics")
                continue

            if current_elbo > best_elbo:
                best_elbo = current_elbo
                best_shd = mean_dict["shd"]
                if wandb is not None:
                    wandb.log({"best elbo": onp.array(best_elbo)}, step=i)
                    wandb.log({"best shd": onp.array(mean_dict["shd"])}, step=i)

            if eval_eid and i % 8_000 == 0:
                t4 = time.time()
                eid = eval_ID(P_params, L_params, L_states, Xs, rk(i), tau,)
                if wandb is not None:
                    wandb.log({"eid_wass": eid}, step=i)
                    print(f"EID_wass is {eid}, after {time.time() - t4}s")
            
            print(f"Iter: {i} | MSE: {ff2(mean_dict['MSE'])} | SHD: {ff2(mean_dict['shd'])}")
            metrics_ = (
                {
                    "shd": mean_dict["shd"],
                    "shd_c": mean_dict["shd_c"],
                    "sid": mean_dict["sid"],
                    "mse": mean_dict["MSE"],
                    "tpr": mean_dict["tpr"],
                    "fdr": mean_dict["fdr"],
                    "fpr": mean_dict["fpr"],
                    "auroc": mean_dict["auroc"],
                    "ELBO Grad std": onp.array(elbo_grad_std),
                    "true KL": mean_dict["true_kl"],
                    "true Wasserstein": mean_dict["true_wasserstein"],
                    "sample KL": mean_dict["sample_kl"],
                    "sample Wasserstein": mean_dict["sample_wasserstein"],
                    "pred_size": mean_dict["pred_size"],
                    "train sample KL": train_mean_dict["sample_kl"],
                    "train sample Wasserstein": train_mean_dict["sample_wasserstein"],
                    "pred_size": mean_dict["pred_size"],
                },
            )

            if wandb is not None:
                wandb.log(
                    metrics_[0], step=i,
                )
            else:
                print(metrics_)
            exit_condition = (
                (i > 10_000)
                and train_mean_dict["tpr"] < 0.5
                and ((time.time() - steps_t0) > 3_600)
            ) or ((train_mean_dict["shd"] > 300) and i > 10)
            exit_condition = False
            if exit_condition:
                # While doing sweeps we don't want the runs to drag on for longer than
                # would be reasonable to run them for
                # So if runs are taking more than 20 mins per 400, we cut them
                print(
                    f"Exiting after {time.time() - t0}s, avg time {(time.time() - t0) * 400 / (i + 1)}, tpr {train_mean_dict['tpr']}"
                )

            print("Plotting fig...")
            full_l_batch, _, _ = jit(sample_L, static_argnums=3)(L_params, L_states, rk(i))
            P_logits = get_P_logits(P_params, full_l_batch, rk(i))
            batched_P_samples = jit(ds.sample_hard_batched_logits)(P_logits, tau, rk(i))
            our_W = (
                batched_P_samples[0]
                @ lower(full_l_batch[0, :l_dim], dim)
                @ batched_P_samples[0].T
            ).T
            plt.imshow(our_W)
            plt.colorbar()
            if wandb is not None:
                plt.savefig(f"./tmp/tmp_{wandb.run.name}.png")
            plt.close()
            if wandb is not None:
                wandb.log(
                    {
                        "Sample": [
                            wandb.Image(
                                Image.open(f"./tmp/tmp_{wandb.run.name}.png"),
                                caption="W sample",
                            )
                        ]
                    },
                    step=i,
                )
            batched_soft_P_samples = jit(ds.sample_soft_batched_logits)(
                P_logits, tau, rk(i)
            )
            our_W_soft = (
                batched_soft_P_samples[0]
                @ lower(full_l_batch[0, :l_dim], dim)
                @ batched_soft_P_samples[0].T
            ).T
            plt.imshow(our_W_soft)
            plt.colorbar()
            if wandb is not None:
                plt.savefig(f"./tmp/tmp_{wandb.run.name}.png")
            plt.close()
            if wandb is not None:
                wandb.log(
                    {
                        "SoftSample": [
                            wandb.Image(
                                Image.open(f"./tmp/tmp_{wandb.run.name}.png"),
                                caption="W sample_soft",
                            )
                        ]
                    },
                    step=i,
                )

            print(f"Max value of P_logits was {ff2(jnp.max(jnp.abs(P_logits)))}")
            steps_t0 = time.time()


@jit
def get_W(P, L):
    return (P @ L @ P.T).T

@jit
def diag(noises):
    return jnp.diag(noises)

rounds = int(((1000 // batch_size) + int(1000 % batch_size != 0)))
res_g, res_w, res_noise = [], [], []

if args.use_my_data:

    for round in range(rounds): 
        rng_key, rng_key_1 = rnd.split(rng_key, 2)
        full_l_batch, full_log_prob_l, out_L_states = sample_L(L_params, L_states, rng_key)
        w_noise = full_l_batch[:, -noise_dim:]
        l_batch = full_l_batch[:, :-noise_dim]
        batched_noises = jnp.ones((batch_size, dim)) * w_noise.reshape((batch_size, noise_dim))
        batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, dim)
        batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key_1)
        batched_P_samples = ds.sample_hard_batched_logits(batched_P_logits, tau, rng_key)

        batched_Sigma = onp.array(vmap(diag, (0), (0))(jnp.exp(2 * batched_noises)))
        batched_W = onp.array(vmap(get_W, (0, 0), (0))(batched_P_samples, batched_lower_samples))
        binary_G = onp.array(jnp.where(jnp.abs(batched_W) >= 0.3, 1, 0))

        res_g.append(binary_G)
        res_w.append(batched_W)
        res_noise.append(batched_Sigma)

    res_g = onp.concatenate(res_g, axis=0)[:1000]
    res_w = onp.concatenate(res_w, axis=0)[:1000]
    res_noise = onp.concatenate(res_noise, axis=0)[:1000]
    os.makedirs(f'{data_path}/{seed_folder}/bcd', exist_ok=True)

    try:
        with open(f'{data_path}/{seed_folder}/bcd/predicted_graphs.npy','wb') as f:
            onp.save(f, res_g)

        with open(f'{data_path}/{seed_folder}/bcd/predicted_thetas.npy','wb') as f:
            onp.save(f, res_w)

        with open(f'{data_path}/{seed_folder}/bcd/predicted_Sigmas.npy','wb') as f:
            onp.save(f, res_noise)
    except:
        print("Error: No space or wrong saving directory")