import jax, os
import pandas as pd
import jax.random as random
from dibs.inference import JointDiBS
from dibs.target import make_linear_gaussian_model
from argparse import ArgumentParser
from dibs.metrics import expected_shd, threshold_metrics
import numpy as onp
import wandb, pdb

parser = ArgumentParser()

parser.add_argument("-s", "--seed", type=int, default=1)
parser.add_argument("-d", "--num_nodes", type=int, default=50)
parser.add_argument("--particles", type=int, default=1000)
parser.add_argument("--num_steps", type=int, default=3000) # 3000 steps for all experiments according to the paper
parser.add_argument("--obs_noise_var", type=float, default=0.1)
parser.add_argument("--data_path", type=str, default='er1-lingauss-d050:v1')
parser.add_argument("--lin_gauss", type=bool, default=True)
parser.add_argument("--root_path", type=str, default='/home/mila/j/jithendaraa.subramanian/scratch/artifacts/')

args = parser.parse_args()

data_path = args.root_path + args.data_path 

if args.seed >= 10: seed_folder = str(args.seed)
else:               seed_folder = '0' + str(args.seed)

train_data_path = data_path + '/' + seed_folder + '/train_data.csv'
gt_graph_path = data_path + '/' + seed_folder + '/adjacency.npy'

x_csv = pd.read_csv(train_data_path)
x = jax.numpy.array(x_csv.to_numpy()[:, 1:])
ground_truth_G = onp.load(gt_graph_path)
n, d = x.shape
args.num_nodes = d

kernel_param = {"h_latent": 5.0, "h_theta": 500.0}
alpha_linear = 0.2

if args.lin_gauss:
    if d == 20:
        kernel_param = {"h_latent": 5.0, "h_theta": 500.0}
        alpha_linear = 0.2
    elif d >= 50:
        kernel_param = {"h_latent": 15.0, "h_theta": 1000.0}
        alpha_linear = 0.02

else:
    if d == 20:
        kernel_param = {"h_latent": 5.0, "h_theta": 1000.0}
        alpha_linear = 0.02
    elif d >= 50:
        kernel_param = {"h_latent": 15.0, "h_theta": 2000.0}
        alpha_linear = 0.01


key = random.PRNGKey(123)
print(f"JAX backend: {jax.default_backend()}")

key, subk = random.split(key)
_, model = make_linear_gaussian_model(key=subk, n_vars=args.num_nodes, graph_prior_str="sf", obs_noise=args.obs_noise_var) # obs_noise corresponds to variance

wandb.init(project="Learning DAGs")
configuration = {
    "dim": args.num_nodes,
    "seed": args.seed,
    "particles": args.particles,
    "num_steps": args.num_steps,
    'data_path': args.data_path,
    'model': 'DiBS',
    'obs_noise_std': onp.sqrt(args.obs_noise_var),
    'gamma_z': kernel_param['h_latent'],
    'gamma_theta': kernel_param['h_theta'],
    'linear_schedule': alpha_linear,
    'lin_gauss': args.lin_gauss
}
wandb.config.update(configuration)
print(configuration)
wandb_name = wandb.run.name
wandb_str = wandb_name.split("-")[0]  # type: ignore
wandb_string = (
    f"dibs_d_{args.num_nodes}_s_{args.seed}_{wandb_str[:4]}"
)
wandb.run.name = wandb_string

wandb.log(
        {   "Data": x_csv   },
        step=0,
    )

dibs = JointDiBS(x=x, interv_mask=None, inference_model=model, kernel_param=kernel_param, alpha_linear=alpha_linear)
key, subk = random.split(key)
gs, thetas = dibs.sample(key=subk, n_particles=args.particles, steps=args.num_steps, callback_every=50, callback=dibs.visualize_callback())

dibs_empirical = dibs.get_empirical(gs, thetas)
dibs_mixture = dibs.get_mixture(gs, thetas)

for descr, dist in [('DiBS ', dibs_empirical), ('DiBS+', dibs_mixture)]:
    eshd = expected_shd(dist=dist, g=ground_truth_G)        
    auroc = threshold_metrics(dist=dist, g=ground_truth_G)['roc_auc']
    print(f'{descr} |  E-SHD: {eshd:4.1f}    AUROC: {auroc:5.2f}  ')

os.makedirs(f'{data_path}/{seed_folder}/DiBS', exist_ok=True)

try:
    with open(f'{data_path}/{seed_folder}/DiBS/predicted_graphs.npy','wb') as f:
        onp.save(f, gs)

    with open(f'{data_path}/{seed_folder}/DiBS/predicted_thetas.npy','wb') as f:
        onp.save(f, thetas)

except:
    print("Error: No space or wrong saving directory")

print(gs.shape, thetas.shape)