# structure-learning-baselines

Before running any of these baselines, make sure wandb artifacts are downloaded and stored in `~/scratch`. That is, the path `~/scratch/artifacts` should exist.

- Change the default value of the arg `--root_path` in `main.py`, to your artifacts directory like `<your_path>/artifacts/`
## 1. BCD Nets
<b>Environment setup:</b> Run `cd BCD-Nets && bash setup.sh`. In the BCD-Nets folder, also run `mkdir slurm_runs`. This is where the slurm .out files will be outputted.

### Running BCD Nets
- To run BCD Nets on seed 1 of the wandb artifact, with 20 nodes and 10000 steps, run: `python main.py -s 1 --dim 20 --num_steps 10000 --do_ev_noise --sem_type linear-gauss --batch_size 256 --use_my_data True --data_path <artifact_folder_name> --use_wandb`. Examples of artifact_folder_name: 'er1-lingauss-d005:v0', 'er2-lingauss-d020:v1', 'er2-lingauss-d050:v1'. Remember to include the version v0, v1 etc.

- The results will be saved in `~/scratch/artifacts/<artifact_folder_name>/<seed_number>/bcd/`. Three numpy files will be created: `predicted_graphs.npy`, `predicted_thetas.npy`, and `predicted_Sigmas.npy`.

- <b>Note</b>: As a baseline method, learning noise variance is switched off by default and the error variance is set to 0.1 (see the variable log_noise_sigma which saves the log std deviation in line 247). Learning noise variance from data can be switched on by including `--learn_noise True`, if needed.

### Launching job arrays
1. If you want to launch SLURM jobs, first set the list of seeds for which you want to launch a job array.
For eg., setting `seeds=(0 2 3 5 7 8)` will run the jobs only for those seeds. You will have to set this <b>both</b> in `runner.sh` <b>and</b> in `bcd_job.sh` before running your jobs. Otherwise you will face unexpected behaviour as to which seeds are run. 

2. Launch job array with `bash runner.sh <num_nodes> <num_steps> <artifact_folder_name> <time_for_job>`. Example: `bash runner.sh 20 10000 sbm1-lingauss-d020:v1 6:00:00` will run 10000 steps on the artifact and request nodes for 6h jobs.

## 2. DiBS

<b>Environment setup:</b> Run `cd dibs && bash setup.sh` to create the required environment and install packages.

### Running DiBS
- To run DiBS on seed 1 of the wandb artifact, with 20 nodes and 1000 particles, run: `python dibs_exp.py -s 1 -d 20 --particles 1000 --data_path <artifact_folder_name>`. Examples of artifact_folder_name: 'er1-lingauss-d005:v0', 'er2-lingauss-d020:v1', 'er2-lingauss-d050:v1'. Remember to include the version v0, v1 etc.

- The results will be saved in `~/scratch/artifacts/<artifact_folder_name>/<seed_number>/DiBS/`. Two numpy files will be created: `predicted_graphs.npy` and `predicted_thetas.npy`. Currently supports only joint inference, but can support marginal inference with minimal changes to code.

- <b>Note</b>: The error variance is set to 0.1 (can be changed via `--obs_noise_var`). Currently supports only linear Gausssian BNs but will soon be extended to include nonlinear experiments from the original paper.

### Launching job arrays
1. If you want to launch SLURM jobs, first set the list of seeds for which you want to launch a job array. For eg., setting `seeds=(0 2 3 5 7 8)` will run the jobs only for those seeds. You will have to set this <b>both</b> in `dibs_runner.sh` <b>and</b> in `dibs_job.sh` before running your jobs. Otherwise you will face unexpected behaviour as to which seeds are run. 

2. Launch job array with `bash dibs_runner.sh <num_nodes> <artifact_folder_name> <time_for_job>`. Example: `bash dibs_runner.sh 50 er1-lingauss-d050:v1 6:00:00` will run 3000 steps on the artifact and request nodes for 6h jobs.