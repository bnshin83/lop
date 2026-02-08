"""
PPO Training Script with Weights & Biases Integration

Based on the original run_ppo.py with added wandb tracking for
visualizing loss of plasticity experiments.

Implements predictive margin analysis to track neuron death risk:
- Predictive margin: θ_{k-1} on M_k (old weights on new data)
- Actual margin: θ_k on M_k (current weights on current data)
"""
import os
import copy
import yaml
import pickle
import argparse
import time
import subprocess
import numpy as np

import gymnasium as gym
import torch
from torch.optim import Adam

# UPGD optimizer imports
try:
    from lop.rl.optim import FirstOrderGlobalUPGD, FirstOrderGlobalUPGDLayerSelective
    UPGD_AVAILABLE = True
except ImportError:
    UPGD_AVAILABLE = False
    print("Warning: UPGD optimizers not available")

# wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb'")

import lop.envs
from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.agent import Agent
from lop.algos.rl.ppo import PPO
from lop.utils.miscellaneous import compute_matrix_rank_summaries  # , compute_margins_per_layer  # MARGIN DISABLED


def save_data(cfg, rets, termination_steps,
              pol_features_activity, stable_rank, mu, pol_weights, val_weights,
              action_probs=None, weight_change=[], friction=-1.0, num_updates=0, previous_change_time=0,
              ):  # margin_data=None removed - MARGIN DISABLED
    data_dict = {
        'rets': np.array(rets),
        'termination_steps': np.array(termination_steps),
        'pol_features_activity': pol_features_activity,
        'stable_rank': stable_rank,
        'action_output': mu,
        'pol_weights': pol_weights,
        'val_weights': val_weights,
        'action_probs': action_probs,
        'weight_change': torch.tensor(weight_change).numpy(),
        'friction': friction,
        'num_updates': num_updates,
        'previous_change_time': previous_change_time,
        # 'margin_data': margin_data  # MARGIN DISABLED
    }
    with open(cfg['log_path'], 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data(cfg):
    with open(cfg['log_path'], 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def save_checkpoint(cfg, step, learner):
    # Save step, model and optimizer states
    ckpt_dict = dict(
        step = step,
        actor = learner.pol.state_dict(),
        critic = learner.vf.state_dict(),
        opt = learner.opt.state_dict()
    )
    torch.save(ckpt_dict, cfg['ckpt_path'])
    print(f'Save checkpoint at step={step}')


def load_checkpoint(cfg, device, learner):
    # Load step, model and optimizer states
    step = 0
    ckpt_dict = torch.load(cfg['ckpt_path'], map_location=device)
    step = ckpt_dict['step']
    learner.pol.load_state_dict(ckpt_dict['actor'])
    learner.vf.load_state_dict(ckpt_dict['critic'])
    learner.opt.load_state_dict(ckpt_dict['opt'])
    print(f"Successfully restore from checkpoint: {cfg['ckpt_path']}.")
    return step, learner


def init_wandb(cfg, args):
    """Initialize wandb with configuration."""
    if not WANDB_AVAILABLE:
        return None
    
    # Get project and run name from environment or defaults
    project = os.environ.get('WANDB_PROJECT', 'loss-of-plasticity-rl')
    
    # Extract config name from path (e.g., 'std' from 'cfg/ant/std.yml')
    config_name = os.path.basename(args.config).replace('.yml', '').replace('.json', '')
    env_name = cfg.get('env_name', 'unknown').replace('-', '_').lower()
    
    # Use WANDB_RUN_NAME if set, otherwise generate default with job ID
    if 'WANDB_RUN_NAME' in os.environ:
        run_name = os.environ['WANDB_RUN_NAME']
    else:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', '')
        run_name = f"{env_name}_{config_name}_seed{args.seed}"
        if slurm_job_id:
            run_name = f"{slurm_job_id}_{run_name}"
    
    # Initialize wandb
    wandb.init(
        project=project,
        name=run_name,
        config={
            **cfg,
            'seed': args.seed,
            'config_file': args.config,
            'device': args.device,
        },
        mode=os.environ.get('WANDB_MODE', 'online'),
        tags=[env_name, config_name, f'seed_{args.seed}'],
        reinit=True
    )
    
    return wandb.run


# === MARGIN ANALYSIS FUNCTIONS DISABLED ===
# def log_margin_to_wandb(step, margin_results, update_number, cfg, prefix='margin'):
#     ...
# def log_predictive_margin_comparison(step, predictive_results, actual_results, update_number, cfg):
#     ...
# === END MARGIN ANALYSIS FUNCTIONS ===


def log_to_wandb(step, rets, termination_steps, pol_features_activity, 
                 stable_rank, pol_weights, val_weights, cfg, to_log, pol=None, vf=None):
    """Log metrics to wandb."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    log_dict = {}
    
    # =========================================
    # 1. EPISODIC RETURN (Total Episode Reward)
    # =========================================
    if len(rets) > 0:
        # Most recent episode return
        log_dict['episode_return'] = float(rets[-1])
        
        # Moving averages (smoothed returns)
        if len(rets) >= 10:
            log_dict['episode_return_avg10'] = float(np.mean(rets[-10:]))
        if len(rets) >= 100:
            log_dict['episode_return_avg100'] = float(np.mean(rets[-100:]))
        
        # Cumulative average
        log_dict['episode_return_cumulative_avg'] = float(np.mean(rets))
    
    log_dict['num_episodes'] = len(rets)
    
    # =========================================
    # 2. PERCENTAGE OF DORMANT UNITS
    # =========================================
    # Dormant units = neurons that are active <= threshold fraction of the time
    # Following the original paper's definition (fig4b.py): threshold = 0.01 (1%)
    # A neuron is "dormant" if it fires on <= 1% of observations
    DORMANT_THRESHOLD = 0.01  # 1% activity threshold
    dormant_interval = cfg.get('dormant_log_interval', 1024)
    stable_interval = cfg.get('stable_rank_interval', 10240)
    if 'pol_features_activity' in to_log:
        try:
            if hasattr(pol_features_activity, 'shape') and pol_features_activity.shape[0] > 0:
                idx = min(step // dormant_interval, pol_features_activity.shape[0] - 1)
                if idx >= 0:
                    activity = pol_features_activity[idx]
                    if hasattr(activity, 'numpy'):
                        activity = activity.numpy()
                    elif hasattr(activity, 'cpu'):
                        activity = activity.cpu().numpy()

                    # Per-layer dormant percentage (fraction of neurons with activity <= threshold)
                    layer_dormant_pcts = []
                    for layer_idx in range(len(activity)):
                        layer_act = activity[layer_idx]  # activity per neuron in this layer
                        dormant_count = np.sum(layer_act <= DORMANT_THRESHOLD)
                        dormant_pct = (dormant_count / len(layer_act)) * 100
                        layer_dormant_pcts.append(dormant_pct)
                        log_dict[f'dormant_units_pct_layer{layer_idx}'] = dormant_pct
                        log_dict[f'active_units_pct_layer{layer_idx}'] = 100 - dormant_pct

                    # Average dormant percentage across all layers
                    # Flatten all neurons and compute fraction that are dormant
                    all_activities = activity.flatten()
                    total_dormant = np.sum(all_activities <= DORMANT_THRESHOLD)
                    dormant_pct_avg = (total_dormant / len(all_activities)) * 100
                    log_dict['dormant_units_pct_avg'] = dormant_pct_avg
                    log_dict['active_units_pct_avg'] = 100 - dormant_pct_avg
        except Exception as e:
            pass  # Skip if error

    # =========================================
    # 3. AVERAGE WEIGHT MAGNITUDE
    # =========================================
    # Policy network weights
    if 'pol_weights' in to_log:
        try:
            if hasattr(pol_weights, 'shape') and pol_weights.shape[0] > 0:
                idx = min(step // dormant_interval, pol_weights.shape[0] - 1)
                if idx >= 0:
                    weight_mags = []
                    for layer_idx in range(pol_weights.shape[1]):
                        w = float(pol_weights[idx, layer_idx])
                        log_dict[f'pol_weight_mag_layer{layer_idx}'] = w
                        weight_mags.append(w)
                    # Average across all layers
                    log_dict['pol_weight_mag_avg'] = float(np.mean(weight_mags))
        except Exception as e:
            pass

    # Value network weights
    if 'val_weights' in to_log:
        try:
            if hasattr(val_weights, 'shape') and val_weights.shape[0] > 0:
                idx = min(step // dormant_interval, val_weights.shape[0] - 1)
                if idx >= 0:
                    weight_mags = []
                    for layer_idx in range(val_weights.shape[1]):
                        w = float(val_weights[idx, layer_idx])
                        log_dict[f'val_weight_mag_layer{layer_idx}'] = w
                        weight_mags.append(w)
                    # Average across all layers
                    log_dict['val_weight_mag_avg'] = float(np.mean(weight_mags))
        except Exception as e:
            pass
    
    # =========================================
    # 4. STABLE RANK (effective dimensionality)
    # =========================================
    # Stable rank measures the effective dimensionality of feature representations
    # Scale to [0, 100] by dividing by hidden dimension (256) and multiplying by 100
    if 'stable_rank' in to_log:
        try:
            if hasattr(stable_rank, 'shape') and stable_rank.shape[0] > 0:
                idx = min(step // stable_interval, stable_rank.shape[0] - 1)
                if idx >= 0:
                    rank_val = stable_rank[idx]
                    if hasattr(rank_val, 'item'):
                        raw_rank = rank_val.item()
                    else:
                        raw_rank = float(rank_val)
                    
                    # Get hidden dimension from config (default 256)
                    h_dim = cfg.get('h_dim', [256, 256])[0]
                    
                    # Raw stable rank
                    log_dict['stable_rank'] = raw_rank
                    # Scaled stable rank [0, 100]
                    log_dict['stable_rank_scaled_0_100'] = (raw_rank / h_dim) * 100
        except Exception as e:
            pass
    
    # Log all metrics at this step
    if log_dict:
        wandb.log(log_dict, step=step)


def main():
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, default='./cfg/ant/std.yml')
    parser.add_argument('-s', '--seed', required=False, type=int, default="1")
    parser.add_argument('-d', '--device', required=False, default='cpu')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh even if checkpoint/logs exist')
    parser.add_argument('--run-suffix', type=str, default=None, help='Optional suffix for output directory')

    args = parser.parse_args()
    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed

    run_suffix = args.run_suffix
    if args.no_resume and not run_suffix:
        run_suffix = f"fresh_{time.strftime('%Y%m%d_%H%M%S')}"
    if run_suffix:
        base_dir = cfg['dir'].rstrip('/')
        cfg['dir'] = os.path.join(base_dir, run_suffix)
    cfg['dir'] = cfg['dir'].rstrip('/') + '/'
    cfg['log_path'] = cfg['dir'] + str(args.seed) + '.log'
    cfg['ckpt_path'] = cfg['dir'] + str(args.seed) + '.pth'
    cfg['done_path'] = cfg['dir'] + str(args.seed) + '.done'

    bash_command = "mkdir -p " + cfg['dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    # Set default values
    cfg.setdefault('wd', 0)
    cfg.setdefault('init', 'lecun')
    cfg.setdefault('to_log', [])
    cfg.setdefault('beta_1', 0.9)
    cfg.setdefault('beta_2', 0.999)
    cfg.setdefault('eps', 1e-8)
    cfg.setdefault('no_clipping', False)
    cfg.setdefault('loss_type', 'ppo')
    cfg.setdefault('frictions_file', 'cfg/frictions')
    cfg.setdefault('max_grad_norm', 1e9)
    cfg.setdefault('perturb_scale', 0)
    # cfg.setdefault('margin_log_interval', 1)  # MARGIN DISABLED
    cfg.setdefault('dormant_log_interval', 1024)  # Log dormant units every N steps (aligned with bs/2)
    cfg.setdefault('stable_rank_interval', 10240)  # Log stable rank every N steps (10 * dormant_log_interval)
    cfg['n_steps'] = int(float(cfg['n_steps']))
    cfg['perturb_scale'] = float(cfg['perturb_scale'])
    n_steps = cfg['n_steps']

    # Set default values for CBP
    cfg.setdefault('mt', 10000)
    cfg.setdefault('rr', 0)
    cfg['rr'] = float(cfg['rr'])
    cfg.setdefault('decay_rate', 0.99)
    cfg.setdefault('redo', False)
    cfg.setdefault('threshold', 0.03)
    cfg.setdefault('reset_period', 1000)
    cfg.setdefault('util_type_val', 'contribution')
    cfg.setdefault('util_type_pol', 'contribution')
    cfg.setdefault('pgnt', (cfg['rr']>0) or cfg['redo'])
    cfg.setdefault('vgnt', (cfg['rr']>0) or cfg['redo'])

    # UPGD optimizer settings
    cfg.setdefault('optimizer_type', 'adam')  # 'adam', 'upgd', or 'upgd_layer_selective'
    cfg.setdefault('beta_utility', 0.9999)
    cfg.setdefault('sigma', 0.001)
    cfg.setdefault('gating_mode', 'full')  # 'full', 'output_only', 'hidden_only'
    cfg.setdefault('non_gated_scale', 0.5)  # scaling for non-gated layers
    cfg.setdefault('gating_fn', 'sigmoid')  # 'sigmoid' (original) or 'clamp'
    # wandb
    if not args.no_wandb and WANDB_AVAILABLE:
        init_wandb(cfg, args)
        print(f"WandB initialized: {wandb.run.name if wandb.run else 'None'}")
    else:
        print("WandB logging disabled")

    # Initialize env
    seed = cfg['seed']
    friction = -1.0
    if cfg['env_name'] in ['SlipperyAnt-v2', 'SlipperyAnt-v3']:
        xml_file = os.path.abspath(cfg['dir'] + f'slippery_ant_{seed}.xml')
        cfg.setdefault('friction', [0.02, 2])
        cfg.setdefault('change_time', int(2e6))

        with open(cfg['frictions_file'], 'rb+') as f:
            frictions = pickle.load(f)
        friction_number = 0
        new_friction = frictions[seed][friction_number]

        if friction < 0: # If no saved friction, use the default value 1.0
            friction = 1.0
        env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
        print(f'Initial friction: {friction:.6f}')
    else:
        env = gym.make(cfg['env_name'])
    env.name = None

    # Set random seeds
    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    
    # Initialize algorithm - select optimizer based on config
    optimizer_type = cfg.get('optimizer_type', 'adam')
    if optimizer_type == 'upgd' and UPGD_AVAILABLE:
        print(f"Using UPGD optimizer (beta_utility={cfg['beta_utility']}, sigma={cfg['sigma']})")
        opt = FirstOrderGlobalUPGD
    elif optimizer_type == 'upgd_layer_selective' and UPGD_AVAILABLE:
        print(f"Using UPGD LayerSelective optimizer (gating_mode={cfg['gating_mode']}, sigma={cfg['sigma']})")
        opt = FirstOrderGlobalUPGDLayerSelective
    else:
        if optimizer_type != 'adam':
            print(f"Warning: Requested optimizer '{optimizer_type}' not available, falling back to Adam")
        opt = Adam
    num_layers = len(cfg['h_dim'])
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    pol = MLPPolicy(o_dim, a_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    vf = MLPVF(o_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    np.random.set_state(random_state)
    buf = Buffer(o_dim, a_dim, cfg['bs'], device=device)

    learner = PPO(pol, buf, cfg['lr'], g=cfg['g'], vf=vf, lm=cfg['lm'], Opt=opt,
                  u_epi_up=cfg['u_epi_ups'], device=device, n_itrs=cfg['n_itrs'], n_slices=cfg['n_slices'],
                  u_adv_scl=cfg['u_adv_scl'], clip_eps=cfg['clip_eps'],
                  max_grad_norm=cfg['max_grad_norm'], init=cfg['init'],
                  wd=float(cfg['wd']),
                  betas=(cfg['beta_1'], cfg['beta_2']), eps=float(cfg['eps']), no_clipping=cfg['no_clipping'],
                  loss_type=cfg['loss_type'], perturb_scale=cfg['perturb_scale'],
                  util_type_val=cfg['util_type_val'], replacement_rate=cfg['rr'], decay_rate=cfg['decay_rate'],
                  vgnt=cfg['vgnt'], pgnt=cfg['pgnt'], util_type_pol=cfg['util_type_pol'], mt=cfg['mt'],
                  redo=cfg['redo'], threshold=cfg['threshold'], reset_period=cfg['reset_period'],
                  beta_utility=cfg['beta_utility'], sigma=cfg['sigma'],
                  gating_mode=cfg['gating_mode'], non_gated_scale=cfg['non_gated_scale'],
                  gating_fn=cfg['gating_fn'])
    to_log = cfg['to_log']
    agent = Agent(pol, learner, device=device, to_log_features=(len(to_log) > 0))

    # Load checkpoint
    if not args.no_resume and os.path.exists(cfg['ckpt_path']):
        start_step, agent.learner = load_checkpoint(cfg, device, agent.learner)
    else:
        start_step = 0
    
    # Initialize log
    if not args.no_resume and os.path.exists(cfg['log_path']):
        data_dict = load_data(cfg)
        num_updates = data_dict['num_updates']
        previous_change_time = data_dict['previous_change_time']
        for k, v in data_dict.items():
            if k == 'margin_data':
                continue  # Don't convert margin_data dict to list
            try:
                data_dict[k] = list(v)
            except:
                pass
        rets = data_dict['rets']
        termination_steps = data_dict['termination_steps']
        pol_features_activity = data_dict['pol_features_activity']
        stable_rank = data_dict['stable_rank']
        dormant_interval = cfg['dormant_log_interval']
        stable_interval = cfg['stable_rank_interval']
        if 'pol_features_activity' in to_log:
            short_term_feature_activity = torch.zeros(size=(dormant_interval, num_layers, cfg['h_dim'][0]))
            if len(pol_features_activity) > 0:
                pol_features_activity = torch.stack(pol_features_activity)
            else:
                pol_features_activity = torch.zeros(size=(n_steps//dormant_interval + 2, num_layers, cfg['h_dim'][0]))
        if 'stable_rank' in to_log:
            if len(stable_rank) > 0:
                stable_rank = torch.stack(stable_rank)
            else:
                stable_rank = torch.zeros(size=(n_steps//stable_interval + 2,))
        mu = data_dict['action_output']
        if 'mu' in to_log:
            mu = np.array(mu)
        pol_weights = data_dict['pol_weights']
        if 'pol_weights' in to_log:
            pol_weights = np.array(pol_weights)
        val_weights = data_dict['val_weights']
        if 'val_weights' in to_log:
            val_weights = np.array(val_weights)
        weight_change = data_dict['weight_change']
        # MARGIN DISABLED - no margin_data loading
    else:
        num_updates = 0
        previous_change_time = 0
        rets, termination_steps = [], []
        mu, weight_change, pol_features_activity, stable_rank, pol_weights, val_weights = [], [], [], [], [], []
        dormant_interval = cfg['dormant_log_interval']
        stable_interval = cfg['stable_rank_interval']
        if 'mu' in to_log:
            mu = np.ones(size=(n_steps, a_dim))
        if 'pol_weights' in to_log:
            pol_weights = np.zeros(shape=(n_steps//dormant_interval + 2, (len(pol.mean_net)+1)//2))
        if 'val_weights' in to_log:
            val_weights = np.zeros(shape=(n_steps//dormant_interval + 2, (len(pol.mean_net)+1)//2))
        if 'pol_features_activity' in to_log:
            short_term_feature_activity = torch.zeros(size=(dormant_interval, num_layers, cfg['h_dim'][0]))
            pol_features_activity = torch.zeros(size=(n_steps//dormant_interval + 2, num_layers, cfg['h_dim'][0]))
        if 'stable_rank' in to_log:
            stable_rank = torch.zeros(size=(n_steps//stable_interval + 2,))
        # MARGIN DISABLED - no margin_data init

    # MARGIN DISABLED - window buffer and margin tracking removed
    update_count = 0  # Track number of PPO updates (still useful)

    ret = 0
    epi_steps = 0
    o, _ = env.reset()
    print('start_step:', start_step)

    # Dormant unit logging intervals (consistent across training loop)
    dormant_interval = cfg['dormant_log_interval']
    stable_interval = cfg['stable_rank_interval']
    print(f'Dormant/weight logging: every {dormant_interval} steps, stable rank: every {stable_interval} steps')

    # wandb logging interval
    wandb_log_interval = dormant_interval

    console_log_interval = int(cfg.get('console_log_interval', 10000))

    # Interaction loop
    for step in range(start_step, n_steps):
        a, logp, dist, new_features = agent.get_action(o)
        op, r, terminated, truncated, infos = env.step(a)
        done = terminated or truncated
        epi_steps += 1
        op_ = op
        val_logs = agent.log_update(o, a, r, op_, logp, dist, done)
        # Logging
        with torch.no_grad():
            if 'weight_change' in to_log and 'weight_change' in val_logs.keys(): weight_change.append(val_logs['weight_change'])
            if 'mu' in to_log: mu[step] = a

            # MARGIN DISABLED - window buffer tracking removed

            if step % dormant_interval == 0:
                if step % stable_interval == 0 and 'stable_rank' in to_log:
                    _, _, _, stable_rank[step//stable_interval] = compute_matrix_rank_summaries(m=short_term_feature_activity[:, -1, :], use_scipy=True)
                if 'pol_features_activity' in to_log:
                    pol_features_activity[step//dormant_interval] = (short_term_feature_activity>0).float().mean(dim=0)
                    short_term_feature_activity *= 0
                if 'pol_weights' in to_log:
                    for layer_idx in range((len(pol.mean_net) + 1) // 2):
                        pol_weights[step//dormant_interval, layer_idx] = pol.mean_net[2 * layer_idx].weight.data.abs().mean()
                if 'val_weights' in to_log:
                    for layer_idx in range((len(learner.vf.v_net) + 1) // 2):
                        val_weights[step//dormant_interval, layer_idx] = learner.vf.v_net[2 * layer_idx].weight.data.abs().mean()
            if 'pol_features_activity' in to_log:
                for i in range(num_layers):
                    short_term_feature_activity[step % dormant_interval, i] = new_features[i]

        o = op
        ret += r
        if done:
            if step % console_log_interval != 0:
                print(f"Episode {len(rets) + 1} done @ step {step} | len={epi_steps} | return={float(ret):.2f}", flush=True)
            rets.append(ret)
            termination_steps.append(step)
            
            # Log to wandb on episode completion
            if WANDB_AVAILABLE and wandb.run is not None:
                episode_log = {
                    'episode/return': float(ret),
                    'episode/length': epi_steps,
                    'episode/count': len(rets),
                }
                # Add running averages
                if len(rets) >= 10:
                    episode_log['episode/return_avg10'] = float(np.mean(rets[-10:]))
                if len(rets) >= 100:
                    episode_log['episode/return_avg100'] = float(np.mean(rets[-100:]))
                wandb.log(episode_log, step=step)
            
            ret = 0
            epi_steps = 0
            if cfg['env_name'] in ['SlipperyAnt-v2', 'SlipperyAnt-v3'] and step - previous_change_time > cfg['change_time']:
                previous_change_time = step
                env.close()
                friction_number += 1
                new_friction = frictions[seed][friction_number]
                print(f'{step}: change friction to {new_friction:.6f}')
                env.close()
                env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
                env.name = None
                agent.env = env
            o, _ = env.reset()

        # Periodic wandb logging for metrics
        if step % wandb_log_interval == 0 and step > 0:
            log_to_wandb(step, rets, termination_steps, pol_features_activity,
                        stable_rank, pol_weights, val_weights, cfg, to_log)
            
            # Log UPGD utility statistics if using UPGD optimizer
            if WANDB_AVAILABLE and wandb.run is not None:
                opt = agent.learner.opt
                if hasattr(opt, 'get_utility_stats'):
                    utility_stats = opt.get_utility_stats()
                    if utility_stats:
                        wandb.log(utility_stats, step=step)

        if step % console_log_interval == 0 and step > start_step:
            if len(rets) > 0:
                avg_ret = np.mean(rets[-100:]) if len(rets) >= 100 else np.mean(rets)
                print(f"Step {step}/{n_steps} | Episodes: {len(rets)} | Avg Return (last 100): {avg_ret:.2f}", flush=True)
            else:
                print(f"Step {step}/{n_steps} | Episodes: 0", flush=True)

        if step % (n_steps//100) == 0 or step == n_steps-1:
            # Save checkpoint
            save_checkpoint(cfg, step, agent.learner)
            # Save data logs
            save_data(cfg=cfg, rets=rets, termination_steps=termination_steps,
                      pol_features_activity=pol_features_activity, stable_rank=stable_rank, mu=mu, pol_weights=pol_weights,
                      val_weights=val_weights, weight_change=weight_change, friction=friction,
                      num_updates=num_updates, previous_change_time=previous_change_time)  # margin_data removed
            
            # Log progress
            if len(rets) > 0:
                progress = (step / n_steps) * 100
                avg_ret = np.mean(rets[-100:]) if len(rets) >= 100 else np.mean(rets)
                print(f"Step {step}/{n_steps} ({progress:.1f}%) | Episodes: {len(rets)} | Avg Return (last 100): {avg_ret:.2f}", flush=True)

    # Final wandb logging
    if WANDB_AVAILABLE and wandb.run is not None:
        # Log final summary
        wandb.run.summary['final_avg_return_100'] = np.mean(rets[-100:]) if len(rets) >= 100 else np.mean(rets)
        wandb.run.summary['final_avg_return_all'] = np.mean(rets)
        wandb.run.summary['total_episodes'] = len(rets)
        wandb.run.summary['total_steps'] = n_steps
        wandb.run.summary['total_ppo_updates'] = update_count
        # MARGIN DISABLED - no margin summary
        wandb.finish()

    with open(cfg['done_path'], 'w') as f:
        f.write('All done!')
        print('The experiment finished successfully!')


if __name__ == "__main__":
    main()

