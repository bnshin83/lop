"""PPO runner with WandB-enabled logging."""
import os
import copy
import yaml
import pickle
import argparse
import subprocess
import numpy as np

import gym
import torch
from torch.optim import Adam

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
from lop.utils.miscellaneous import compute_matrix_rank_summaries, compute_margins_per_layer


def save_data(cfg, rets, termination_steps,
              pol_features_activity, stable_rank, mu, pol_weights, val_weights,
              action_probs=None, weight_change=[], friction=-1.0, num_updates=0, previous_change_time=0,
              margin_data=None):
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
        'margin_data': margin_data
    }
    with open(cfg['log_path'], 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


def load_data(cfg):
    with open(cfg['log_path'], 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict


def save_checkpoint(cfg, step, learner):
    ckpt_dict = dict(
        step = step,
        actor = learner.pol.state_dict(),
        critic = learner.vf.state_dict(),
        opt = learner.opt.state_dict()
    )
    torch.save(ckpt_dict, cfg['ckpt_path'])
    print(f'Save checkpoint at step={step}')


def load_checkpoint(cfg, device, learner):
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
    
    project = os.environ.get('WANDB_PROJECT', 'loss-of-plasticity-rl')
    
    config_name = os.path.basename(args.config).replace('.yml', '').replace('.json', '')
    env_name = cfg.get('env_name', 'unknown').replace('-', '_').lower()
    
    if 'WANDB_RUN_NAME' in os.environ:
        run_name = os.environ['WANDB_RUN_NAME']
    else:
        slurm_job_id = os.environ.get('SLURM_JOB_ID', '')
        run_name = f"{env_name}_{config_name}_seed{args.seed}"
        if slurm_job_id:
            run_name = f"{slurm_job_id}_{run_name}"
    
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


def log_margin_to_wandb(step, margin_results, update_number, cfg, prefix='margin'):
    """Log margin analysis results to wandb for stationary environments."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    total_neurons = sum(cfg['h_dim'])
    total_dead = sum(margin_results['dead_counts'].values())
    total_at_risk = sum(margin_results['at_risk_counts'].values())
    total_always_on = sum(margin_results['always_on_counts'].values())

    log_dict = {
        f'{prefix}/total_dead': total_dead,
        f'{prefix}/total_at_risk': total_at_risk,
        f'{prefix}/total_always_on': total_always_on,
        f'{prefix}/dead_pct': 100 * total_dead / total_neurons,
        f'{prefix}/at_risk_pct': 100 * total_at_risk / total_neurons,
        f'{prefix}/always_on_pct': 100 * total_always_on / total_neurons,
        f'{prefix}/update_number': update_number,
    }

    for l in range(margin_results['num_hidden_layers']):
        layer_neurons = cfg['h_dim'][l]
        log_dict[f'{prefix}/layer{l}_dead'] = margin_results['dead_counts'][l]
        log_dict[f'{prefix}/layer{l}_dead_pct'] = 100 * margin_results['dead_counts'][l] / layer_neurons
        log_dict[f'{prefix}/layer{l}_at_risk'] = margin_results['at_risk_counts'][l]
        log_dict[f'{prefix}/layer{l}_at_risk_pct'] = 100 * margin_results['at_risk_counts'][l] / layer_neurons
        log_dict[f'{prefix}/layer{l}_always_on'] = margin_results['always_on_counts'][l]
        log_dict[f'{prefix}/layer{l}_mean_margin'] = margin_results['mean_margins'][l].mean().item()

    wandb.log(log_dict, step=step)


def log_predictive_margin_comparison(step, predictive_results, actual_results, update_number, cfg):
    """Log comparison between predictive and actual margins."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    total_neurons = sum(cfg['h_dim'])
    pred_dead = sum(predictive_results['dead_counts'].values())
    actual_dead = sum(actual_results['dead_counts'].values())
    pred_at_risk = sum(predictive_results['at_risk_counts'].values())
    actual_at_risk = sum(actual_results['at_risk_counts'].values())

    log_dict = {
        'predictive/predicted_dead': pred_dead,
        'predictive/predicted_dead_pct': 100 * pred_dead / total_neurons,
        'predictive/actual_dead': actual_dead,
        'predictive/actual_dead_pct': 100 * actual_dead / total_neurons,
        'predictive/predicted_at_risk': pred_at_risk,
        'predictive/actual_at_risk': actual_at_risk,
        'predictive/update_number': update_number,
    }

    for l in range(predictive_results['num_hidden_layers']):
        layer_neurons = cfg['h_dim'][l]
        log_dict[f'predictive/layer{l}_pred_dead'] = predictive_results['dead_counts'][l]
        log_dict[f'predictive/layer{l}_actual_dead'] = actual_results['dead_counts'][l]
        log_dict[f'predictive/layer{l}_pred_dead_pct'] = 100 * predictive_results['dead_counts'][l] / layer_neurons
        log_dict[f'predictive/layer{l}_actual_dead_pct'] = 100 * actual_results['dead_counts'][l] / layer_neurons

    wandb.log(log_dict, step=step)


def log_to_wandb(step, rets, termination_steps, pol_features_activity, 
                 stable_rank, pol_weights, val_weights, cfg, to_log, pol=None, vf=None):
    """Log metrics to wandb."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    log_dict = {}
    
    if len(rets) > 0:
        log_dict['episode_return'] = float(rets[-1])
        
        if len(rets) >= 10:
            log_dict['episode_return_avg10'] = float(np.mean(rets[-10:]))
        if len(rets) >= 100:
            log_dict['episode_return_avg100'] = float(np.mean(rets[-100:]))
        
        log_dict['episode_return_cumulative_avg'] = float(np.mean(rets))
    
    log_dict['num_episodes'] = len(rets)
    
    DORMANT_THRESHOLD = 0.01
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

                    layer_dormant_pcts = []
                    for layer_idx in range(len(activity)):
                        layer_act = activity[layer_idx]
                        dormant_count = np.sum(layer_act <= DORMANT_THRESHOLD)
                        dormant_pct = (dormant_count / len(layer_act)) * 100
                        layer_dormant_pcts.append(dormant_pct)
                        log_dict[f'dormant_units_pct_layer{layer_idx}'] = dormant_pct
                        log_dict[f'active_units_pct_layer{layer_idx}'] = 100 - dormant_pct

                    all_activities = activity.flatten()
                    total_dormant = np.sum(all_activities <= DORMANT_THRESHOLD)
                    dormant_pct_avg = (total_dormant / len(all_activities)) * 100
                    log_dict['dormant_units_pct_avg'] = dormant_pct_avg
                    log_dict['active_units_pct_avg'] = 100 - dormant_pct_avg
        except Exception:
            pass

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
                    log_dict['pol_weight_mag_avg'] = float(np.mean(weight_mags))
        except Exception as e:
            pass

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
                    log_dict['val_weight_mag_avg'] = float(np.mean(weight_mags))
        except Exception as e:
            pass
    
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
                    
                    h_dim = cfg.get('h_dim', [256, 256])[0]
                    
                    log_dict['stable_rank'] = raw_rank
                    log_dict['stable_rank_scaled_0_100'] = (raw_rank / h_dim) * 100
        except Exception as e:
            pass
    
    if log_dict:
        wandb.log(log_dict, step=step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, default='./cfg/ant/std.yml')
    parser.add_argument('-s', '--seed', required=False, type=int, default="1")
    parser.add_argument('-d', '--device', required=False, default='cpu')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()
    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    cfg['log_path'] = cfg['dir'] + str(args.seed) + '.log'
    cfg['ckpt_path'] = cfg['dir'] + str(args.seed) + '.pth'
    cfg['done_path'] = cfg['dir'] + str(args.seed) + '.done'

    bash_command = "mkdir -p " + cfg['dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

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
    cfg.setdefault('margin_log_interval', 1)
    cfg.setdefault('dormant_log_interval', 1024)
    cfg.setdefault('stable_rank_interval', 10240)
    cfg['n_steps'] = int(float(cfg['n_steps']))
    cfg['perturb_scale'] = float(cfg['perturb_scale'])
    n_steps = cfg['n_steps']

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

    if not args.no_wandb and WANDB_AVAILABLE:
        init_wandb(cfg, args)
        print(f"WandB initialized: {wandb.run.name if wandb.run else 'None'}")
    else:
        print("WandB logging disabled")

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

        if friction < 0:
            friction = 1.0
        env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
        print(f'Initial friction: {friction:.6f}')
    else:
        env = gym.make(cfg['env_name'])
    env.name = None

    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    
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
                  redo=cfg['redo'], threshold=cfg['threshold'], reset_period=cfg['reset_period']
                  )

    to_log = cfg['to_log']
    agent = Agent(pol, learner, device=device, to_log_features=(len(to_log) > 0))

    if os.path.exists(cfg['ckpt_path']):
        start_step, agent.learner = load_checkpoint(cfg, device, agent.learner)
    else:
        start_step = 0
    
    if os.path.exists(cfg['log_path']):
        data_dict = load_data(cfg)
        num_updates = data_dict['num_updates']
        previous_change_time = data_dict['previous_change_time']
        for k, v in data_dict.items():
            if k == 'margin_data':
                continue
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
        loaded_margin_data = data_dict.get('margin_data', None)
        if loaded_margin_data is None or 'predictive' not in loaded_margin_data:
            margin_data = {
                'predictive': {
                    'dead_counts': [], 'at_risk_counts': [],
                    'always_on_counts': [], 'mean_margins': [],
                    'timestamps': [], 'update_numbers': []
                },
                'actual': {
                    'dead_counts': [], 'at_risk_counts': [],
                    'always_on_counts': [], 'mean_margins': [],
                    'timestamps': [], 'update_numbers': []
                }
            }
        else:
            margin_data = loaded_margin_data
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
        margin_data = {
            'predictive': {
                'dead_counts': [], 'at_risk_counts': [],
                'always_on_counts': [], 'mean_margins': [],
                'timestamps': [], 'update_numbers': []
            },
            'actual': {
                'dead_counts': [], 'at_risk_counts': [],
                'always_on_counts': [], 'mean_margins': [],
                'timestamps': [], 'update_numbers': []
            }
        }

    window_size = cfg['bs']
    margin_log_interval = cfg['margin_log_interval']

    window_obs_buffer = torch.zeros(size=(window_size, o_dim), device=device)
    window_obs_idx = 0

    update_count = 0
    last_margin_update = -margin_log_interval

    saved_policy_state = None

    ret = 0
    epi_steps = 0
    o = env.reset()
    print('start_step:', start_step)
    print(f'Predictive margin analysis: window_size={window_size}, log_interval={margin_log_interval} updates')

    dormant_interval = cfg['dormant_log_interval']
    stable_interval = cfg['stable_rank_interval']
    print(f'Dormant/weight logging: every {dormant_interval} steps, stable rank: every {stable_interval} steps')

    wandb_log_interval = dormant_interval

    for step in range(start_step, n_steps):
        a, logp, dist, new_features = agent.get_action(o)
        op, r, done, infos = env.step(a)
        epi_steps += 1
        op_ = op
        val_logs = agent.log_update(o, a, r, op_, logp, dist, done)
        with torch.no_grad():
            if 'weight_change' in to_log and 'weight_change' in val_logs.keys(): weight_change.append(val_logs['weight_change'])
            if 'mu' in to_log: mu[step] = a

            window_obs_buffer[window_obs_idx] = torch.tensor(o, device=device)
            window_obs_idx += 1

            if window_obs_idx == window_size:
                update_count += 1

                if (update_count - last_margin_update) >= margin_log_interval:

                    if saved_policy_state is not None:
                        current_state = copy.deepcopy(pol.mean_net.state_dict())
                        pol.mean_net.load_state_dict(saved_policy_state)

                        predictive_results = compute_margins_per_layer(pol.mean_net, window_obs_buffer)

                        pol.mean_net.load_state_dict(current_state)

                        margin_data['predictive']['timestamps'].append(step)
                        margin_data['predictive']['update_numbers'].append(update_count)
                        margin_data['predictive']['dead_counts'].append(predictive_results['dead_counts'])
                        margin_data['predictive']['at_risk_counts'].append(predictive_results['at_risk_counts'])
                        margin_data['predictive']['always_on_counts'].append(predictive_results['always_on_counts'])
                        pred_mean_margins = {l: m.mean().item() for l, m in predictive_results['mean_margins'].items()}
                        margin_data['predictive']['mean_margins'].append(pred_mean_margins)

                        log_margin_to_wandb(step, predictive_results, update_count, cfg, prefix='predictive_margin')

                    actual_results = compute_margins_per_layer(pol.mean_net, window_obs_buffer)

                    margin_data['actual']['timestamps'].append(step)
                    margin_data['actual']['update_numbers'].append(update_count)
                    margin_data['actual']['dead_counts'].append(actual_results['dead_counts'])
                    margin_data['actual']['at_risk_counts'].append(actual_results['at_risk_counts'])
                    margin_data['actual']['always_on_counts'].append(actual_results['always_on_counts'])
                    actual_mean_margins = {l: m.mean().item() for l, m in actual_results['mean_margins'].items()}
                    margin_data['actual']['mean_margins'].append(actual_mean_margins)

                    log_margin_to_wandb(step, actual_results, update_count, cfg, prefix='actual_margin')

                    if saved_policy_state is not None and update_count >= 2:
                        log_predictive_margin_comparison(step, predictive_results, actual_results, update_count, cfg)

                    last_margin_update = update_count

                    if update_count % 100 == 0:
                        total_neurons = sum(cfg['h_dim'])
                        actual_dead = sum(actual_results['dead_counts'].values())
                        actual_at_risk = sum(actual_results['at_risk_counts'].values())
                        print(f'[Update {update_count}] Step {step}: Actual Dead={actual_dead}/{total_neurons} '
                              f'({100*actual_dead/total_neurons:.1f}%), At-risk={actual_at_risk}')

                saved_policy_state = copy.deepcopy(pol.mean_net.state_dict())

                window_obs_idx = 0

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
            rets.append(ret)
            termination_steps.append(step)
            
            if WANDB_AVAILABLE and wandb.run is not None:
                episode_log = {
                    'episode/return': float(ret),
                    'episode/length': epi_steps,
                    'episode/count': len(rets),
                }
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
            o = env.reset()

        if step % wandb_log_interval == 0 and step > 0:
            log_to_wandb(step, rets, termination_steps, pol_features_activity,
                        stable_rank, pol_weights, val_weights, cfg, to_log)

        if step % (n_steps//100) == 0 or step == n_steps-1:
            save_checkpoint(cfg, step, agent.learner)
            save_data(cfg=cfg, rets=rets, termination_steps=termination_steps,
                      pol_features_activity=pol_features_activity, stable_rank=stable_rank, mu=mu, pol_weights=pol_weights,
                      val_weights=val_weights, weight_change=weight_change, friction=friction,
                      num_updates=num_updates, previous_change_time=previous_change_time,
                      margin_data=margin_data)
            
            if len(rets) > 0:
                progress = (step / n_steps) * 100
                avg_ret = np.mean(rets[-100:]) if len(rets) >= 100 else np.mean(rets)
                print(f"Step {step}/{n_steps} ({progress:.1f}%) | Episodes: {len(rets)} | Avg Return (last 100): {avg_ret:.2f}")

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.run.summary['final_avg_return_100'] = np.mean(rets[-100:]) if len(rets) >= 100 else np.mean(rets)
        wandb.run.summary['final_avg_return_all'] = np.mean(rets)
        wandb.run.summary['total_episodes'] = len(rets)
        wandb.run.summary['total_steps'] = n_steps
        wandb.run.summary['total_ppo_updates'] = update_count
        if margin_data['actual']['dead_counts']:
            final_dead = sum(margin_data['actual']['dead_counts'][-1].values())
            final_at_risk = sum(margin_data['actual']['at_risk_counts'][-1].values())
            total_neurons = sum(cfg['h_dim'])
            wandb.run.summary['final_dead_neurons'] = final_dead
            wandb.run.summary['final_dead_pct'] = 100 * final_dead / total_neurons
            wandb.run.summary['final_at_risk_neurons'] = final_at_risk
            wandb.run.summary['final_at_risk_pct'] = 100 * final_at_risk / total_neurons
        wandb.finish()

    with open(cfg['done_path'], 'w') as f:
        f.write('All done!')
        print('The experiment finished successfully!')


if __name__ == "__main__":
    main()

