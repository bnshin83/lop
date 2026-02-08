import sys
import json
import pickle
import argparse
import torch
from tqdm import tqdm
import numpy as np
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


# -------------------- For calculating curvature (Hessian approximation) -------------------- #
def compute_curvature_for_dataset(net, data_loader, h=1e-3, niter=10, temp=1.0):
    """
    Computes curvature metrics for the dataset, adapted from calc_curv_fz_models.
    
    Args:
        net: The neural network (ResNet-18).
        data_loader: DataLoader for CIFAR-100 training data.
        h (float): Perturbation size for finite differences. Default is 1e-3.
        niter (int): Number of iterations for curvature estimation. Default is 10.
        temp (float): Temperature scaling for softmax. Default is 1.0.
    
    Returns:
        tuple: (curvature, eig_values), averaged over the dataset.
    """
    device = next(net.parameters()).device
    net.eval()
    curvature_sum = torch.zeros(1, device=device)
    eig_sum = torch.zeros(1, device=device)
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    for sample in tqdm(data_loader, desc="Curvature Batch", leave=False):
        # Handle both tuple format (x, _) and dictionary format (batch['image'])
        if isinstance(sample, dict):
            batch_data = sample['image'].to(device)
            batch_labels = sample['label'].to(device)
        else:
            batch_data, batch_labels = sample
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

        # Convert one-hot labels to class indices
        if len(batch_labels.shape) > 1 and batch_labels.shape[1] > 1:  # Check if one-hot
            batch_labels = torch.argmax(batch_labels, dim=1)  # Convert to indices
        
        num_samples = batch_data.shape[0]
        total_samples += num_samples

        regr = torch.zeros(num_samples, device=device)
        eigs = torch.zeros(num_samples, device=device)

        # Perturb each image in 10 random directions
        for _ in range(niter):
            v = torch.randint_like(batch_data, high=2, device=device) * 2 - 1  # Rademacher (±1)
            v = h * v  # Scale perturbation

            with torch.enable_grad():
                batch_data = batch_data.detach().requires_grad_(True)
                outputs_pos = net(batch_data + v)
                outputs_orig = net(batch_data)
                loss_pos = criterion(outputs_pos / temp, batch_labels)
                loss_orig = criterion(outputs_orig / temp, batch_labels)
                # Compute gradient changes per direction
                grad_diff = torch.autograd.grad(loss_pos - loss_orig, batch_data, create_graph=False)[0]

            regr += grad_diff.reshape(num_samples, -1).norm(dim=1)
            eigs += (v.reshape(num_samples, -1) * grad_diff.reshape(num_samples, -1)).sum(dim=1)
            net.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()

        curvature_sum += regr.sum() / niter
        eig_sum += eigs.sum() / niter

    return curvature_sum.item() / total_samples, eig_sum.item() / total_samples


def add_cfg_performance(cfg='', setting_idx=0, m=2*10*1000, num_runs=30, metric='accuracy'):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    for idx in range(num_runs):
        file = '../' + params['data_dir'] + str(setting_idx) + '/' + str(idx)
        with open(file, 'rb') as f:
            data = pickle.load(f)

        if metric == 'weight':
            num_weights = 9588000
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['weight_mag_sum'].sum(dim=1)/num_weights, m=m)))
        elif metric == 'dead_neurons':
            num_units = 3*2000
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['dead_neurons'].sum(dim=1)/num_units*100, m=m)))
        elif metric == 'effective_rank':
            rank_normlization = 3*2000/100
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['effective_ranks'].sum(dim=1)/rank_normlization, m=m)))
        elif metric == 'curvature':
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['curvatures'], m=m)))
        else:
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['accuracies'] * 100, m=m)))
    print(param_settings[setting_idx], setting_idx, np.array(per_param_setting_performance).mean())
    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
                            default='../cfg/bp/std_net.json')
    parser.add_argument('--metric', help="Specify the metric you want to plot, the options are: accuracy, weight,"
                                         " dead_neurons, effective_rank, and curvature", type=str, default='accuracy')

    args = parser.parse_args(arguments)
    cfg_file = args.cfg_file
    metric = args.metric

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)

    performances = []
    m = {'weight': 60*1000, 'accuracy': 60*1000, 'dead_neurons': 1, 'effective_rank': 1, 'curvature': 1}[metric]
    num_runs = params['num_runs']

    indices = [i for i in range(3)]
    for i in indices:
        performances.append(add_cfg_performance(cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs, metric=metric))

    yticks = {'weight': [0, 0.02, 0.04, 0.06, 0.08, 0.10], 'accuracy': [88, 90, 92, 94, 96],
              'dead_neurons': [0, 10, 20, 30], 'effective_rank': [0, 10, 20, 30, 40, 50],
              'curvature': [0, .001, .002, .003, .004, .005]}[metric]
    generate_online_performance_plot(
        performances=performances,
        colors=['C1', 'C3', 'C5', 'C2', 'C4', 'C6'],
        yticks=yticks,
        xticks=[0, 200*m, 400*m, 600*m, 800*m],
        xticks_labels=['0', '200', '400', '600', '800'],
        m=m,
        fontsize=18,
        labels=param_settings,
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

