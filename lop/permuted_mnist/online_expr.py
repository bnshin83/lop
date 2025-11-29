import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.nets.linear import MyLinear
from torch.nn.functional import softmax
from lop.nets.deep_ffnn import DeepFFNN
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries


def online_expr(params: {}):
    agent_type = params['agent']
    num_tasks = 200
    if 'num_tasks' in params.keys():
        num_tasks = params['num_tasks']
    if 'num_examples' in params.keys() and "change_after" in params.keys():
        num_tasks = int(params["num_examples"]/params["change_after"])

    step_size = params['step_size']
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev = 'cpu'
    to_log = False
    num_features = 2000
    change_after = 10 * 6000
    to_perturb = False
    perturb_scale = 0.1
    num_hidden_layers = 1
    mini_batch_size = 1
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'adaptable_contribution'

    if 'to_log' in params.keys():
        to_log = params['to_log']
    if 'weight_decay' in params.keys():
        weight_decay = params['weight_decay']
    if 'num_features' in params.keys():
        num_features = params['num_features']
    if 'change_after' in params.keys():
        change_after = params['change_after']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'to_perturb' in params.keys():
        to_perturb = params['to_perturb']
    if 'perturb_scale' in params.keys():
        perturb_scale = params['perturb_scale']
    if 'num_hidden_layers' in params.keys():
        num_hidden_layers = params['num_hidden_layers']
    if 'mini_batch_size' in params.keys():
        mini_batch_size = params['mini_batch_size']
    if 'replacement_rate' in params.keys():
        replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys():
        decay_rate = params['decay_rate']
    if 'maturity_threshold' in params.keys():
        maturity_threshold = params['mt']
    if 'util_type' in params.keys():
        util_type = params['util_type']

    only_curvature = False
    if 'only_curvature' in params.keys():
        only_curvature = params['only_curvature']

    classes_per_task = 10
    images_per_class = 6000
    input_size = 784
    num_hidden_layers = num_hidden_layers
    net = DeepFFNN(input_size=input_size, num_features=num_features, num_outputs=classes_per_task,
                   num_hidden_layers=num_hidden_layers)

    if agent_type == 'linear':
        net = MyLinear(
            input_size=input_size, num_outputs=classes_per_task
        )
        net.layers_to_log = []

    if agent_type in ['bp', 'linear', "l2"]:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
        )
    elif agent_type in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            decay_rate=decay_rate,
            util_type=util_type,
            accumulate=True,
            device=dev,
        )

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after)
    total_iters = int(total_examples/mini_batch_size)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    accuracies = torch.zeros(total_iters, dtype=torch.float)
    weight_mag_sum = torch.zeros((total_iters, num_hidden_layers+1), dtype=torch.float)

    rank_measure_period = 60000
    effective_ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    approximate_ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    approximate_ranks_abs = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    ranks = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    dead_neurons = torch.zeros((int(total_examples/rank_measure_period), num_hidden_layers), dtype=torch.float)
    curvatures = torch.zeros(num_tasks, dtype=torch.float)

    iter = 0
    with open('data/mnist_', 'rb+') as f:
        x, y, _, _ = pickle.load(f)
        if use_gpu == 1:
            x = x.to(dev)
            y = y.to(dev)

    for task_idx in (range(num_tasks)):
        new_iter_start = iter
        # 1. Create new task (permute pixels and shuffle data)
        pixel_permutation = np.random.permutation(input_size)
        x = x[:, pixel_permutation]
        data_permutation = np.random.permutation(examples_per_task)
        x, y = x[data_permutation], y[data_permutation]

        # 2. Measure metrics BEFORE training on this task
        if agent_type != 'linear':
            with torch.no_grad():
                if not only_curvature:
                    new_idx = int(iter / rank_measure_period)
                    m = net.predict(x[:2000])[1]
                    for rep_layer_idx in range(num_hidden_layers):
                        ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                        approximate_ranks[new_idx][rep_layer_idx], approximate_ranks_abs[new_idx][rep_layer_idx] = \
                            compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)
                        dead_neurons[new_idx][rep_layer_idx] = (m[rep_layer_idx].abs().sum(dim=0) == 0).sum()
                    print('approximate rank: ', approximate_ranks[new_idx], ', dead neurons: ', dead_neurons[new_idx])

                # Calculate curvature
                dataset = (x[:2000], y[:2000])
                loader = DataLoader(dataset, batch_size=256)
                curv, _ = compute_curvature_for_dataset(net, loader)
                curvatures[task_idx] = curv

        for start_idx in tqdm(range(0, change_after, mini_batch_size), desc=f'Task {task_idx + 1}/{num_tasks}'):
            start_idx = start_idx % examples_per_task
            batch_x = x[start_idx: start_idx+mini_batch_size]
            batch_y = y[start_idx: start_idx+mini_batch_size]

            # 3. NOW train the network 
            loss, network_output = learner.learn(x=batch_x, target=batch_y)

            if not only_curvature:
                if to_log and agent_type != 'linear':
                    for idx, layer_idx in enumerate(learner.net.layers_to_log):
                        weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()
                # log accuracy
                with torch.no_grad():
                    accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
            iter += 1

        if not only_curvature:
            print('recent accuracy', accuracies[new_iter_start:iter - 1].mean())
        if task_idx % save_after_every_n_tasks == 0:
            data = {
                'curvatures': curvatures.cpu(),
            }
            if not only_curvature:
                data.update({
                    'accuracies': accuracies.cpu(),
                    'weight_mag_sum': weight_mag_sum.cpu(),
                    'ranks': ranks.cpu(),
                    'effective_ranks': effective_ranks.cpu(),
                    'approximate_ranks': approximate_ranks.cpu(),
                    'abs_approximate_ranks': approximate_ranks_abs.cpu(),
                    'dead_neurons': dead_neurons.cpu(),
                })
            save_data(file=params['data_file'], data=data)

    data = {
        'curvatures': curvatures.cpu(),
    }
    if not only_curvature:
        data.update({
            'accuracies': accuracies.cpu(),
            'weight_mag_sum': weight_mag_sum.cpu(),
            'ranks': ranks.cpu(),
            'effective_ranks': effective_ranks.cpu(),
            'approximate_ranks': approximate_ranks.cpu(),
            'abs_approximate_ranks': approximate_ranks_abs.cpu(),
            'dead_neurons': dead_neurons.cpu(),
        })
    save_data(file=params['data_file'], data=data)


def save_data(file, data):
    with open(file, 'wb+') as f:
        pickle.dump(data, f)


def compute_curvature_for_dataset(net, data_loader, h=1e-3, niter=10, temp=1.0):
    """
    Computes curvature metrics for the dataset, adapted from calc_curv_fz_models.
    
    Args:
        net: The neural network.
        data_loader: DataLoader for the data.
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

    net.train()
    return curvature_sum.item() / total_samples, eig_sum.item() / total_samples


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the file containing the parameters for the experiment",
                        type=str, default='temp_cfg/0.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    online_expr(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
