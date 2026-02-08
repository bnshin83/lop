import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from lop.algos.bp import Backprop
from lop.nets.conv_net import ConvNet
from lop.nets.conv_net2 import ConvNet2
from lop.algos.convCBP import ConvCBP
from torch.nn.functional import softmax
from lop.nets.linear import MyLinear
from lop.utils.miscellaneous import nll_accuracy as accuracy
from torch.utils.data import TensorDataset, DataLoader
from scipy.linalg import svd

train_images_per_class = 600
test_images_per_class = 100
images_per_class = train_images_per_class + test_images_per_class


# -------------------- For computing analysis of the network -------------------- #
@torch.no_grad()
def compute_average_weight_magnitude(net):
    """ Computes average magnitude of the weights in the network """
    num_weights = 0
    sum_weight_magnitude = torch.tensor(0.0)
    
    for p in net.parameters():
        num_weights += p.numel()
        sum_weight_magnitude += torch.sum(torch.abs(p))
    
    return sum_weight_magnitude.item() / num_weights


@torch.no_grad()
def compute_dormant_units_proportion(net, data_loader, dormant_unit_threshold=0.01):
    """
    Computes the proportion of dormant units in the network.
    For ConvNet/ConvNet2, we analyze the hidden layers.
    """
    device = next(net.parameters()).device
    net.eval()
    
    # Get a batch of data to analyze activations
    for batch in data_loader:
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch[0]
        x = x.to(device)
        
        # Use the network's forward method to get proper activations
        # We'll trace through the network layers properly
        activations = []
        
        # Use the same logic as in ConvNet.predict() to properly process layers
        # Conv1 -> ReLU -> Pool
        x1 = net.layers[0](x)  # conv1
        x1 = net.layers[1](x1)  # ReLU
        x1 = net.pool(x1)      # MaxPool
        activations.append(x1.detach())
        
        # Conv2 -> ReLU -> Pool
        x2 = net.layers[2](x1)  # conv2
        x2 = net.layers[3](x2)  # ReLU
        x2 = net.pool(x2)      # MaxPool
        activations.append(x2.detach())
        
        # Conv3 -> ReLU -> Pool
        x3 = net.layers[4](x2)  # conv3
        x3 = net.layers[5](x3)  # ReLU
        x3 = net.pool(x3)      # MaxPool
        activations.append(x3.detach())
        
        # Flatten for FC layers
        x3_flat = x3.view(-1, net.num_conv_outputs)
        
        # FC1 -> ReLU
        x4 = net.layers[6](x3_flat)  # fc1
        x4 = net.layers[7](x4)       # ReLU
        activations.append(x4.detach())
        
        # FC2 -> ReLU
        x5 = net.layers[8](x4)   # fc2
        x5 = net.layers[9](x5)   # ReLU
        activations.append(x5.detach())
        
        # Only process first batch
        break
    
    # Calculate dormant neurons for each layer
    total_units = 0
    dormant_units = 0
    
    for i, act in enumerate(activations):
        # For conv layers: (batch, channels, height, width)
        # For linear layers: (batch, features)
        if act.dim() == 4:  # Convolutional layer
            # Average across batch, height, width
            mean_activation = act.abs().mean(dim=(0, 2, 3))
        else:  # Linear layer
            # Average across batch
            mean_activation = act.abs().mean(dim=0)
        
        # Count dormant units
        is_dormant = mean_activation < dormant_unit_threshold
        dormant_units += is_dormant.sum().item()
        total_units += mean_activation.numel()
    
    dormant_proportion = dormant_units / total_units if total_units > 0 else 0.0
    
    # Return last hidden layer features for rank analysis (FC2 output)
    last_features = activations[-1]  # This is already flattened (FC2 output)
    
    net.train()
    return dormant_proportion, last_features


def compute_effective_rank(singular_values):
    """ Computes the effective rank from singular values """
    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)
    return np.exp(entropy)


def compute_stable_rank(singular_values):
    """ Computes the stable rank from singular values """
    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
    return np.sum(cumsum_sorted_singular_values < 0.99) + 1


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

    net.train()
    return curvature_sum.item() / total_samples, eig_sum.item() / total_samples


def load_imagenet(classes=[]):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = 'data/classes/' + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([idx] * train_images_per_class))
        y_test.append(np.array([idx] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    y_train = torch.from_numpy(np.concatenate(y_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test


def save_data(data, data_file):
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)


def repeat_expr(params: {}):
    agent_type = params['agent']
    num_tasks = params['num_tasks']
    num_showings = params['num_showings']

    step_size = params['step_size']
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'contribution'
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev='cpu'
    num_classes = 10
    total_classes = 1000
    new_heads = True
    mini_batch_size = 100
    perturb_scale = 0
    momentum = 0
    net_type = 1
    if 'replacement_rate' in params.keys(): replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys(): decay_rate = params['decay_rate']
    if 'util_type' in params.keys(): util_type = params['util_type']
    if 'maturity_threshold' in params.keys():   maturity_threshold = params['maturity_threshold']
    if 'weight_decay' in params.keys(): weight_decay = params['weight_decay']
    if 'use_gpu' in params.keys():
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'num_classes' in params.keys():  num_classes = params['num_classes']
    if 'new_heads' in params.keys():    new_heads = params['new_heads']
    if 'mini_batch_size' in params.keys():  mini_batch_size = params['mini_batch_size']
    if 'perturb_scale' in params.keys():    perturb_scale = params['perturb_scale']
    if 'momentum' in params.keys(): momentum = params['momentum']
    if 'net_type' in params.keys(): net_type = params['net_type']
    num_epochs = num_showings

    classes_per_task = num_classes
    net = ConvNet()
    if net_type == 2:
        net = ConvNet2(replacement_rate=replacement_rate, maturity_threshold=maturity_threshold)
    if agent_type == 'linear':
        net = MyLinear( 
            input_size=3072, num_outputs=classes_per_task
        )

    if agent_type in ['bp', 'linear']:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            to_perturb=(perturb_scale != 0),
            perturb_scale=perturb_scale,
            device=dev,
            momentum=momentum,
        )
    elif agent_type == 'cbp':
        learner = ConvCBP(
            net=net,
            step_size=step_size,
            momentum=momentum,
            loss='nll',
            weight_decay=weight_decay,
            opt=opt,
            init='default',
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            util_type=util_type,
            device=dev,
            maturity_threshold=maturity_threshold,
        )

    with open('class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[int([params['run_idx']][0])]
    num_class_repetitions_required = int(num_classes * num_tasks / total_classes) + 1
    class_order = np.concatenate([class_order]*num_class_repetitions_required)
    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    examples_per_epoch = train_images_per_class * classes_per_task

    train_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    test_accuracies = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    curvatures = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    weight_magnitudes = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    dormant_proportions = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    effective_ranks = torch.zeros((num_tasks, num_epochs), dtype=torch.float)
    stable_ranks = torch.zeros((num_tasks, num_epochs), dtype=torch.float)

    x_train, x_test, y_train, y_test = None, None, None, None
    for task_idx in range(num_tasks):
        del x_train, x_test, y_train, y_test
        x_train, y_train, x_test, y_test = load_imagenet(class_order[task_idx*classes_per_task:(task_idx+1)*classes_per_task])
        x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
        if agent_type == 'linear':
            x_train, x_test = x_train.flatten(1), x_test.flatten(1)
        if use_gpu == 1:
            x_train, x_test, y_train, y_test = x_train.to(dev), x_test.to(dev), y_train.to(dev), y_test.to(dev)
        if new_heads:
            net.layers[-1].weight.data *= 0
            net.layers[-1].bias.data *= 0

        for epoch_idx in tqdm(range(num_epochs)):
            example_order = np.random.permutation(train_images_per_class * classes_per_task)
            x_train = x_train[example_order]
            y_train = y_train[example_order]
            new_train_accuracies = torch.zeros((int(examples_per_epoch/mini_batch_size),), dtype=torch.float)
            epoch_iter = 0
            for start_idx in range(0, examples_per_epoch, mini_batch_size):
                batch_x = x_train[start_idx: start_idx+mini_batch_size]
                batch_y = y_train[start_idx: start_idx+mini_batch_size]

                # train the network
                loss, network_output = learner.learn(x=batch_x, target=batch_y)
                with torch.no_grad():
                    new_train_accuracies[epoch_iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()
                    epoch_iter += 1

            # log accuracy
            with torch.no_grad():
                train_accuracies[task_idx][epoch_idx] = new_train_accuracies.mean()
                new_test_accuracies = torch.zeros((int(x_test.shape[0] / mini_batch_size),), dtype=torch.float)
                test_epoch_iter = 0
                for start_idx in range(0, x_test.shape[0], mini_batch_size):
                    test_batch_x = x_test[start_idx: start_idx + mini_batch_size]
                    test_batch_y = y_test[start_idx: start_idx + mini_batch_size]

                    network_output = net.forward(x=test_batch_x)
                    new_test_accuracies[test_epoch_iter] = accuracy(softmax(network_output, dim=1), test_batch_y)
                    test_epoch_iter += 1

                test_accuracies[task_idx][epoch_idx] = new_test_accuracies.mean()
                print('accuracy for task', task_idx, 'in epoch', epoch_idx, ': train, ',
                      train_accuracies[task_idx][epoch_idx], ', test,', test_accuracies[task_idx][epoch_idx])

            # Calculate all metrics after each epoch
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False)
            
            # Curvature
            curvature, _ = compute_curvature_for_dataset(net, train_loader)
            curvatures[task_idx][epoch_idx] = curvature
            
            # Weight magnitude
            weight_mag = compute_average_weight_magnitude(net)
            weight_magnitudes[task_idx][epoch_idx] = weight_mag
            
            # Dormant units and get features for rank analysis
            dormant_prop, last_features = compute_dormant_units_proportion(net, train_loader)
            dormant_proportions[task_idx][epoch_idx] = dormant_prop
            
            # Compute ranks from SVD of last layer features
            if last_features is not None and last_features.size(0) > 1:
                try:
                    # Move to CPU for SVD computation
                    features_cpu = last_features.cpu().numpy()
                    singular_values = svd(features_cpu, compute_uv=False)
                    effective_ranks[task_idx][epoch_idx] = compute_effective_rank(singular_values)
                    stable_ranks[task_idx][epoch_idx] = compute_stable_rank(singular_values)
                except:
                    # If SVD fails, use default values
                    effective_ranks[task_idx][epoch_idx] = 0.0
                    stable_ranks[task_idx][epoch_idx] = 0.0
            
            print(f'Metrics for task {task_idx}, epoch {epoch_idx}:')
            print(f'  Curvature: {curvature:.6f}')
            print(f'  Weight magnitude: {weight_mag:.6f}')
            print(f'  Dormant proportion: {dormant_prop:.4f}')
            print(f'  Effective rank: {effective_ranks[task_idx][epoch_idx]:.2f}')
            print(f'  Stable rank: {stable_ranks[task_idx][epoch_idx]:.2f}')

        if task_idx % save_after_every_n_tasks == 0:
            save_data(data={
                'train_accuracies': train_accuracies.cpu(),
                'test_accuracies': test_accuracies.cpu(),
                'curvatures': curvatures.cpu(),
                'weight_magnitudes': weight_magnitudes.cpu(),
                'dormant_proportions': dormant_proportions.cpu(),
                'effective_ranks': effective_ranks.cpu(),
                'stable_ranks': stable_ranks.cpu(),
            }, data_file=params['data_file'])
    
    save_data(data={
        'train_accuracies': train_accuracies.cpu(),
        'test_accuracies': test_accuracies.cpu(),
        'curvatures': curvatures.cpu(),
        'weight_magnitudes': weight_magnitudes.cpu(),
        'dormant_proportions': dormant_proportions.cpu(),
        'effective_ranks': effective_ranks.cpu(),
        'stable_ranks': stable_ranks.cpu(),
    }, data_file=params['data_file'])


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

    repeat_expr(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
