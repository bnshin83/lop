"""
Script for computing the effective rank, stable rank, number of dormant neurons, and average weight magnitude of the
models trained during the incremental cifar experiment.
"""

# built-in libraries
import time
import os
import argparse

# third party libraries
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from scipy.linalg import svd

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize

from lop.nets.torchvision_modified_resnet import ResNet, build_resnet18


# -------------------- For loading data and network parameters -------------------- #
def load_model_parameters(parameter_dir_path: str, index: int, epoch_number:int):
    """
    Loads the model parameters stored in parameter_dir_path corresponding to the index and epoch number
    return: torch module state dictionary
    """

    model_parameters_file_name = "index-{0}_epoch-{1}.pt".format(index, epoch_number)
    model_parameters_file_path = os.path.join(parameter_dir_path, model_parameters_file_name)

    if not os.path.isfile(model_parameters_file_path):
        error_message = "Couldn't find model parameters for index {0} and epoch number {1}.".format(index, epoch_number)
        raise ValueError(error_message)
    # Boonam: Added map_location to load the model parameters on CPU
    return torch.load(model_parameters_file_path, map_location=torch.device('cpu'))


def load_classes(classes_dir_path: str, index: int):
    """
    Loads the list of ordered classes used for partitioning the data during the experiment
    return: list
    """

    classes_file_name = "index-{0}.npy".format(index)
    classes_file_path = os.path.join(classes_dir_path, classes_file_name)

    if not os.path.isfile(classes_file_path):
        error_message = "Couldn't find list of classes for index {0}.".format(index)
        raise ValueError(error_message)

    return np.load(classes_file_path)


def load_cifar_data(data_path: str, train: bool = True) -> (CifarDataSet, DataLoader):
    """
    Loads the cifar 100 data set with normalization
    :param data_path: path to the directory containing the data set
    :param train: bool that indicates whether to load the train or test data
    :return: torch DataLoader object
    """
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=train,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=True)

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
    ]

    cifar_data.set_transformation(transforms.Compose(transformations))

    num_workers = 12
    batch_size = 1000
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return cifar_data, dataloader


# -------------------- For computing analysis of the network -------------------- #
@torch.no_grad()
def compute_average_weight_magnitude(net: ResNet):
    """ Computes average magnitude of the weights in the network """

    num_weights = 0
    sum_weight_magnitude = torch.tensor(0.0, device=net.fc.weight.device)

    for p in net.parameters():
        num_weights += p.numel()
        sum_weight_magnitude += torch.sum(torch.abs(p))

    return sum_weight_magnitude.cpu().item() / num_weights

# Boonam: for debugging
@torch.no_grad()
def compute_dormant_units_proportion(net: ResNet, cifar_data_loader: DataLoader, dormant_unit_threshold: float = 0.01):
    """
    Computes the proportion of dormant units in a ResNet. It also returns the features of the last layer for the first
    1000 samples
    """
    # Boonam: Updated to use the new get_features function for debugging
    device = next(net.parameters()).device
    net.eval()
    features_per_layer = []
    last_layer_activations = None

    print(f"    Extracting features from first batch...")
    with torch.no_grad():
        # Process only one batch as in the original implementation
        for batch in cifar_data_loader:
            # Handle both tuple format (x, _) and dictionary format (batch['image'])
            if isinstance(batch, dict):
                x = batch['image'].to(device)
            else:
                x, _ = batch
                x = x.to(device)
                
            print(f"    Input shape: {x.shape}")
            # Get features from the network
            try:
                temp_features = []
                net.forward(x, temp_features) # Boonam: modified forward by torchvision_modified_resnet.py
                features_per_layer = temp_features
                last_layer_activations = temp_features[-1].detach()  # Keep on GPU
                print(f"    Successfully extracted features from {len(features_per_layer)} layers")
            except Exception as e:
                print(f"    Error extracting features: {e}")
                # Last resort, try regular forward pass
                outputs = net(x)
                print(f"    Regular forward pass output shape: {outputs.shape}")
                return 0.0, np.zeros((x.shape[0], 512))  # Return placeholder data on error
            
            # Only process the first batch
            break

    print(f"    Computing dormant units across all layers...")
    # Calculate dormant neurons for each layer as in original implementation
    dead_neurons = torch.zeros(len(features_per_layer), dtype=torch.float32)
    for layer_idx in range(len(features_per_layer) - 1):
        # For convolutional layers, we need to average across batch, height, and width
        dead_neurons[layer_idx] = ((features_per_layer[layer_idx] != 0).float().mean(dim=(0, 2, 3)) < dormant_unit_threshold).sum()
    
    # For the final layer (which is likely fully connected)
    dead_neurons[-1] = ((features_per_layer[-1] != 0).float().mean(dim=0) < dormant_unit_threshold).sum()
    
    # Calculate the total proportion
    number_of_features = torch.sum(torch.tensor([layer_feats.shape[1] for layer_feats in features_per_layer])).item()
    dormant_prop = dead_neurons.sum().item() / number_of_features
    
    print(f"    Total dormant units proportion: {dormant_prop:.4f}")
    return dormant_prop, last_layer_activations


def compute_effective_rank(singular_values: np.ndarray):
    """ Computes the effective rank of the representation layer """

    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)

    return np.e ** entropy


def compute_stable_rank(singular_values: np.ndarray):
    """ Computes the stable rank of the representation layer """
    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
    return np.sum(cumsum_sorted_singular_values < 0.99) + 1


def compute_effective_rank_torch(singular_values: torch.Tensor) -> torch.Tensor:
    """ Computes the effective rank from singular values using PyTorch. """
    # Ensure singular_values are on the same device for calculations
    device = singular_values.device
    # Normalize singular values
    norm_sv = singular_values / torch.sum(torch.abs(singular_values))
    entropy = torch.tensor(0.0, device=device)
    # Filter out zero or negative probabilities for log
    positive_norm_sv = norm_sv[norm_sv > 0.0]
    if positive_norm_sv.numel() > 0: # Check if any positive values exist
        entropy = -torch.sum(positive_norm_sv * torch.log(positive_norm_sv))
    
    return torch.exp(entropy)

def compute_stable_rank_torch(singular_values: torch.Tensor) -> torch.Tensor:
    """ Computes the stable rank from singular values using PyTorch. """
    # Ensure singular_values are on the same device for calculations
    device = singular_values.device
    # Sort singular values in descending order
    sorted_singular_values, _ = torch.sort(singular_values, descending=True)
    # Compute cumulative sum and normalize
    cumsum_normalized_singular_values = torch.cumsum(sorted_singular_values, dim=0) / torch.sum(sorted_singular_values)
    # Count how many are less than 0.99
    stable_rank_val = torch.sum(cumsum_normalized_singular_values < 0.99).float() + 1.0
    return stable_rank_val


@torch.no_grad()
def compute_last_task_accuracy_per_class_in_order(net: torch.nn.Module, ordered_classes: np.ndarray,
                                                  test_data: DataLoader, experiment_index: int):
    """
    Computes the accuracy of each class in the order they were presented
    :param net: resnet with the parameters stored at the end of the experiment
    :param ordered_classes: numpy array with the cifar 100 classes in the order they were presented
    :param test_data: cifar100 test data
    :return: numpy array
    """

    ordered_classes = np.int32(ordered_classes)
    # Boonam: Added this to make sure the input tensors are properly moved to the same device as the model's device.
    # device = net.fc.weight.device
    device = next(net.parameters()).device
    num_classes = 100
    num_examples_per_class = 100

    class_correct = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for i, sample in enumerate(test_data):
        image = sample["image"].to(device)
        labels = sample["label"].to(device)
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)    # Get the class with the highest score
        _, labels = torch.max(labels, 1)        # Get the class with the highest score

        # Update the counts for each class
        for i, class_label in enumerate(ordered_classes):
            class_correct[i] += (predicted == labels).masked_select(labels == class_label).sum().item()

    return class_correct.cpu().numpy() / num_examples_per_class


# -------------------- For storing the results of the analysis -------------------- #
def store_analysis_results(weight_magnitude_results: np.ndarray,
                           dormant_units_results: (np.ndarray, np.ndarray),
                           effective_rank_results: (np.ndarray, np.ndarray),
                           stable_rank_results: (np.ndarray, np.ndarray),
                           accuracy_per_class_in_order: np.ndarray,
                           results_dir: str, experiment_index: int):
    """
    Stores the results of the post run analysis
    :param weight_magnitude_results: np array containing the output of compute_average_weight_magnitude
    :param dormant_units_results: tuple containing the results of the dormant unit analysis for the previous tasks and
                                  the next task for each different task
    :param effective_rank_results: tuple containing the results of the effective rank analysis for the previous tasks
                                   and the next task for each different task
    :param stable_rank_results: tuple containing the results of the stable rank analysis for the previous tasks and the
                                next task for each different task
    :param accuracy_per_class_in_order: np array containing the accuracy of the final model for each class in the order
                                        they were presented
    :param results_dir: path to the results directory
    :param experiment_index: experiment index
    """

    index_file_name = "index-{0}.npy".format(experiment_index)
    result_dir_names_and_arrays = [
        ("weight_magnitude_analysis", weight_magnitude_results),
        ("previous_tasks_dormant_units_analysis", dormant_units_results[0]),
        ("next_task_dormant_units_analysis", dormant_units_results[1]),
        ("previous_tasks_effective_rank_analysis", effective_rank_results[0]),
        ("next_task_effective_rank_analysis", effective_rank_results[1]),
        ("previous_tasks_stable_rank_analysis", stable_rank_results[0]),
        ("next_task_stable_rank_analysis", stable_rank_results[1]),
        ("accuracy_per_class_in_order", accuracy_per_class_in_order)
    ]

    # store results in the corresponding dir
    for results_name, results_array in result_dir_names_and_arrays:
        temp_results_dir = os.path.join(results_dir, results_name)
        os.makedirs(temp_results_dir, exist_ok=True)
        np.save(os.path.join(temp_results_dir, index_file_name), results_array)


def analyze_results(results_dir: str, data_path: str, dormant_unit_threshold: float = 0.01):
    """
    Analyses the parameters of a run and creates files with the results of the analysis
    :param results_dir: path to directory containing the results for a parameter combination
    :param data_path: path to the cifar100 data set
    :param dormant_unit_threshold: hidden units whose activation fall bellow this threshold are considered dormant
    """

    parameter_dir_path = os.path.join(results_dir, "model_parameters")
    experiment_indices_file_path = os.path.join(results_dir, "experiment_indices.npy")
    class_order_dir_path = os.path.join(results_dir, "class_order")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Boonam: Check which epoch files actually exist
    available_epochs = []
    max_epoch = 4000  # Your planned final epoch
    step_size = 200   # From your original script
    
    # Find the available epochs
    for epoch in range(0, max_epoch + 1, step_size):
        if os.path.exists(os.path.join(parameter_dir_path, f"index-0_epoch-{epoch}.pt")):
            available_epochs.append(epoch)
    
    if not available_epochs:
        raise ValueError("No model parameter files found.")
    
    print(f"Found model parameters for epochs: {available_epochs}")
    
    # Boonam: Updated, Use the detected epochs instead of the hardcoded range
    number_of_epochs = np.array(available_epochs)
    classes_per_task = 5                    # by design each task increases the data set by 5 classes
    last_epoch = available_epochs[-1]       # Use the last available epoch
    
    try:
        experiment_indices = np.load(experiment_indices_file_path)
        if experiment_indices.ndim == 0:
            # Convert scalar to a 1-element array
            experiment_indices = np.array([experiment_indices.item()])
    except:
        # If file doesn't exist or has issues, default to analyzing experiment index 0
        experiment_indices = np.array([0])

    net = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    net.to(device)
    cifar_data, cifar_data_loader = load_cifar_data(data_path, train=True)
    test_data, test_data_loader = load_cifar_data(data_path, train=False)

    for exp_index in tqdm(experiment_indices):
        # Create class order file if it doesn't exist
        class_order_file = os.path.join(class_order_dir_path, f"index-{exp_index}.npy")
        if not os.path.exists(class_order_file):
            print(f"Creating temporary class order for index {exp_index} for interim analysis")
            # Create a sequential ordering of classes
            ordered_classes = np.arange(100, dtype=np.int32)
            # Save it for later reference
            np.save(class_order_file, ordered_classes)
        else:
            ordered_classes = load_classes(class_order_dir_path, index=exp_index)

        # Skip analysis if we only have one epoch (need at least two for before/after comparison)
        if len(number_of_epochs) < 2:
            print("Need at least two checkpoints for comparative analysis. Skipping.")
            continue

        average_weight_magnitude_per_epoch = np.zeros(len(number_of_epochs) - 1, dtype=np.float32)
        dormant_units_prop_before = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        dormant_units_prop_after = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)

        # Boonam: Updated to use the new get_features function for debugging
        for i, epoch_number in enumerate(number_of_epochs[:-1]):
            print(f"Processing epoch {epoch_number} ({i+1}/{len(number_of_epochs)-1})...")
            
            # get model parameters from before training on the task
            print(f"  Loading model parameters for epoch {epoch_number}...")
            model_parameters = load_model_parameters(parameter_dir_path, index=exp_index, epoch_number=epoch_number)
            net.load_state_dict(model_parameters)

            # compute average weight magnitude
            print(f"  Computing average weight magnitude...")
            average_weight_magnitude_per_epoch[i] = compute_average_weight_magnitude(net)

            # compute summaries for next task
            print(f"  Analyzing next task data (classes {i*classes_per_task}-{(i+1)*classes_per_task-1})...")
            current_classes = ordered_classes[(i * classes_per_task):((i + 1) * classes_per_task)]
            print(f"  Selecting new partition for CIFAR data...")
            cifar_data.select_new_partition(current_classes)

            print(f"  Computing dormant units proportion...")
            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)
            dormant_units_prop_after[i] = prop_dormant
            
            print(f"  Computing SVD for {last_layer_features.shape} matrix...")
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            print(f"  Computing effective and stable rank...")
            effective_rank_after[i] = compute_effective_rank(singular_values)
            stable_rank_after[i] = compute_stable_rank(singular_values)

            # compute summaries from data from previous tasks
            if i == 0: 
                print(f"  Skipping previous task analysis for first task")
                continue
                
            print(f"  Analyzing previous tasks data (classes 0-{i*classes_per_task-1})...")
            current_classes = ordered_classes[:(i * classes_per_task)]
            print(f"  Selecting previous partition for CIFAR data...")
            cifar_data.select_new_partition(current_classes)
            
            print(f"  Computing dormant units proportion for previous tasks...")
            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)

            dormant_units_prop_before[i] = prop_dormant
            
            print(f"  Computing SVD for previous tasks ({last_layer_features.shape} matrix)...")
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            print(f"  Computing effective and stable rank for previous tasks...")
            effective_rank_before[i] = compute_effective_rank(singular_values)
            stable_rank_before[i] = compute_stable_rank(singular_values)
            
            print(f"  Completed analysis for epoch {epoch_number}")

        net.load_state_dict(load_model_parameters(parameter_dir_path, exp_index, last_epoch))
        accuracy_per_class_in_order = compute_last_task_accuracy_per_class_in_order(net, ordered_classes,
                                                                                    test_data_loader, exp_index)

        store_analysis_results(weight_magnitude_results=average_weight_magnitude_per_epoch,
                               dormant_units_results=(dormant_units_prop_before, dormant_units_prop_after),
                               effective_rank_results=(effective_rank_before, effective_rank_after),
                               stable_rank_results=(stable_rank_before, stable_rank_after),
                               accuracy_per_class_in_order=accuracy_per_class_in_order,
                               results_dir=results_dir,
                               experiment_index=exp_index)


def parse_arguments() -> dict:

    file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', action="store", type=str,
                        default=os.path.join(file_path, "results", "base_deep_learning_system"),
                        help="Path to directory with the results of a parameter combination.")
    parser.add_argument('--data_path', action="store", type=str, default=os.path.join(file_path, "data"),
                        help="Path to directory with the CIFAR 100 data set.")
    parser.add_argument('--dormant_unit_threshold', action="store", type=float, default=0.01,
                        help="Units whose activations are less than this threshold are considered dormant.")

    args = parser.parse_args()
    return vars(args)


def main():

    analysis_arguments = parse_arguments()

    initial_time = time.perf_counter()
    analyze_results(results_dir=analysis_arguments["results_dir"],
                    data_path=analysis_arguments["data_path"],
                    dormant_unit_threshold=analysis_arguments["dormant_unit_threshold"])
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
