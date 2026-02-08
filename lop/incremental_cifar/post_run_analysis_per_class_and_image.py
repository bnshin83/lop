"""
Script for computing per-class and per-image metrics for the models trained during the incremental cifar experiment.
Analyzes both class-level and individual image-level performance metrics.
"""

# built-in libraries
import json
import os
import time
from tqdm import tqdm
from pathlib import Path

# Define EXPECTED_NUM_FEATURES_IN_LIST for ResNet18
# ResNet18: conv1_relu(1) + layer1(2blocks*2appends=4) + layer2(4) + layer3(4) + layer4(4) + fc_input(1) = 18
EXPECTED_NUM_FEATURES_IN_LIST = 18

import argparse
import json

# third party libraries
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
# from scipy.linalg import svd # No longer needed here for compute_metrics_per_class

# Import torch rank functions from the modified analysis script
from lop.incremental_cifar.post_run_analysis_modified import (
    compute_effective_rank_torch,
    compute_stable_rank_torch,
    compute_effective_rank as compute_effective_rank_numpy, # Alias if numpy version is needed elsewhere in this file
    compute_stable_rank as compute_stable_rank_numpy   # Alias if numpy version is needed elsewhere in this file
)

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize

# Import ResNet with proper path
import sys
sys.path.append('/scratch/gautschi/shin283/loss-of-plasticity')  # Add the parent directory to path
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

    # Use map_location to handle models saved on CUDA devices when running on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.load(model_parameters_file_path, map_location=device)


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


# -------------------- Analysis Helper Functions -------------------- #
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


# -------------------- For computing per-class metrics -------------------- #
@torch.no_grad()
def compute_metrics_per_class(net: torch.nn.Module, ordered_classes: np.ndarray, data_loader: DataLoader, dormant_threshold: float = 0.01, debug_class_limit: int = 3, epoch_num_for_logging: int = -1):
    """
    Computes various metrics for each class, now with multi-layer dormant unit analysis.
    :param net: neural network with loaded parameters
    :param ordered_classes: numpy array of ordered class labels
    :param data_loader: cifar100 data loader
    :param dormant_threshold: threshold for considering a unit dormant
    :param debug_class_limit: if > 0, limits detailed console logging to this many classes.
    :param epoch_num_for_logging: epoch number, used for conditional diagnostic prints.
    :return: dictionary with per-class metrics
    """
    device = next(net.parameters()).device
    net.eval()
    
    # Ensure ordered_classes are integers for dictionary keys and indexing
    ordered_classes_int = [int(c) for c in ordered_classes]
    num_classes_to_process = len(ordered_classes_int)
    
    # Create a mapping from class ID to its position in the ordered_classes array
    # This replaces the 'i' from enumerate(ordered_classes) in the old code
    class_position_mapping = {cls_id: pos for pos, cls_id in enumerate(ordered_classes_int)}

    # Initialize accumulators
    class_correct = {cls_idx: 0 for cls_idx in ordered_classes_int}
    class_total = {cls_idx: 0 for cls_idx in ordered_classes_int}
    class_correct_top5 = {cls_idx: 0 for cls_idx in ordered_classes_int}
    class_confidence_predicted = {cls_idx: 0.0 for cls_idx in ordered_classes_int}
    class_confidence_true = {cls_idx: 0.0 for cls_idx in ordered_classes_int}
    
    # Storage for multi-layer features (dormancy) and FC output features (SVD)
    # Keys are integer class indices from ordered_classes_int
    class_features_multilayer = {cls_idx: {feat_idx: [] for feat_idx in range(EXPECTED_NUM_FEATURES_IN_LIST)} for cls_idx in ordered_classes_int}
    fc_output_features_for_svd = {cls_idx: [] for cls_idx in ordered_classes_int}
    
    # Max samples to collect per class for feature analysis (dormancy and SVD)
    MAX_SAMPLES_PER_CLASS_FOR_FEATURES = 10000

    print(f"Starting per-class metrics computation for {num_classes_to_process} classes on device: {device}...")

    for batch_idx, sample_batch in enumerate(tqdm(data_loader, desc="Processing batches")):
        # Assuming sample_batch is a dictionary from the original Dataloader structure
        inputs = sample_batch["image"].to(device)
        labels_raw = sample_batch["label"].to(device)
        
        # Handle both one-hot encoded labels and class index labels
        if labels_raw.dim() > 1 and labels_raw.shape[1] > 1:  # One-hot encoded format [batch_size, num_classes]
            print(f"Detected one-hot encoded labels with shape {labels_raw.shape}")
            # Convert from one-hot to class indices
            labels = torch.argmax(labels_raw, dim=1)
        else:  # Already in class index format [batch_size]
            labels = labels_raw

        batch_feature_list = [] # This list will be populated by the model's forward pass # Boonam: actually feature tensors list
        model_outputs = net(inputs, feature_list=batch_feature_list) 
        
        last_layer_logits = model_outputs.detach() # Final output of the network (logits)
        
        probabilities = torch.softmax(last_layer_logits, dim=1)
        batch_confidence_scores, batch_predicted_classes = torch.max(probabilities, dim=1)
        _, batch_top5_predicted_indices = torch.topk(probabilities, 5, dim=1)
        
        # Boonam: main loop over classes
        for cls_idx in ordered_classes_int:
            mask = (labels == cls_idx)
            num_class_samples_in_batch = mask.sum().item()

            if num_class_samples_in_batch > 0:
                # Accuracy and Confidence
                class_correct[cls_idx] += (batch_predicted_classes == labels).masked_select(mask).sum().item()
                class_total[cls_idx] += num_class_samples_in_batch
                
                for i_sample in range(inputs.shape[0]): # Iterate samples in batch
                    if mask[i_sample]: # If this sample belongs to cls_idx
                        # Get the true class label (ensuring it's a scalar)
                        true_label = labels[i_sample].item()
                        if true_label in batch_top5_predicted_indices[i_sample].tolist():
                            class_correct_top5[cls_idx] += 1
                        class_confidence_predicted[cls_idx] += batch_confidence_scores[i_sample].item()
                        class_confidence_true[cls_idx] += probabilities[i_sample, true_label].item()

                # Collect FC output features for SVD
                current_svd_samples = sum(f.shape[0] for f in fc_output_features_for_svd[cls_idx])
                if current_svd_samples < MAX_SAMPLES_PER_CLASS_FOR_FEATURES:
                    needed_svd = MAX_SAMPLES_PER_CLASS_FOR_FEATURES - current_svd_samples
                    samples_to_take_svd = min(num_class_samples_in_batch, needed_svd)
                    if samples_to_take_svd > 0:
                        # The last feature in batch_feature_list is assumed to be the input to the FC layer
                        if batch_feature_list: # If the list is not empty
                            actual_num_features = len(batch_feature_list)
                            # We expect the input to FC to be the last feature.
                            # If the number of features isn't what we hardcoded as EXPECTED, log it for debug classes/epochs but proceed.
                            if actual_num_features != EXPECTED_NUM_FEATURES_IN_LIST and \
                               (epoch_num_for_logging >= 0 and cls_idx in classes_to_log_details): 
                                print(f"  E{epoch_num_for_logging} C{cls_idx} SVD Info: batch_feature_list length is {actual_num_features}, expected {EXPECTED_NUM_FEATURES_IN_LIST}. Using last feature (index {actual_num_features - 1}) for SVD.")
                            
                            # Use the actual last feature provided by the model for SVD
                            fc_input_tensor_for_batch = batch_feature_list[-1] 
                            fc_output_features_for_svd[cls_idx].append(fc_input_tensor_for_batch[mask][:samples_to_take_svd].cpu())
                        else:
                            # This means the model provided NO features at all in batch_feature_list
                            if epoch_num_for_logging >= 0 and cls_idx in classes_to_log_details:
                                print(f"  E{epoch_num_for_logging} C{cls_idx} SVD Warning: batch_feature_list is EMPTY. Cannot collect SVD features.")


                # Collect multi-layer features for dormancy
                current_dormancy_samples_collected = sum(f.shape[0] for f in class_features_multilayer[cls_idx][0]) if class_features_multilayer[cls_idx].get(0) and class_features_multilayer[cls_idx][0] else 0

                if current_dormancy_samples_collected < MAX_SAMPLES_PER_CLASS_FOR_FEATURES:
                    needed_dormancy = MAX_SAMPLES_PER_CLASS_FOR_FEATURES - current_dormancy_samples_collected
                    samples_to_take_dormancy = min(num_class_samples_in_batch, needed_dormancy)

                    if samples_to_take_dormancy > 0:
                        # feat_idx is the layer index?
                        for feat_idx, layer_batch_tensor in enumerate(batch_feature_list):
                            # Slice the layer_batch_tensor for the current class, take up to samples_to_take_dormancy
                            class_specific_features = layer_batch_tensor[mask][:samples_to_take_dormancy].detach().cpu()
                            if class_features_multilayer[cls_idx].get(feat_idx) is not None:
                                class_features_multilayer[cls_idx][feat_idx].append(class_specific_features)
                            else:
                                # This case should ideally not happen if EXPECTED_NUM_FEATURES_IN_LIST is correct
                                print(f"Warning: Feature index {feat_idx} not initialized for class {cls_idx}. Skipping feature collection for this index.")
        
        # Optional: Early break if all classes have enough features
        if batch_idx > 10: # Check after a few batches to allow collection
            all_done_svd = all(sum(f.shape[0] for f in fc_output_features_for_svd[cid]) >= MAX_SAMPLES_PER_CLASS_FOR_FEATURES for cid in ordered_classes_int)
            all_done_dormancy = all( (class_features_multilayer[cid].get(0) and class_features_multilayer[cid][0] and sum(f.shape[0] for f in class_features_multilayer[cid][0]) >= MAX_SAMPLES_PER_CLASS_FOR_FEATURES) for cid in ordered_classes_int)
            if all_done_svd and all_done_dormancy:
                print(f"Collected enough samples for all classes by batch {batch_idx+1}. Breaking early.")
                break


    results = {}
    print("\nCalculating final metrics (dormancy, rank, etc.)...")
    
    classes_to_log_details = ordered_classes_int[:debug_class_limit] if debug_class_limit > 0 else []

    # Add enumeration to keep track of the position in the class order
    for i, cls_idx in enumerate(tqdm(ordered_classes_int, desc="Finalizing metrics per class")):
        # Initialize the results dictionary for this class
        current_class_results = {}
        total_samples_for_class = class_total[cls_idx]

        current_class_results['accuracy'] = class_correct[cls_idx] / total_samples_for_class if total_samples_for_class > 0 else 0.0
        current_class_results['top5_accuracy'] = class_correct_top5[cls_idx] / total_samples_for_class if total_samples_for_class > 0 else 0.0
        current_class_results['avg_confidence_predicted'] = class_confidence_predicted[cls_idx] / total_samples_for_class if total_samples_for_class > 0 else 0.0
        current_class_results['avg_confidence_true'] = class_confidence_true[cls_idx] / total_samples_for_class if total_samples_for_class > 0 else 0.0

        # Weight Magnitude needs to be modified for the original (Dohare, 2024) calculation.
        weight_mag = 0.0
        if hasattr(net, 'fc') and net.fc is not None and hasattr(net.fc, 'weight') and net.fc.weight is not None:
            if cls_idx < net.fc.weight.shape[0]:
                weight_mag = torch.norm(net.fc.weight[cls_idx, :]).cpu().item()
        current_class_results['weight_magnitude'] = weight_mag

        # Dormant Units Proportion (Multi-Layer)
        total_dormant_units_aggregate = 0
        total_units_aggregate = 0
        
        if total_samples_for_class > 0: # Only calculate if class had samples
            for feat_idx in range(EXPECTED_NUM_FEATURES_IN_LIST):
                collected_tensors = class_features_multilayer[cls_idx].get(feat_idx, [])
                if not collected_tensors:
                    continue
                
                feature_tensor_for_class = torch.cat(collected_tensors, dim=0).to(device)
                if feature_tensor_for_class.shape[0] == 0: continue

                activity_rate_per_unit = None
                num_units_this_layer = 0

                is_fc_input_layer = (feat_idx == EXPECTED_NUM_FEATURES_IN_LIST - 1)

                if is_fc_input_layer:
                    if feature_tensor_for_class.ndim == 2:
                        activity_rate_per_unit = (feature_tensor_for_class != 0).float().mean(dim=0)
                        num_units_this_layer = feature_tensor_for_class.shape[1]
                    elif feature_tensor_for_class.ndim == 1 and feature_tensor_for_class.shape[0] > 0 : # Single sample feature vector
                        activity_rate_per_unit = (feature_tensor_for_class != 0).float()
                        num_units_this_layer = feature_tensor_for_class.shape[0]
                else: # Conv layers
                    if feature_tensor_for_class.ndim == 4:
                        activity_rate_per_unit = (feature_tensor_for_class != 0).float().mean(dim=(0, 2, 3)) # Avg over N, H, W
                        num_units_this_layer = feature_tensor_for_class.shape[1] # Channels
                    elif feature_tensor_for_class.ndim == 3 and feature_tensor_for_class.shape[0] == 1 and not is_fc_input_layer: # Single sample (from batch), (C,H,W)
                         # Squeeze batch dim if it was 1, then process as C,H,W
                        squeezed_tensor = feature_tensor_for_class.squeeze(0)
                        if squeezed_tensor.ndim == 3: # Should be C, H, W
                             activity_rate_per_unit = (squeezed_tensor != 0).float().mean(dim=(1,2)) # Avg over H,W
                             num_units_this_layer = squeezed_tensor.shape[0] # Channels

                if activity_rate_per_unit is not None and num_units_this_layer > 0:
                    dormant_this_layer = (activity_rate_per_unit < dormant_threshold).sum().item()
                    total_dormant_units_aggregate += dormant_this_layer
                    total_units_aggregate += num_units_this_layer

                    if cls_idx in classes_to_log_details and epoch_num_for_logging >= 0: 
                        print(f"  E{epoch_num_for_logging} C{cls_idx} F{feat_idx}: Shape {list(feature_tensor_for_class.shape)}, ActRate(Min/Max) {activity_rate_per_unit.min():.2e}/{activity_rate_per_unit.max():.2e}, Dormant {dormant_this_layer}/{num_units_this_layer}")
                elif cls_idx in classes_to_log_details and epoch_num_for_logging >=0 and collected_tensors and not (activity_rate_per_unit is not None and num_units_this_layer > 0):
                     # Log if we expected to process but didn't (e.g. wrong ndim)
                     print(f"  E{epoch_num_for_logging} C{cls_idx} F{feat_idx}: Skipped processing. Tensor shape {list(feature_tensor_for_class.shape)}, ndim {feature_tensor_for_class.ndim}. Expected 2D for FC-input, 4D for conv (or 3D if single sample conv output).")

        current_class_results['dormant_units_proportion'] = total_dormant_units_aggregate / total_units_aggregate if total_units_aggregate > 0 else 0.0
        if cls_idx in classes_to_log_details and epoch_num_for_logging >=0 and total_samples_for_class > 0:
            dormant_prop_debug = current_class_results['dormant_units_proportion']
            print(f"E{epoch_num_for_logging} C{cls_idx} Final Dormant Prop: {dormant_prop_debug:.4f} ({total_dormant_units_aggregate}/{total_units_aggregate})")

        # Effective and Stable Rank (using FC output features)
        if fc_output_features_for_svd[cls_idx]:
            cls_fc_tensor_for_svd = torch.cat(fc_output_features_for_svd[cls_idx], dim=0).cpu().numpy()
            if cls_fc_tensor_for_svd.shape[0] > 1 and cls_fc_tensor_for_svd.shape[1] > 1:
                try:
                    sv_values = np.linalg.svd(cls_fc_tensor_for_svd, compute_uv=False)
                    current_class_results['effective_rank'] = compute_effective_rank(sv_values)
                    current_class_results['stable_rank'] = compute_stable_rank(sv_values)
                except np.linalg.LinAlgError:
                    current_class_results['effective_rank'] = 0.0
                    current_class_results['stable_rank'] = 0.0
            else:
                current_class_results['effective_rank'] = 0.0
                current_class_results['stable_rank'] = 0.0

            # --- Dormant Units Proportion Calculation ---
            dormant_threshold = 0.01
            dormant_prop = 0.0 # Default to 0.0
            
            # Convert numpy array back to tensor for dormancy calculation
            cls_features_tensor = torch.tensor(cls_fc_tensor_for_svd)
            
            if cls_features_tensor.numel() > 0 and cls_features_tensor.shape[0] > 0: # Must have samples
                if cls_features_tensor.ndim == 2 and cls_features_tensor.shape[1] > 0: # FC-like: N x Features
                    total_units = cls_features_tensor.shape[1]
                    activity_rate_per_unit = (cls_features_tensor != 0).float().mean(dim=0) # Average over samples (dim=0)
                    if i < 3: # Print for first 3 overall classes processed (i is from enumerate(class_order))
                        print(f"DEBUG_DORMANCY (Class {cls_idx}, OrderPos {i}, 2D Features):")
                        print(f"  cls_features_tensor.shape: {cls_features_tensor.shape}")
                        print(f"  activity_rate_per_unit (first 10): {activity_rate_per_unit[:10]}")
                        print(f"  min_activity_rate: {torch.min(activity_rate_per_unit).item()}, max_activity_rate: {torch.max(activity_rate_per_unit).item()}")
                        print(f"  dormant_threshold: {dormant_threshold}")
                    num_dormant_units = (activity_rate_per_unit < dormant_threshold).sum().cpu().item()
                    if i < 3:
                        print(f"  num_dormant_units (units with rate < threshold): {num_dormant_units}")
                    dormant_prop = num_dormant_units / total_units
                elif cls_features_tensor.ndim == 4 and cls_features_tensor.shape[1] > 0: # Conv-like: N x Channels x H x W
                    total_units = cls_features_tensor.shape[1] # Number of channels
                    activity_rate_per_unit = (cls_features_tensor != 0).float().mean(dim=(0, 2, 3)) # Average over N, H, W for each channel
                    if i < 3: # Print for first 3 overall classes processed (i is from enumerate(class_order))
                        print(f"DEBUG_DORMANCY (Class {cls_idx}, OrderPos {i}, 4D Features):")
                        print(f"  cls_features_tensor.shape: {cls_features_tensor.shape}")
                        print(f"  activity_rate_per_unit (first 10 channels): {activity_rate_per_unit[:10]}")
                        print(f"  min_activity_rate: {torch.min(activity_rate_per_unit).item()}, max_activity_rate: {torch.max(activity_rate_per_unit).item()}")
                        print(f"  dormant_threshold: {dormant_threshold}")
                    num_dormant_units = (activity_rate_per_unit < dormant_threshold).sum().cpu().item()
                    if i < 3:
                        print(f"  num_dormant_units (units with rate < threshold): {num_dormant_units}")
                    dormant_prop = num_dormant_units / total_units
                else: # Unsuitable shape for dormant calculation or 0 features in the relevant dimension
                    if (cls_features_tensor.ndim == 2 or cls_features_tensor.ndim == 4) and cls_features_tensor.shape[1] == 0:
                        print(f"Info: cls_features_tensor for class {cls_idx} (shape: {cls_features_tensor.shape}) has 0 units in feature/channel dimension. Dormant proportion set to 0.0.")
                    else:
                        print(f"Warning: cls_features_tensor for class {cls_idx} (shape: {cls_features_tensor.shape}) has an unexpected shape for dormant unit calculation. Dormant proportion set to 0.0.")
                    # dormant_prop remains 0.0 as initialized above this if/elif/else block for dormant calculation
            else: # Tensor is empty or has no samples
                if cls_features_tensor.numel() > 0: # Has elements but shape[0] is 0 (no samples)
                    print(f"Info: cls_features_tensor for class {cls_idx} (shape: {cls_features_tensor.shape}) has no samples. Dormant proportion set to 0.0.")
                # If tensor is empty (numel == 0), dormant_prop remains 0.0 as initialized
        else:
            eff_rank = 0.0
            stable_rank = 0.0
            dormant_prop = 0.0 # Ensure dormant_prop is set when features are not available
            
        # Extract values safely from current_class_results
        effective_rank_value = current_class_results.get('effective_rank', 0.0)
        stable_rank_value = current_class_results.get('stable_rank', 0.0)
        dormant_prop_value = current_class_results.get('dormant_units_proportion', 0.0) # B: retrieve dormant prop from current_class_results
        
        # Create the final results entry in the expected format
        results[cls_idx] = {
            "class_id": int(cls_idx),  # Class ID in CIFAR-100
            "order_position": int(class_position_mapping.get(cls_idx, -1)),  # Position in the order they were presented
            "accuracy": float(current_class_results['accuracy']),  # Top-1 accuracy
            "accuracy_top5": float(current_class_results['top5_accuracy']),  # Top-5 accuracy
            "confidence": float(current_class_results['avg_confidence_predicted']),  # Avg confidence of predictions
            "true_class_confidence": float(current_class_results['avg_confidence_true']),  # Avg confidence for true class
            "effective_rank": float(effective_rank_value),  # Effective rank of features
            "stable_rank": float(stable_rank_value),  # Stable rank of features
            "samples": int(class_total[cls_idx]),  # Number of samples
            "weight_magnitude": float(current_class_results['weight_magnitude']),
            "dormant_units_proportion": float(dormant_prop_value) # B: "dormant_units_proportion": float(dormant_prop_value)
        }
    
    return results


# -------------------- For computing per-image metrics -------------------- #
@torch.no_grad()
def compute_metrics_per_image(net: torch.nn.Module, data_loader: DataLoader, max_images: int = 1000):
    """
    Computes metrics for individual images
    :param net: neural network with loaded parameters
    :param data_loader: cifar100 data loader
    :param max_images: maximum number of images to analyze
    :return: dictionary with per-image metrics
    """
    device = net.fc.weight.device
    image_metrics = []
    image_count = 0
    
    for sample in data_loader:
        image = sample["image"].to(device)
        labels = sample["label"].to(device)
        _, labels_idx = torch.max(labels, 1)   # Get the class with the highest score
        
        # Forward pass and collect features
        temp_features = []
        outputs = net.forward(image, temp_features)
        _, predicted = torch.max(outputs, 1)    # Get the class with the highest score
        
        # Get top-k predictions and their confidence
        k = 5  # Top-5 predictions
        _, top_k_indices = torch.topk(outputs, k, dim=1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)
        
        # Process each image in the batch
        batch_size = image.shape[0]
        for i in range(batch_size):
            if image_count >= max_images:
                break
                
            true_class = labels_idx[i].item()
            pred_class = predicted[i].item()
            correct = (pred_class == true_class)
            
            # Top-1 confidence (predicted class)
            confidence_score = confidence[i, pred_class].item()
            
            # Confidence for true class
            true_class_confidence = confidence[i, true_class].item()
            
            # Top-5 correct
            top_k_correct = true_class in top_k_indices[i]
            
            # Top-5 confidences
            top_k_confidences = [confidence[i, idx].item() for idx in top_k_indices[i]]
            
            # Entropy of confidence distribution (uncertainty measure)
            probs = confidence[i].cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            # Compute activation patterns
            activation_stats = {}
            
            # Compute activation sparsity
            activation_sparsity = 0
            activation_magnitude = 0
            for layer_idx, layer_features in enumerate(temp_features):
                layer_name = f"layer_{layer_idx}"
                
                if len(layer_features.shape) == 4:  # Conv features
                    # Sparsity (fraction of zero activations)
                    sparse_ratio = (layer_features[i] == 0).float().mean().item()
                    # Magnitude (average absolute activation value)
                    magnitude = torch.abs(layer_features[i]).mean().item()
                    # Max activation
                    max_activation = torch.max(torch.abs(layer_features[i])).item()
                    
                    activation_stats[layer_name] = {
                        "sparsity": float(sparse_ratio),
                        "magnitude": float(magnitude),
                        "max_activation": float(max_activation)
                    }
                    
                elif len(layer_features.shape) == 2:  # FC features
                    sparse_ratio = (layer_features[i] == 0).float().mean().item()
                    magnitude = torch.abs(layer_features[i]).mean().item()
                    max_activation = torch.max(torch.abs(layer_features[i])).item()
                    
                    activation_stats[layer_name] = {
                        "sparsity": float(sparse_ratio),
                        "magnitude": float(magnitude),
                        "max_activation": float(max_activation)
                    }
                
                activation_sparsity += sparse_ratio
                activation_magnitude += magnitude
            
            # Average across layers
            activation_sparsity /= len(temp_features)  
            activation_magnitude /= len(temp_features)
            
            # Store metrics for this image
            image_metrics.append({
                "image_id": int(image_count),
                "true_class": int(true_class),
                "predicted_class": int(pred_class),
                "correct": bool(correct),
                "top5_correct": bool(top_k_correct),
                "confidence": float(confidence_score),
                "true_class_confidence": float(true_class_confidence),
                "top5_confidences": [float(c) for c in top_k_confidences],
                "top5_classes": [int(idx.item()) for idx in top_k_indices[i]],
                "entropy": float(entropy),
                "activation_sparsity": float(activation_sparsity),
                "activation_magnitude": float(activation_magnitude),
                "layer_activations": activation_stats
            })
            
            image_count += 1
            if image_count >= max_images:
                break
    
    return image_metrics


# -------------------- For storing the results of the analysis -------------------- #
def store_detailed_metrics(per_class_metrics: dict, per_image_metrics: list, results_dir: str, experiment_index: int):
    """
    Stores the per-class and per-image metrics results
    :param per_class_metrics: dictionary containing metrics for each class
    :param per_image_metrics: list containing metrics for individual images
    :param results_dir: path to the results directory
    :param experiment_index: experiment index
    """
    # Create directories if they don't exist
    per_class_metrics_dir = os.path.join(results_dir, "per_class_metrics")
    if not os.path.isdir(per_class_metrics_dir): 
        os.makedirs(per_class_metrics_dir)
        
    per_image_metrics_dir = os.path.join(results_dir, "per_image_metrics")
    if not os.path.isdir(per_image_metrics_dir): 
        os.makedirs(per_image_metrics_dir)

    # Store per-class metrics
    if per_class_metrics is not None:
        per_class_metrics_file_name = "per_class_metrics-{0}.json".format(experiment_index)
        with open(os.path.join(per_class_metrics_dir, per_class_metrics_file_name), 'w') as f:
            json.dump(per_class_metrics, f, indent=2)
        print(f"Stored per-class metrics for experiment index {experiment_index} in {per_class_metrics_dir}")

    # Store per-image metrics
    if per_image_metrics is not None:
        per_image_metrics_file_name = "per_image_metrics-{0}.json".format(experiment_index)
        with open(os.path.join(per_image_metrics_dir, per_image_metrics_file_name), 'w') as f:
            json.dump(per_image_metrics, f, indent=2)
        print(f"Stored per-image metrics for experiment index {experiment_index} in {per_image_metrics_dir}")


def analyze_detailed_metrics(results_dir: str, data_path: str, max_images: int = 1000, dormant_threshold: float = 0.01, debug_class_limit: int = 3, epoch_to_analyze: int = 4000):
    """
    Analyzes classes and individual images to generate detailed metrics
    :param results_dir: path to directory containing the results for a parameter combination
    :param data_path: path to the cifar100 data set
    :param max_images: maximum number of images to analyze per experiment
    """
    parameter_dir_path = os.path.join(results_dir, "model_parameters")
    experiment_indices_file_path = os.path.join(results_dir, "experiment_indices.npy")
    class_order_dir_path = os.path.join(results_dir, "class_order")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # epoch_to_analyze is now a parameter, replacing last_epoch
    # Load experiment indices, ensuring it's an iterable (handle scalar case)
    experiment_indices_raw = np.load(experiment_indices_file_path)
    
    # Check if it's a scalar (0-dim array)
    if experiment_indices_raw.ndim == 0:  # It's a scalar
        experiment_indices = [int(experiment_indices_raw)]  # Convert to list with single value
        print(f"Loaded scalar experiment index: {experiment_indices[0]}")
    else:  # It's already an array
        experiment_indices = experiment_indices_raw
        print(f"Loaded {len(experiment_indices)} experiment indices")

    net = build_resnet18(num_classes=100, norm_layer=torch.nn.BatchNorm2d)
    net.to(device)
    
    # We'll use the test data for our analysis
    _, test_data_loader = load_cifar_data(data_path, train=False)

    for exp_index in tqdm(experiment_indices):
        ordered_classes = load_classes(class_order_dir_path, index=exp_index)
        
        # Load the model from the final epoch
        print(f"Loading model for experiment index {exp_index}...")
        net.load_state_dict(load_model_parameters(parameter_dir_path, exp_index, epoch_to_analyze))
        
        # Compute per-class metrics
        print(f"Computing per-class metrics for experiment index {exp_index}...")
        per_class_metrics = compute_metrics_per_class(net, ordered_classes, test_data_loader, dormant_threshold=dormant_threshold, debug_class_limit=debug_class_limit, epoch_num_for_logging=epoch_to_analyze)
        
        # Compute per-image metrics
        print(f"Computing per-image metrics for experiment index {exp_index}...")
        per_image_metrics = compute_metrics_per_image(net, test_data_loader, max_images=max_images)
        
        # Store the metrics
        store_detailed_metrics(
            per_class_metrics=per_class_metrics,
            per_image_metrics=per_image_metrics,
            results_dir=results_dir,
            experiment_index=exp_index
        )


def parse_arguments() -> dict:
    file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', action="store", type=str,
                        default=os.path.join(file_path, "results", "base_deep_learning_system"),
                        help="Path to directory with the results of a parameter combination.")
    parser.add_argument('--data_path', action="store", type=str, default=os.path.join(file_path, "data"),
                        help="Path to directory with the CIFAR 100 data set.")
    parser.add_argument('--max_images', action="store", type=int, default=1000,
                        help="Maximum number of images to analyze per experiment.")
    parser.add_argument('--dormant_threshold', action="store", type=float, default=0.01,
                        help="Threshold for considering a unit dormant.")
    parser.add_argument('--debug_class_limit', action="store", type=int, default=3,
                        help="Limits detailed console logging to this many classes (e.g., 3 for first 3). Set to 0 or negative for no detailed class-specific logging.")
    parser.add_argument('--epoch_to_analyze', action="store", type=int, default=4000,
                        help="The epoch number of the model to load and analyze.")

    args = parser.parse_args()
    return vars(args)


def main():
    analysis_arguments = parse_arguments()

    initial_time = time.perf_counter()
    analyze_detailed_metrics(
        results_dir=analysis_arguments["results_dir"],
        data_path=analysis_arguments["data_path"],
        max_images=analysis_arguments["max_images"],
        dormant_threshold=analysis_arguments["dormant_threshold"],
        debug_class_limit=analysis_arguments["debug_class_limit"],
        epoch_to_analyze=analysis_arguments["epoch_to_analyze"]
    )
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
