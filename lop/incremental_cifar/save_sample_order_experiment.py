#!/usr/bin/env python3.8
"""
Modified incremental CIFAR experiment that saves sample order for every epoch to npy/npz files.
This enables exact replication of random sample ordering patterns.

Based on: incremental_cifar_memo_ordered_experiment.py
Purpose: Save sample order for each epoch to enable replication of successful random ordering
"""

# built-in libraries
import time
import os
import pickle
from copy import deepcopy
import json
import argparse
from functools import partialmethod
import pandas as pd

# third party libraries
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
from mlproj_manager.file_management.file_and_directory_management import store_object_with_several_attempts

from lop.nets.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module
from lop.algos.res_gnt import ResGnT


def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """
    Sub-samples the CIFAR 100 data set according to the given indices
    :param sub_sample_indices: array of indices in the same format as the cifar data set (numpy or torch)
    :param cifar_data: cifar data to be sub-sampled
    :return: None, but modifies the given cifar_dataset
    """
    cifar_data.data["data"] = cifar_data.data["data"][sub_sample_indices.numpy()]
    cifar_data.data["labels"] = cifar_data.data["labels"][sub_sample_indices.numpy()]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[sub_sample_indices.numpy()].tolist()
    cifar_data.current_data = cifar_data.partition_data()


class SampleOrderSavingExperiment(Experiment):
    """
    Modified experiment class that saves sample order for every epoch to npy/npz files.
    """

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True, 
                 gpu_id=0, scratch=False, epochs_per_task=200, start_task=0, max_tasks=1,
                 save_format='npz', random_seed=None):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        # disable tqdm if verbose is enabled
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=self.verbose)

        """ For reproducibility """
        if random_seed is not None:
            self.random_seed = random_seed
        else:
            random_seeds = get_random_seeds()
            self.random_seed = random_seeds[self.run_index]
        
        torch.random.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        print(f"Using random seed: {self.random_seed}")

        """ Experiment parameters """
        self.data_path = exp_params["data_path"]
        self.num_workers = access_dict(exp_params, key="num_workers", default=1, val_type=int)

        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        self.early_stopping = access_dict(exp_params, "early_stopping", default=False, val_type=bool)

        # cbp parameters
        self.use_cbp = access_dict(exp_params, "use_cbp", default=False, val_type=bool)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        assert (not self.use_cbp) or (self.replacement_rate > 0.0), "Replacement rate should be greater than 0."
        self.utility_function = access_dict(exp_params, "utility_function", default="weight", val_type=str,
                                            choices=["weight", "contribution"])
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        assert (not self.use_cbp) or (self.maturity_threshold > 0), "Maturity threshold should be greater than 0."

        # shrink and perturb parameters
        self.noise_std = access_dict(exp_params, "noise_std", default=0.0, val_type=float)
        self.perturb_weights_indicator = self.noise_std > 0.0

        """ For task independence and variable epochs """
        self.scratch = scratch
        self.epochs_per_task = epochs_per_task
        self.start_task = start_task
        self.max_tasks = max_tasks
        self.save_format = save_format  # 'npy', 'npz', or 'both'

        # Log experiment configuration
        device_str = f"GPU {gpu_id}" if torch.cuda.is_available() else "CPU"
        print(f"[{device_str}] Sample Order Saving Experiment")
        print(f"[{device_str}] Task {self.start_task}, epochs={self.epochs_per_task}, scratch={self.scratch}")
        print(f"[{device_str}] Save format: {self.save_format}")

        """ Training constants """
        self.num_epochs = epochs_per_task
        # Single task mode: start with ALL classes up to and including the target task
        self.current_num_classes = (self.start_task + 1) * 5
        self.batch_sizes = {"train": 90, "test": 100, "validation": 50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.num_images_per_class = 450
        self.current_task = self.start_task

        """ Network set up """
        # initialize network
        self.net = build_resnet18(num_classes=self.num_classes, norm_layer=torch.nn.BatchNorm2d)
        self.net.apply(kaiming_init_resnet_module)

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # move network to device
        self.net.to(self.device)
        self.current_epoch = 0

        # for cbp
        self.resgnt = None
        if self.use_cbp:
            self.resgnt = ResGnT(net=self.net,
                                 hidden_activation="relu",
                                 replacement_rate=self.replacement_rate,
                                 decay_rate=0.99,
                                 util_type=self.utility_function,
                                 maturity_threshold=self.maturity_threshold,
                                 device=self.device)
        self.current_features = [] if self.use_cbp else None

        """ For data partitioning """
        # Load random class order for consistency with successful experiments
        self.all_classes = self.load_random_class_order()
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_accuracy_model_parameters = {}

        """ For creating sample order storage """
        self.sample_orders_dir = os.path.join(self.results_dir, "sample_orders")
        os.makedirs(self.sample_orders_dir, exist_ok=True)
        print(f"Sample orders will be saved to: {self.sample_orders_dir}")

        """ For summaries """
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.running_samples, self.running_batches = (0, 0.0, 0.0, 0, 0)
        self.epoch_loss_accumulator = 0.0
        self.epoch_accuracy_accumulator = 0.0
        self.epoch_batches_accumulator = 0
        self._initialize_summaries()

    def load_random_class_order(self):
        """Load random class order for consistency with successful experiments."""
        np.random.seed(42)  # For reproducibility
        classes = np.arange(100)
        np.random.shuffle(classes)
        return classes

    def save_epoch_sample_order(self, sample_indices, epoch, task_id, metadata=None):
        """
        Save sample order for a specific epoch to npy/npz files.
        
        :param sample_indices: Tensor or list of sample indices in training order
        :param epoch: Current epoch number
        :param task_id: Current task identifier
        :param metadata: Optional dictionary with additional metadata
        """
        try:
            # Convert to numpy if tensor
            if torch.is_tensor(sample_indices):
                sample_order = sample_indices.cpu().numpy()
            else:
                sample_order = np.array(sample_indices)

            # Prepare metadata
            epoch_metadata = {
                'epoch': epoch,
                'task_id': task_id,
                'random_seed': self.random_seed,
                'batch_size': self.batch_sizes["train"],
                'total_samples': len(sample_order),
                'timestamp': time.time(),
                'scratch_mode': self.scratch,
                'current_num_classes': self.current_num_classes
            }
            
            if metadata:
                epoch_metadata.update(metadata)

            # Create filenames
            base_filename = f"task_{task_id}_epoch_{epoch:04d}_sample_order"
            
            if self.save_format in ['npy', 'both']:
                # Save as npy (just the sample order array)
                npy_filepath = os.path.join(self.sample_orders_dir, f"{base_filename}.npy")
                np.save(npy_filepath, sample_order)
                
            if self.save_format in ['npz', 'both']:
                # Save as npz (sample order + metadata)
                npz_filepath = os.path.join(self.sample_orders_dir, f"{base_filename}.npz")
                np.savez_compressed(npz_filepath, 
                                    sample_order=sample_order,
                                    **epoch_metadata)

            # Log progress every 10 epochs or on first/last epoch
            if epoch == 0 or epoch == self.epochs_per_task - 1 or epoch % 10 == 0:
                print(f"Saved sample order for task {task_id}, epoch {epoch} ({len(sample_order)} samples)")
                print(f"  First 10 samples: {sample_order[:10].tolist()}")
                if self.save_format in ['npz', 'both']:
                    print(f"  Metadata: seed={self.random_seed}, classes={self.current_num_classes}, batch_size={self.batch_sizes['train']}")

        except Exception as e:
            print(f"Error saving sample order for epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()

    def get_dataloader_sample_order(self, dataloader):
        """
        Extract the actual sample order that will be used by the DataLoader.
        
        :param dataloader: The DataLoader object
        :return: numpy array of sample indices in training order
        """
        try:
            # Get the dataset from the dataloader
            dataset = dataloader.dataset
            
            # Get the sampler
            sampler = dataloader.sampler
            
            # Extract indices based on sampler type
            if hasattr(sampler, '__iter__'):
                # For RandomSampler, we need to replicate the same random generation
                if isinstance(sampler, torch.utils.data.RandomSampler):
                    # Use the current random state to generate the same permutation
                    generator = torch.Generator()
                    generator.manual_seed(self.random_seed + self.current_epoch)  # Include epoch for variation
                    
                    n_samples = len(dataset)
                    indices = torch.randperm(n_samples, generator=generator).numpy()
                    
                    return indices
                else:
                    # For other samplers, iterate through them
                    return np.array(list(sampler))
            else:
                # Fallback: assume sequential order
                return np.arange(len(dataset))
                
        except Exception as e:
            print(f"Warning: Could not extract sample order from DataLoader: {e}")
            # Fallback to sequential order
            return np.arange(len(dataloader.dataset))

    def run_epoch_with_sample_logging(self, train_dataloader, test_dataloader, 
                                      original_cifar_data, task_id):
        """
        Run training for specified epochs while saving sample order for each epoch.
        
        :param train_dataloader: Training DataLoader
        :param test_dataloader: Test DataLoader  
        :param original_cifar_data: Original CIFAR dataset
        :param task_id: Current task identifier
        """
        print(f"\nStarting training for task {task_id} with sample order logging...")
        print(f"Total epochs: {self.epochs_per_task}")
        print(f"Save format: {self.save_format}")
        
        for epoch in range(self.epochs_per_task):
            self.current_epoch = epoch
            
            # Extract and save sample order for this epoch BEFORE training
            sample_order = self.get_dataloader_sample_order(train_dataloader)
            
            # Save the sample order
            self.save_epoch_sample_order(sample_order, epoch, task_id)
            
            # Run actual training epoch
            self.net.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            epoch_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optim.zero_grad()
                output = self.net(data)
                loss = self.loss(output, target)
                
                # Backward pass
                loss.backward()
                self.optim.step()
                
                # Track metrics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_accuracy += pred.eq(target.view_as(pred)).sum().item()
                epoch_batches += 1
                
                # CBP update if enabled
                if self.use_cbp:
                    self.current_features = data
                    self.resgnt.update_plasticity()
            
            # Calculate epoch metrics
            avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            avg_accuracy = epoch_accuracy / len(train_dataloader.dataset) if len(train_dataloader.dataset) > 0 else 0.0
            
            # Log progress
            if epoch % 10 == 0 or epoch == self.epochs_per_task - 1:
                print(f"Epoch {epoch:4d}/{self.epochs_per_task}: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")
                
        print(f"Training completed for task {task_id}. Sample orders saved for all {self.epochs_per_task} epochs.")

    def save_final_summary(self, task_id):
        """Save a summary of all saved sample orders."""
        try:
            summary_data = {
                'experiment_type': 'sample_order_logging',
                'task_id': task_id,
                'total_epochs': self.epochs_per_task,
                'random_seed': self.random_seed,
                'save_format': self.save_format,
                'batch_size': self.batch_sizes["train"],
                'current_num_classes': self.current_num_classes,
                'scratch_mode': self.scratch,
                'timestamp': time.time(),
                'description': 'Sample orders saved for each epoch to enable exact replication of random ordering patterns'
            }
            
            summary_filepath = os.path.join(self.sample_orders_dir, f"task_{task_id}_experiment_summary.json")
            with open(summary_filepath, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
            print(f"\nExperiment summary saved to: {summary_filepath}")
            print(f"Total sample order files created: {self.epochs_per_task}")
            print(f"Files located in: {self.sample_orders_dir}")
            
        except Exception as e:
            print(f"Error saving experiment summary: {e}")

    def run_experiment(self):
        """Main experiment execution with sample order logging."""
        print(f"\n{'='*60}")
        print(f"SAMPLE ORDER LOGGING EXPERIMENT")
        print(f"Task: {self.start_task}, Epochs: {self.epochs_per_task}")
        print(f"Random Seed: {self.random_seed}")
        print(f"Save Format: {self.save_format}")
        print(f"{'='*60}")

        # Load CIFAR data
        cifar_data = CifarDataSet(data_path=self.data_path)
        print(f"Loaded CIFAR-100 dataset: {len(cifar_data.data['data'])} samples")

        # For single task mode, select classes up to the target task
        num_classes_for_task = (self.start_task + 1) * 5
        selected_classes = self.all_classes[:num_classes_for_task]
        print(f"Selected {len(selected_classes)} classes: {selected_classes.tolist()}")

        # Get samples for selected classes
        selected_indices = []
        for class_idx in selected_classes:
            class_samples = [i for i, label in enumerate(cifar_data.integer_labels) if label == class_idx]
            selected_indices.extend(class_samples)
        
        selected_indices = torch.tensor(selected_indices)
        print(f"Total samples for task: {len(selected_indices)}")

        # Subsample the dataset
        subsample_cifar_data_set(selected_indices, cifar_data)

        # Create transforms
        train_transforms = transforms.Compose([
            ToTensor(),
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
        
        test_transforms = transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        # Apply transforms
        cifar_data.apply_augmentations(train_transforms)
        cifar_data_test = deepcopy(cifar_data)
        cifar_data_test.apply_augmentations(test_transforms)

        # Create data loaders - IMPORTANT: shuffle=True for random sample ordering
        train_dataloader = DataLoader(
            cifar_data, 
            batch_size=self.batch_sizes["train"],
            shuffle=True,  # This creates the random ordering we want to capture
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_dataloader = DataLoader(
            cifar_data_test,
            batch_size=self.batch_sizes["test"],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        print(f"Created DataLoaders: train={len(train_dataloader)} batches, test={len(test_dataloader)} batches")

        # Run training with sample order logging
        self.run_epoch_with_sample_logging(
            train_dataloader, 
            test_dataloader, 
            cifar_data,
            self.start_task
        )

        # Save final summary
        self.save_final_summary(self.start_task)

        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"Sample orders saved for {self.epochs_per_task} epochs")
        print(f"Location: {self.sample_orders_dir}")
        print(f"{'='*60}")

    def _initialize_summaries(self):
        """Initialize the summaries for the experiment."""
        self.results_dict = {}


def main():
    """Main function to run the sample order saving experiment."""
    parser = argparse.ArgumentParser(description='Save sample order for each epoch')
    parser.add_argument('--config_file', type=str, required=True, help='Path to config file')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--run_index', type=int, default=0, help='Run index')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--scratch', action='store_true', help='Use scratch mode')
    parser.add_argument('--epochs_per_task', type=int, default=200, help='Epochs per task')
    parser.add_argument('--start_task', type=int, default=19, help='Task to run (19 = all 100 classes)')
    parser.add_argument('--save_format', type=str, default='npz', choices=['npy', 'npz', 'both'],
                        help='Save format: npy (array only), npz (array+metadata), both')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed (default: use run_index)')
    
    args = parser.parse_args()

    # Load config
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Create experiment
    experiment = SampleOrderSavingExperiment(
        exp_params=config,
        results_dir=args.results_dir,
        run_index=args.run_index,
        verbose=True,
        gpu_id=args.gpu_id,
        scratch=args.scratch,
        epochs_per_task=args.epochs_per_task,
        start_task=args.start_task,
        max_tasks=1,
        save_format=args.save_format,
        random_seed=args.random_seed
    )

    # Run experiment
    experiment.run_experiment()


if __name__ == "__main__":
    main()
