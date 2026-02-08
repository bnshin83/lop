# built-in libraries
import time
import os
import pickle
import random
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


class IndexedDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper that adds sample indices to dataset items for tracking actual DataLoader order.
    """
    def __init__(self, dataset, original_indices):
        self.dataset = dataset
        self.original_indices = original_indices  # Maps dataset index -> actual sample ID
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Create a copy of the sample and add the actual sample ID
        sample_copy = sample.copy() if isinstance(sample, dict) else sample
        if isinstance(sample_copy, dict):
            sample_copy['index'] = self.original_indices[idx]
        else:
            # If sample is not a dict, create a dict wrapper
            sample_copy = {
                'image': sample_copy[0] if hasattr(sample_copy, '__getitem__') else sample_copy,
                'label': sample_copy[1] if hasattr(sample_copy, '__getitem__') and len(sample_copy) > 1 else None,
                'index': self.original_indices[idx]
            }
        return sample_copy


def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """
    Sub-samples the CIFAR 100 data set according to the given indices
    :param sub_sample_indices: array of indices in the same format as the cifar data set (numpy or torch)
    :param cifar_data: cifar data to be sub-sampled
    :return: None, but modifies the given cifar_dataset
    """

    cifar_data.data["data"] = cifar_data.data["data"][sub_sample_indices.numpy()]       # .numpy wasn't necessary with torch 2.0
    cifar_data.data["labels"] = cifar_data.data["labels"][sub_sample_indices.numpy()]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[sub_sample_indices.numpy()].tolist()
    cifar_data.current_data = cifar_data.partition_data()


class IncrementalCIFARMemoOrderedExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True, memo_order="low_to_high", gpu_id=0, no_ordering=False, scratch=False, epochs_per_task=200, incremental_epochs=False, start_task=0, max_tasks=20, class_order="memo_low_to_high", within_task_class_order="task_order", predetermined_sample_order=None, csv_file_path=None, predetermined_sample_csv_path=None, use_memo_weighted_loss=False, memo_threshold=0.25, high_memo_weight=3.0, memo_csv_path="", save_epoch_orders=False, use_predetermined_weights=False, predetermined_weights_csv_path="", weight_dramaticity=1.0):
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
        random_seeds = get_random_seeds()
        self.random_seed = random_seeds[self.run_index]
        torch.random.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        """ Experiment parameters """
        self.data_path = exp_params["data_path"]
        self.num_workers = access_dict(exp_params, key="num_workers", default=1, val_type=int)  # set to 1 when using cpu

        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print(Warning("Resetting the whole network supersedes resetting the head of the network. There's no need to set both to True."))
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
        self.incremental_epochs = incremental_epochs
        self.start_task = start_task
        self.max_tasks = max_tasks
        self.predetermined_sample_order = predetermined_sample_order
        self.csv_file_path = csv_file_path
        self.predetermined_sample_csv_path = predetermined_sample_csv_path
        
        # Load predetermined sample order if specified
        self.predetermined_sample_indices = None
        if self.predetermined_sample_order:
            self._load_predetermined_sample_order()
            
        """ For memorization-aware weighted loss """
        self.use_memo_weighted_loss = use_memo_weighted_loss
        self.memo_threshold = memo_threshold
        self.high_memo_weight = high_memo_weight
        self.memo_csv_path = memo_csv_path if memo_csv_path else "sample_class_map.csv"
        
        # Load memorization scores if weighted loss is enabled
        self.memo_scores_dict = {}
        if self.use_memo_weighted_loss:
            self._load_memorization_scores()
            
        """ For predetermined sample weights """
        self.use_predetermined_weights = use_predetermined_weights
        self.predetermined_weights_csv_path = predetermined_weights_csv_path
        self.weight_dramaticity = weight_dramaticity
        self.predetermined_weights_dict = {}  # Dict to store {sample_id: weight}
        if self.use_predetermined_weights:
            self._load_predetermined_weights()
            
        # For saving epoch sample orders and random seeds
        self.epoch_sample_orders = {}  # Dict to store {epoch_num: [sample_ids]}
        self.epoch_random_seeds = {}   # Dict to store {epoch_num: random_seed_state}
        self.save_epoch_orders = save_epoch_orders  # Enable saving all epoch sample orders
        
        # Log experiment configuration
        device_str = f"GPU {gpu_id}" if torch.cuda.is_available() else "CPU"
        print(f"[{device_str}] Starting experiment with start_task={self.start_task}, max_tasks={self.max_tasks}, scratch={self.scratch}")
        if self.predetermined_sample_order:
            print(f"[{device_str}] Using predetermined sample order: {self.predetermined_sample_order}")
        if self.use_memo_weighted_loss:
            print(f"[{device_str}] Using memorization-aware weighted loss: threshold={self.memo_threshold}, weight={self.high_memo_weight}x")
            print(f"[{device_str}] Memorization scores loaded from: {self.memo_csv_path}")
        if self.use_predetermined_weights:
            print(f"[{device_str}] Using predetermined sample weights from: {self.predetermined_weights_csv_path}")
            print(f"[{device_str}] Weight dramaticity: {self.weight_dramaticity} (formula: 1.0 + dramaticity * memo_score)")
        if self.save_epoch_orders:
            print(f"[{device_str}] Epoch sample order tracking enabled - will save all epoch orders and random seeds")

        """ Training constants """
        self.num_epochs = 50000  # Increased to accommodate ~40,000 checkpoints (40k checkpoints + buffer)
        # Start with classes for the specified start task
        if self.max_tasks == 1:
            # Single task mode: start with ALL classes up to and including the target task
            # Task 0 = 5 classes, Task 1 = 10 classes, ..., Task 19 = 100 classes
            self.current_num_classes = (self.start_task + 1) * 5
        else:
            # Multi-task mode: start with classes for first task, then add incrementally
            self.current_num_classes = 5 + (self.start_task * 5)  # Each task adds 5 classes
        self.batch_sizes = {"train": 90, "test": 100, "validation":50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.num_images_per_class = 450
        self.current_task = self.start_task  # Start from specified task

        """ Network set up """
        # initialize network
        self.net = build_resnet18(num_classes=self.num_classes, norm_layer=torch.nn.BatchNorm2d)
        self.net.apply(kaiming_init_resnet_module)

        # initialize optimizer
        self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, momentum=self.momentum,
                                     weight_decay=self.weight_decay)

        # define loss function
        if self.use_memo_weighted_loss:
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")  # Per-sample loss for weighted training
        else:
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

        """ For memo score ordering """
        self.memo_order = memo_order
        self.no_ordering = no_ordering
        self.class_order = class_order
        self.within_task_class_order = within_task_class_order
        
        # Flag to track whether predetermined sample reordering actually worked
        self.predetermined_reordering_successful = True

        """ For data partitioning """
        self.class_increase_frequency = self.get_current_task_epochs()
        # Load predetermined class order instead of random permutation
        self.all_classes = self.load_class_order()
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_accuracy_model_parameters = {}
        
        if not self.no_ordering:
            self.sample_memo_scores = self.load_sample_memo_scores()
            print(f"Loaded memorization scores for {len(self.sample_memo_scores)} samples")
            print(f"Memo ordering: {memo_order}")
        else:
            self.sample_memo_scores = None
            print(f"No ordering mode: samples will be shuffled randomly (like original experiment)")

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        os.makedirs(self.experiment_checkpoints_dir_path, exist_ok=True)  # Ensure directory exists
        self.checkpoint_identifier_name = "current_epoch"
        # In single task mode, save checkpoint every epoch for detailed analysis
        if self.max_tasks == 1:
            self.checkpoint_save_frequency = 1  # Save every epoch in single task mode
            self.delete_old_checkpoints = False  # Keep all checkpoints for analysis
            print(f"Single task mode: Saving checkpoints every epoch (total: {self.get_current_task_epochs()} checkpoints)")
        else:
            self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added
            self.delete_old_checkpoints = True

        """ For summaries """
        # Will be set adaptively based on training data size
        self.running_avg_window = 25  # Default fallback
        self.current_running_avg_step, self.running_loss, self.running_accuracy, self.running_samples, self.running_batches = (0, 0.0, 0.0, 0, 0)
        
        # Separate accumulators for per-epoch metrics (not reset during epoch)
        self.epoch_loss_accumulator = 0.0
        self.epoch_accuracy_accumulator = 0.0
        self.epoch_batches_accumulator = 0
        
        self._initialize_summaries()

    def load_class_order(self):
        """Load the class order based on class_order parameter."""
        if self.class_order == "memo_low_to_high":
            # Use memorization-based ordering: easiest classes first
            memo_stats_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/per_class_memorization_stats.csv"
            memo_stats_df = pd.read_csv(memo_stats_path)
            memo_stats_df = memo_stats_df.sort_values('mean_memo_score', ascending=True)
            return memo_stats_df['class_label'].values
        elif self.class_order == "memo_high_to_low":
            # Use memorization-based ordering: hardest classes first
            memo_stats_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/per_class_memorization_stats.csv"
            memo_stats_df = pd.read_csv(memo_stats_path)
            memo_stats_df = memo_stats_df.sort_values('mean_memo_score', ascending=False)
            return memo_stats_df['class_label'].values
        elif self.class_order == "random":
            # Use random class order
            np.random.seed(42)  # For reproducibility
            classes = np.arange(100)
            np.random.shuffle(classes)
            return classes
        elif self.class_order == "predetermined":
            # Use predetermined order from index-0.csv
            predetermined_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/base_deep_learning_system/class_order/index-0.csv"
            predetermined_df = pd.read_csv(predetermined_path)
            return predetermined_df['Value'].values
        else:
            # Default: sequential order (0, 1, 2, ..., 99)
            return np.arange(100)

    def load_sample_memo_scores(self):
        """Load sample memorization scores."""
        sample_map_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map.csv"
        sample_df = pd.read_csv(sample_map_path)
        
        # Create a dictionary mapping sample_id to memo score
        memo_scores = {}
        for _, row in sample_df.iterrows():
            memo_scores[row['sample_id']] = row['memorization_score']
        
        print(f"Memo score range: {sample_df['memorization_score'].min():.6f} to {sample_df['memorization_score'].max():.6f}")
        
        return memo_scores
    
    def _track_sample_metadata(self, sample_ids, images, labels, epoch_num, batch_num):
        """Track comprehensive sample metadata for analysis."""
        batch_size = len(sample_ids)
        
        # Convert labels to class IDs if they're one-hot encoded
        if labels.dim() > 1 and labels.shape[1] > 1:
            class_ids = torch.argmax(labels, dim=1).cpu().tolist()
        else:
            class_ids = labels.cpu().tolist()
        
        # Get memo scores for each sample
        memo_scores = []
        for sample_id in sample_ids:
            if hasattr(self, 'sample_memo_scores') and self.sample_memo_scores and sample_id in self.sample_memo_scores:
                memo_scores.append(self.sample_memo_scores[sample_id])
            elif hasattr(self, 'memo_scores_dict') and sample_id in self.memo_scores_dict:
                memo_scores.append(self.memo_scores_dict[sample_id])
            else:
                memo_scores.append(0.0)  # Default if no memo score found
        
        # Extend tracking lists
        self.sample_tracking['sample_ids'].extend(sample_ids)
        self.sample_tracking['memo_scores'].extend(memo_scores)
        self.sample_tracking['labels'].extend(labels.cpu().tolist() if labels.dim() > 1 else labels.cpu().tolist())
        self.sample_tracking['classes'].extend(class_ids)
        self.sample_tracking['epoch_nums'].extend([epoch_num] * batch_size)
        self.sample_tracking['batch_nums'].extend([batch_num] * batch_size)
        self.sample_tracking['task_nums'].extend([self.current_task] * batch_size)
    
    def save_sample_tracking_data(self, filename_suffix=""):
        """Save comprehensive sample tracking data to CSV."""
        if not self.sample_tracking['sample_ids']:
            print("No sample tracking data to save.")
            return
        
        # Create DataFrame with all tracking data
        tracking_df = pd.DataFrame({
            'sample_id': self.sample_tracking['sample_ids'],
            'memo_score': self.sample_tracking['memo_scores'],
            'true_label': self.sample_tracking['labels'],
            'class_id': self.sample_tracking['classes'],
            'epoch_num': self.sample_tracking['epoch_nums'],
            'batch_num': self.sample_tracking['batch_nums'],
            'task_num': self.sample_tracking['task_nums']
        })
        
        # Save to results directory
        filename = f"sample_tracking_data{filename_suffix}.csv"
        filepath = os.path.join(self.results_dir, filename)
        tracking_df.to_csv(filepath, index=False)
        
        print(f"Saved sample tracking data: {filepath}")
        print(f"Total samples tracked: {len(tracking_df)}")
        print(f"Epochs covered: {tracking_df['epoch_num'].min()} to {tracking_df['epoch_num'].max()}")
        print(f"Tasks covered: {sorted(tracking_df['task_num'].unique())}")
        print(f"Memo score range: {tracking_df['memo_score'].min():.4f} to {tracking_df['memo_score'].max():.4f}")
        
        return filepath

    def _load_memorization_scores(self):
        """Load memorization scores from CSV file for weighted loss training."""
        print(f"Loading memorization scores for weighted loss from: {self.memo_csv_path}")
        
        try:
            sample_df = pd.read_csv(self.memo_csv_path)
            
            # Create a dictionary mapping sample_id to memo score
            for _, row in sample_df.iterrows():
                self.memo_scores_dict[int(row['sample_id'])] = float(row['memorization_score'])
                
            print(f"Loaded {len(self.memo_scores_dict)} memorization scores")
            print(f"Memo score range: {sample_df['memorization_score'].min():.6f} to {sample_df['memorization_score'].max():.6f}")
            
            # Calculate statistics about high-memo samples
            high_memo_samples = (sample_df['memorization_score'] > self.memo_threshold).sum()
            total_samples = len(sample_df)
            high_memo_pct = (high_memo_samples / total_samples) * 100
            print(f"High-memo samples (>{self.memo_threshold}): {high_memo_samples}/{total_samples} ({high_memo_pct:.2f}%) will get {self.high_memo_weight}x weight")
            
        except FileNotFoundError:
            print(f"ERROR: Memorization scores file not found: {self.memo_csv_path}")
            print("Disabling memorization-aware weighted loss.")
            self.use_memo_weighted_loss = False
        except Exception as e:
            print(f"ERROR loading memorization scores: {e}")
            print("Disabling memorization-aware weighted loss.")
            self.use_memo_weighted_loss = False

    def _load_predetermined_weights(self):
        """Load predetermined sample weights from CSV file."""
        print(f"Loading predetermined sample weights from: {self.predetermined_weights_csv_path}")
        
        try:
            weights_df = pd.read_csv(self.predetermined_weights_csv_path)
            
            # Expected CSV format: infl,mem,tr_idx,tr_class,tt_idx,tt_class
            # Use tr_idx as sample_id and mem as weight
            # Handle duplicates by keeping the last occurrence (or could average them)
            duplicate_count = 0
            for _, row in weights_df.iterrows():
                sample_id = int(row['tr_idx'])  # training index as sample ID
                weight = float(row['mem'])      # memorization score as weight
                if sample_id in self.predetermined_weights_dict:
                    duplicate_count += 1
                self.predetermined_weights_dict[sample_id] = weight
            
            if duplicate_count > 0:
                print(f"WARNING: Found {duplicate_count} duplicate sample IDs - kept last occurrence")
                
            print(f"Loaded {len(self.predetermined_weights_dict)} predetermined sample weights")
            print(f"Memo score range: {weights_df['mem'].min():.6f} to {weights_df['mem'].max():.6f}")
            
            # Calculate statistics about the final weights (1.0 + dramaticity * memo_score)
            final_weights = 1.0 + (self.weight_dramaticity * weights_df['mem'])
            mean_final_weight = final_weights.mean()
            moderate_boost_samples = (final_weights > 1.5).sum()  # Samples getting >1.5x weight
            high_boost_samples = (final_weights > 2.0).sum()  # Samples getting >2.0x weight  
            extreme_boost_samples = (final_weights > 3.0).sum()  # Samples getting >3.0x weight
            total_samples = len(weights_df)
            moderate_boost_pct = (moderate_boost_samples / total_samples) * 100
            high_boost_pct = (high_boost_samples / total_samples) * 100
            extreme_boost_pct = (extreme_boost_samples / total_samples) * 100
            print(f"Final weight range: {final_weights.min():.2f}x to {final_weights.max():.2f}x")
            print(f"Mean final weight: {mean_final_weight:.2f}x")
            print(f"Moderate-boost samples (>1.5x): {moderate_boost_samples}/{total_samples} ({moderate_boost_pct:.1f}%)")
            print(f"High-boost samples (>2.0x): {high_boost_samples}/{total_samples} ({high_boost_pct:.1f}%)")
            if extreme_boost_samples > 0:
                print(f"Extreme-boost samples (>3.0x): {extreme_boost_samples}/{total_samples} ({extreme_boost_pct:.1f}%)")
            
        except FileNotFoundError:
            print(f"ERROR: Predetermined weights file not found: {self.predetermined_weights_csv_path}")
            print("Disabling predetermined sample weights.")
            self.use_predetermined_weights = False
        except Exception as e:
            print(f"ERROR loading predetermined weights: {e}")
            print("Disabling predetermined sample weights.")
            self.use_predetermined_weights = False

    def _save_epoch_sample_orders(self):
        """Save all epoch sample orders and random seeds to disk in efficient format."""
        if not self.save_epoch_orders or not self.epoch_sample_orders:
            return
            
        print(f"Saving epoch sample orders and random seeds for {len(self.epoch_sample_orders)} epochs...")
        
        # Prepare data for efficient storage
        epoch_data = {
            'epoch_sample_orders': self.epoch_sample_orders,
            'epoch_random_seeds': self.epoch_random_seeds,
            'experiment_info': {
                'task_id': self.current_task,
                'total_epochs': len(self.epoch_sample_orders),
                'use_memo_weighted_loss': self.use_memo_weighted_loss,
                'memo_threshold': self.memo_threshold if self.use_memo_weighted_loss else None,
                'high_memo_weight': self.high_memo_weight if self.use_memo_weighted_loss else None,
                'scratch_mode': self.scratch,
                'no_ordering': self.no_ordering,
                'memo_order': self.memo_order,
                'class_order': self.class_order
            }
        }
        
        # Save as compressed pickle for efficiency
        orders_file = os.path.join(self.results_dir, f"epoch_sample_orders_task_{self.current_task}.pkl")
        try:
            with open(orders_file, 'wb') as f:
                pickle.dump(epoch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Successfully saved epoch sample orders to: {orders_file}")
            
            # Also save as numpy compressed format for easier analysis
            orders_npz = os.path.join(self.results_dir, f"epoch_sample_orders_task_{self.current_task}.npz")
            np_data = {}
            for epoch_num, sample_order in self.epoch_sample_orders.items():
                np_data[f'epoch_{epoch_num}'] = np.array(sample_order, dtype=np.int32)
            
            np.savez_compressed(orders_npz, **np_data)
            print(f"Successfully saved epoch sample orders (numpy format) to: {orders_npz}")
            
        except Exception as e:
            print(f"ERROR saving epoch sample orders: {e}")

    def _save_test_predictions(self, test_predictions_data, epoch, task_id):
        """
        Save test predictions data for analysis.
        
        :param test_predictions_data: List of prediction records
        :param epoch: Current epoch number
        :param task_id: Current task ID
        """
        try:
            import pandas as pd
            
            # Create filename
            test_predictions_file = os.path.join(self.results_dir, f"test_predictions_task_{task_id}_epoch_{epoch}.csv")
            
            # Convert to DataFrame and save
            df = pd.DataFrame(test_predictions_data)
            df.to_csv(test_predictions_file, index=False)
            
            self._print(f"Saved test predictions: {test_predictions_file} ({len(test_predictions_data)} predictions)")
            
        except Exception as e:
            self._print(f"Warning: Could not save test predictions: {e}")

    def _load_predetermined_sample_order(self):
        """Load predetermined sample order from CSV file."""
        # Use custom CSV path if provided, otherwise use default files
        if self.predetermined_sample_csv_path:
            csv_path = self.predetermined_sample_csv_path
            print(f"Loading predetermined sample order from custom file: {csv_path}")
        else:
            # Use default files based on ordering mode
            if self.predetermined_sample_order == "ascending":
                csv_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map_ascending.csv"
            elif self.predetermined_sample_order == "descending":
                csv_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map_descending.csv"
            else:
                raise ValueError(f"Invalid predetermined_sample_order: {self.predetermined_sample_order}. Must be 'ascending' or 'descending'")
            print(f"Loading predetermined sample order from: {csv_path}")
        
        try:
            sample_df = pd.read_csv(csv_path)
            
            # For custom CSV files, limit to first 45,000 samples for training (unless using CSV-based split)
            if self.predetermined_sample_csv_path and len(sample_df) > 45000:
                # Don't truncate if we're using CSV-based train/validation split (needs all 50k samples)
                total_samples = pd.read_csv(csv_path).shape[0]
                if len(sample_df) >= 50000:
                    # Keep all samples for CSV-based split (45k train + 5k validation)
                    print(f"Using all {len(sample_df)} samples from custom CSV for CSV-based train/validation split")
                else:
                    # Truncate to 45k if CSV has fewer than 50k samples
                    sample_df = sample_df.head(45000)
                    print(f"Using first 45,000 samples from custom CSV for training (out of {total_samples} total)")
            
            order_type = self.predetermined_sample_order if not self.predetermined_sample_csv_path else "custom"
            print(f"Loaded {len(sample_df)} samples with predetermined order ({order_type})")
            
            # Store the sample IDs in their predetermined order
            self.predetermined_sample_indices = sample_df['sample_id'].tolist()
            
            print(f"Sample order range: memo scores {sample_df['memorization_score'].min():.6f} to {sample_df['memorization_score'].max():.6f}")
            print(f"First 10 samples: {self.predetermined_sample_indices[:10]}")
            
        except Exception as e:
            print(f"Error loading predetermined sample order: {e}")
            raise

    def get_current_task_epochs(self):
        """Calculate epochs for current task based on mode."""
        if self.incremental_epochs:
            # Incremental epochs: base_epochs * (task_number + 1)
            # If base is 200: task1=200, task2=400, task3=600, ...
            # If base is 300: task1=300, task2=600, task3=900, ...
            return self.epochs_per_task * (self.current_task + 1)
        else:
            # Fixed epochs per task
            return self.epochs_per_task

    def order_samples_by_memo_score(self, indices):
        """Order the indices by memorization score according to memo_order or predetermined order."""
        
        # If no ordering is requested, return indices as-is (will be shuffled later)
        if self.no_ordering:
            return indices
            
        # If predetermined sample order is specified, use that instead of memo-based ordering
        if self.predetermined_sample_order and self.predetermined_sample_indices:
            return self._apply_predetermined_sample_order(indices)
            
        # Create list of (index, memo_score) tuples
        indexed_scores = []
        for idx in indices:
            sample_id = idx.item()
            if sample_id in self.sample_memo_scores:
                memo_score = self.sample_memo_scores[sample_id]
                indexed_scores.append((idx, memo_score))
            else:
                # If no memo score found, assign neutral score
                indexed_scores.append((idx, 0.5))
        
        # Sort by memo score based on ordering strategy
        if self.memo_order == "low_to_high":
            indexed_scores.sort(key=lambda x: x[1])  # ascending order
        elif self.memo_order == "high_to_low":
            indexed_scores.sort(key=lambda x: x[1], reverse=True)  # descending order
        elif self.memo_order == "alternating_hardest_easiest":
            # Sort by memo score to get ordered lists
            sorted_by_score = sorted(indexed_scores, key=lambda x: x[1])
            easiest = sorted_by_score  # low memo scores (easiest)
            hardest = sorted_by_score[::-1]  # high memo scores (hardest)
            
            # Alternate between hardest and easiest
            alternating = []
            for i in range(len(indexed_scores)):
                if i % 2 == 0:
                    alternating.append(hardest[i // 2])  # pick from hardest first
                else:
                    alternating.append(easiest[i // 2])  # then from easiest
            indexed_scores = alternating
        elif self.memo_order == "middle_out":
            # Sort by memo score
            sorted_by_score = sorted(indexed_scores, key=lambda x: x[1])
            
            # Find samples closest to 0.5 memo score
            middle_distances = [(abs(score - 0.5), idx, score) for idx, score in sorted_by_score]
            middle_distances.sort(key=lambda x: x[0])  # sort by distance from 0.5
            
            # Start with middle samples, then alternate easier/harder
            middle_out = []
            used_indices = set()
            
            # First, add the sample closest to 0.5
            if middle_distances:
                middle_out.append((middle_distances[0][1], middle_distances[0][2]))
                used_indices.add(middle_distances[0][1])
            
            # Create lists of easier and harder samples relative to current position
            for i in range(1, len(indexed_scores)):
                # Find next easiest unused sample
                easiest_candidate = None
                hardest_candidate = None
                
                for _, idx, score in middle_distances:
                    if idx not in used_indices:
                        if easiest_candidate is None or score < easiest_candidate[1]:
                            easiest_candidate = (idx, score)
                        if hardest_candidate is None or score > hardest_candidate[1]:
                            hardest_candidate = (idx, score)
                
                # Alternate between easier and harder
                if i % 2 == 1 and easiest_candidate:  # odd positions get easier
                    middle_out.append(easiest_candidate)
                    used_indices.add(easiest_candidate[0])
                elif i % 2 == 0 and hardest_candidate:  # even positions get harder
                    middle_out.append(hardest_candidate)
                    used_indices.add(hardest_candidate[0])
                else:
                    # Fallback: add any remaining unused sample
                    for _, idx, score in middle_distances:
                        if idx not in used_indices:
                            middle_out.append((idx, score))
                            used_indices.add(idx)
                            break
            
            indexed_scores = middle_out
        else:
            valid_orders = ["low_to_high", "high_to_low", "alternating_hardest_easiest", "middle_out"]
            raise ValueError(f"Invalid memo_order: {self.memo_order}. Must be one of {valid_orders}")
        
        # Extract sorted indices
        sorted_indices = torch.stack([item[0] for item in indexed_scores])
        
        return sorted_indices

    def _extract_actual_training_order(self, dataloader, original_indices, is_shuffled):
        """
        Extract the actual sample order that will be used during training.
        
        :param dataloader: The DataLoader object
        :param original_indices: The pre-shuffle indices 
        :param is_shuffled: Whether DataLoader is using shuffle=True
        :return: Tensor of indices in actual training order
        """
        if not is_shuffled:
            # If no shuffling, the order is exactly the original_indices
            print("DEBUG: No shuffling - using original order for sample saving")
            return original_indices
        
        # For shuffled DataLoader, we need to extract the actual order
        print("DEBUG: Extracting actual shuffled training order from DataLoader")
        
        # Use the same random state as the training will use
        # Get the sampler from DataLoader and extract its order
        sampler = dataloader.sampler
        
        try:
            # Get the indices in the order they will be used
            if hasattr(sampler, '__iter__'):
                # Create a fresh sampler with the same random state
                # to get the exact same order that training will use
                generator = torch.Generator()
                generator.manual_seed(int(self.random_seed))  # Use actual experiment seed (convert to int)
                
                # Create a new instance of the same sampler type to get the order
                if isinstance(sampler, torch.utils.data.RandomSampler):
                    # For RandomSampler, generate the same permutation
                    n_samples = len(original_indices)
                    indices = torch.randperm(n_samples, generator=generator).tolist()
                    # Map the shuffled indices to the actual dataset sample IDs
                    actual_sample_indices = [original_indices[i].item() for i in indices]
                else:
                    # Fall back to iterating through the sampler
                    # Make sure we don't exceed the bounds of original_indices
                    actual_sample_indices = []
                    for i in sampler:
                        if i < len(original_indices):
                            actual_sample_indices.append(original_indices[i].item())
                        else:
                            print(f"WARNING: Sampler index {i} exceeds original_indices length {len(original_indices)}")
                            break
                
                result = torch.tensor(actual_sample_indices, dtype=torch.int32)
                print(f"DEBUG: Extracted {len(result)} samples in actual training order")
                print(f"DEBUG: First 10 actual training samples: {result[:10]}")
                print(f"DEBUG: Sampler type: {type(sampler)}")
                return result
                
        except Exception as e:
            print(f"Warning: Could not extract actual training order: {e}")
            print("DEBUG: Falling back to original indices")
            return original_indices
        
        # Fallback
        print("DEBUG: Could not determine actual order, falling back to original indices")
        return original_indices

    def save_complete_50k_order(self, complete_order, class_lookup, memo_lookup, task_id, epoch=None):
        """
        Save the complete 50k sample order (45k training + 5k validation) to CSV.
        
        :param complete_order: Tensor of 50k sample IDs in order (training first, then validation)
        :param class_lookup: Dict mapping sample_id -> class_label
        :param memo_lookup: Dict mapping sample_id -> memorization_score
        :param task_id: Current task identifier
        :param epoch: Optional epoch number for filename
        """
        try:
            # Create the filename
            if epoch is not None:
                filename = f"actual_sample_order_task_{task_id}_epoch_{epoch}.csv"
            else:
                filename = f"actual_sample_order_task_{task_id}.csv"
            
            filepath = os.path.join(self.results_dir, filename)
            
            print(f"DEBUG: Saving complete 50k sample order to: {filepath}")
            print(f"DEBUG: Total samples to save: {len(complete_order)}")
            
            # Create the sample data
            sample_data = []
            training_samples = 45000  # First 45k are training samples
            
            for idx, sample_id in enumerate(complete_order.numpy()):
                sample_id = int(sample_id)
                
                # Get class label and memo score from lookups
                class_label = class_lookup.get(sample_id, -1)  # -1 if not found
                memo_score = memo_lookup.get(sample_id, 0.0)   # 0.0 if not found
                
                # Determine if this is training or validation sample
                is_training = idx < training_samples
                split_type = "train" if is_training else "validation"
                
                sample_data.append({
                    'sample_id': sample_id,
                    'class_label': class_label,
                    'memorization_score': memo_score
                })
            
            # Save to CSV in exact sample_class_map.csv format
            import pandas as pd
            df = pd.DataFrame(sample_data)
            df.to_csv(filepath, index=False)
            
            print(f"Successfully saved complete sample order to: {filepath}")
            print(f"Total samples saved: {len(sample_data)}")
            print(f"Training samples: {training_samples} (rows 0-{training_samples-1})")
            print(f"Validation samples: {len(sample_data) - training_samples} (rows {training_samples}-{len(sample_data)-1})")
            print(f"First 5 training samples: {[x['sample_id'] for x in sample_data[:5]]}")
            print(f"First 5 validation samples: {[x['sample_id'] for x in sample_data[training_samples:training_samples+5]]}")
            print(f"Format: Exactly matches sample_class_map.csv (sample_id,class_label,memorization_score)")
            
        except Exception as e:
            print(f"Error saving complete 50k sample order: {e}")
            import traceback
            traceback.print_exc()

    def save_sample_order(self, ordered_indices, original_cifar_data, task_id, epoch=None):
        """
        Save the actual sample order used during training to a CSV file.
        Compatible with existing sample_class_map.csv format.
        
        :param ordered_indices: The ordered indices of samples (refers to original dataset indices)
        :param original_cifar_data: The original CIFAR dataset before subsampling
        :param task_id: Current task identifier
        :param epoch: Optional epoch number for filename
        """
        try:
            # Create the filename
            if epoch is not None:
                filename = f"actual_sample_order_task_{task_id}_epoch_{epoch}.csv"
            else:
                filename = f"actual_sample_order_task_{task_id}.csv"
            
            filepath = os.path.join(self.results_dir, filename)
            
            # Try to load existing memorization scores from sample_class_map.csv
            memo_scores_map = {}
            try:
                memo_df = pd.read_csv('sample_class_map.csv')
                memo_scores_map = dict(zip(memo_df['sample_id'], memo_df['memorization_score']))
                print(f"Loaded {len(memo_scores_map)} memorization scores from sample_class_map.csv")
            except Exception as e:
                print(f"Warning: Could not load memorization scores from sample_class_map.csv: {e}")
            
            # Extract sample information in compatible format
            sample_data = []
            for training_order_idx, original_sample_idx in enumerate(ordered_indices.numpy()):
                original_sample_idx = int(original_sample_idx)
                
                # Get the class label for this sample from the original dataset
                class_labels = original_cifar_data.data["labels"][original_sample_idx]  # One-hot encoded
                class_id = torch.argmax(class_labels).item()
                
                # Get memorization score from loaded map
                memo_score = memo_scores_map.get(original_sample_idx, 0.0)  # Default to 0.0 if not found
                
                # Use compatible format: actual CIFAR dataset index as sample_id
                sample_data.append({
                    'sample_id': original_sample_idx,  # Use actual dataset index for compatibility
                    'class_label': class_id,
                    'memorization_score': memo_score,
                    'training_order': training_order_idx  # Keep track of training order
                })
            
            # Save to CSV with compatible column order
            df = pd.DataFrame(sample_data)
            # Reorder columns to match existing format (plus training_order for reference)
            df = df[['sample_id', 'class_label', 'memorization_score', 'training_order']]
            df.to_csv(filepath, index=False)
            
            print(f"Saved actual sample order to: {filepath}")
            print(f"Total samples saved: {len(sample_data)}")
            print(f"Format: Compatible with sample_class_map.csv (sample_id = actual dataset indices)")
            if len(sample_data) > 0:
                print(f"Sample ID range: {df['sample_id'].min()} - {df['sample_id'].max()}")
                print(f"First 5 sample IDs in training order: {df.head()['sample_id'].tolist()}")
                print(f"Memo score range: {df['memorization_score'].min():.4f} - {df['memorization_score'].max():.4f}")
                
        except Exception as e:
            print(f"Warning: Could not save sample order: {e}")

    def _apply_predetermined_sample_order(self, indices):
        """Apply predetermined sample order from loaded CSV file."""
        print(f"Applying predetermined sample order ({self.predetermined_sample_order})")
        
        # Convert indices to sample IDs
        sample_ids = [idx.item() for idx in indices]
        available_samples = set(sample_ids)
        
        # Create mapping from sample_id to index tensor
        sample_to_tensor = {sample_id: idx for sample_id, idx in zip(sample_ids, indices)}
        
        # Order indices according to predetermined sample order
        # Only include samples that are actually available in the current indices
        ordered_indices = []
        for sample_id in self.predetermined_sample_indices:
            if sample_id in available_samples:
                ordered_indices.append(sample_to_tensor[sample_id])
        
        # Add any remaining samples not in predetermined order at the end
        predetermined_set = set(self.predetermined_sample_indices)
        for sample_id in sample_ids:
            if sample_id not in predetermined_set:
                ordered_indices.append(sample_to_tensor[sample_id])
        
        # Convert back to tensor
        if ordered_indices:
            result = torch.stack(ordered_indices)
            print(f"Applied predetermined order: {len(result)} samples ordered")
            print(f"DEBUG: First 10 reordered indices: {result[:10]}")
            print(f"DEBUG: Original first 10: {indices[:10]}")
            
            # Check if reordering actually changed the order
            reordering_successful = not torch.equal(result[:10], indices[:10])
            print(f"DEBUG: Reordering successful: {reordering_successful}")
            
            # Set flag to indicate whether reordering worked
            self.predetermined_reordering_successful = reordering_successful
            
            return result
        else:
            print("Warning: No samples matched predetermined order, returning original indices")
            self.predetermined_reordering_successful = False
            return indices

    def _set_adaptive_running_window(self, training_data_size):
        """
        Set running average window adaptively based on training data size.
        Ensures consistent logging across different dataset sizes.
        """
        batch_size = self.batch_sizes["train"]
        steps_per_epoch = max(1, training_data_size // batch_size)
        
        # Use 10% of steps per epoch, but ensure it's between 5 and 50 steps
        adaptive_window = max(5, min(50, steps_per_epoch // 10))
        
        self.running_avg_window = adaptive_window
        self._print(f"\tAdaptive running window set to {self.running_avg_window} steps "
                   f"(dataset: {training_data_size}, batch: {batch_size}, steps/epoch: {steps_per_epoch})")

    # ------------------------------ Methods for initializing the experiment ------------------------------#
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        self.results_dict = {}
        summary_names = ["train_loss_per_checkpoint", "train_accuracy_per_checkpoint", 
                         "train_loss_per_epoch", "train_accuracy_per_epoch", "epoch_runtime",
                         "test_evaluation_runtime", "test_loss_per_epoch", "test_accuracy_per_epoch",
                         "validation_evaluation_runtime", "validation_loss_per_epoch", "validation_accuracy_per_epoch",
                         "learning_rate_per_epoch", "train_samples_per_epoch", "test_samples_per_epoch"]

        for s in summary_names:
            self.results_dict[s] = torch.zeros(size=(self.num_epochs,), dtype=torch.float32, device=self.device)

    def inject_noise(self):
        """
        Injects noise into the network parameters
        """
        if self.perturb_weights_indicator:
            with torch.no_grad():
                for param in self.net.parameters():
                    param.add_(torch.randn(param.size()).to(param.device) * self.noise_std)

    def set_lr(self):
        """ Changes the learning rate of the optimizer according to the current epoch of the task """
        current_stepsize = None
        if (self.current_epoch % self.class_increase_frequency) == 0:
            current_stepsize = self.stepsize
        elif (self.current_epoch % self.class_increase_frequency) == 60:
            current_stepsize = round(self.stepsize * 0.2, 5)
        elif (self.current_epoch % self.class_increase_frequency) == 120:
            current_stepsize = round(self.stepsize * (0.2 ** 2), 5)
        elif (self.current_epoch % self.class_increase_frequency) == 160:
            current_stepsize = round(self.stepsize * (0.2 ** 3), 5)

        if current_stepsize is not None:
            for param_group in self.optim.param_groups:
                param_group['lr'] = current_stepsize
            self._print("\tCurrent stepsize: {0:.5f}".format(current_stepsize))
        
        # Store learning rate for summary (using current lr from param_groups)
        current_lr = self.optim.param_groups[0]['lr']
        self.results_dict["learning_rate_per_epoch"][self.current_epoch] += current_lr

    def increase_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet):
        """
        Increases the number of classes according to the predetermined schedule
        """
        # For single task mode, check if we've reached the epoch limit for this task
        if self.max_tasks == 1:
            # Check if we've completed the required epochs for this single task
            if (self.current_epoch + 1) >= self.get_current_task_epochs():
                self._print(f"\tCompleted {self.get_current_task_epochs()} epochs for single task {self.current_task}. Stopping training.")
                return True  # Signal to stop training
            return False  # Continue training without class changes
            
        if (self.current_epoch + 1) % self.class_increase_frequency == 0:
            # Check if we've reached the maximum number of tasks
            if self.current_task >= self.start_task + self.max_tasks - 1:
                self._print(f"\tReached maximum tasks ({self.max_tasks}). Stopping training.")
                return True  # Signal to stop training
            
            self.current_num_classes += 5
            self.current_num_classes = min(self.current_num_classes, self.num_classes)
            self._print(f"\t[GPU {self.device.index if hasattr(self.device, 'index') else 'N/A'}] Adding 5 new classes. Moving to Task {self.current_task + 1}. Total classes: {self.current_num_classes}")
            
            # Move to next task
            self.current_task += 1
            
            # If scratch mode is enabled, reinitialize the entire network
            if self.scratch:
                self._print(f"\t[GPU {self.device.index if hasattr(self.device, 'index') else 'N/A'}] SCRATCH MODE: Reinitializing network for Task {self.current_task}")
                self.net.apply(kaiming_init_resnet_module)
                # Reinitialize optimizer to reset momentum and other state
                self.optim = torch.optim.SGD(self.net.parameters(), lr=self.stepsize, 
                                           momentum=self.momentum, weight_decay=self.weight_decay)
                # Reinitialize CBP if used
                if self.use_cbp:
                    self.resgnt = ResGnT(net=self.net,
                                       hidden_activation="relu",
                                       replacement_rate=self.replacement_rate,
                                       decay_rate=0.99,
                                       util_type=self.utility_function,
                                       maturity_threshold=self.maturity_threshold,
                                       device=self.device)
            else:
                # Original behavior
                if self.reset_head:
                    self._reset_head()
                if self.reset_network:
                    self._reset_network()
            
            # Update epochs per task for next task
            self.class_increase_frequency = self.get_current_task_epochs()
            self._print(f"\tNext task will run for {self.class_increase_frequency} epochs")

            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])
            
            # Reset running averages when new classes are added to prevent accuracy calculation bugs
            self.running_loss = 0.0
            self.running_accuracy = 0.0
            self.running_samples = 0
            self.running_batches = 0
            
            # Reset epoch accumulators as well
            self.epoch_loss_accumulator = 0.0
            self.epoch_accuracy_accumulator = 0.0
            self.epoch_batches_accumulator = 0
            
            # Reset current_running_avg_step to align with the new task's epoch range
            # This ensures training checkpoints are stored at the correct array indices
            self.current_running_avg_step = self.current_epoch
            
            # Update adaptive window for new dataset size
            new_training_data_size = len(training_data.data["data"])
            self._set_adaptive_running_window(new_training_data_size)
            
            self._save_model_parameters()
            
        return False  # Continue training

    def _reset_head(self):
        """
        Resets the last layer of the network
        """
        self.net.fc.reset_parameters()

    def _reset_network(self):
        """
        Resets the entire network
        """
        self.net.apply(kaiming_init_resnet_module)

    def _save_model_parameters(self):
        """Save model parameters at current epoch."""
        model_params_dir = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_params_dir, exist_ok=True)
        
        if self.predetermined_sample_order:
            model_path = os.path.join(model_params_dir, f"checkpoint_predetermined_{self.predetermined_sample_order}_task-{self.current_task}_epoch-{self.current_epoch}.pt")
        elif self.no_ordering:
            model_path = os.path.join(model_params_dir, f"checkpoint_no_ordering_task-{self.current_task}_epoch-{self.current_epoch}.pt")
        else:
            model_path = os.path.join(model_params_dir, f"checkpoint_{self.memo_order}_task-{self.current_task}_epoch-{self.current_epoch}.pt")
        torch.save(self.net.state_dict(), model_path)

    def create_experiment_checkpoint(self):
        """
        Creates a checkpoint of the experiment
        """
        experiment_checkpoint = {
            "model_weights": self.net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.results_dict.items()}
        }

        if self.use_cbp:
            experiment_checkpoint["resgnt"] = self.resgnt

        if self.predetermined_sample_order:
            checkpoint_file_name = "task-{0}_predetermined_{1}_{2}-{3}.p".format(self.current_task, self.predetermined_sample_order, self.checkpoint_identifier_name, self.current_epoch)
        elif self.no_ordering:
            checkpoint_file_name = "task-{0}_no_ordering_{1}-{2}.p".format(self.current_task, self.checkpoint_identifier_name, self.current_epoch)
        else:
            checkpoint_file_name = "task-{0}_{1}_{2}-{3}.p".format(self.current_task, self.memo_order, self.checkpoint_identifier_name, self.current_epoch)
        experiment_checkpoint_file_path = os.path.join(self.experiment_checkpoints_dir_path, checkpoint_file_name)
        store_object_with_several_attempts(experiment_checkpoint, experiment_checkpoint_file_path, storing_format="pickle", num_attempts=10)

        if self.delete_old_checkpoints and self.current_epoch > self.checkpoint_save_frequency:
            if self.predetermined_sample_order:
                old_checkpoint_file_name = "task-{0}_predetermined_{1}_{2}-{3}.p".format(self.current_task, self.predetermined_sample_order, self.checkpoint_identifier_name, 
                                                                        self.current_epoch - self.checkpoint_save_frequency)
            elif self.no_ordering:
                old_checkpoint_file_name = "task-{0}_no_ordering_{1}-{2}.p".format(self.current_task, self.checkpoint_identifier_name, 
                                                                        self.current_epoch - self.checkpoint_save_frequency)
            else:
                old_checkpoint_file_name = "task-{0}_{1}_{2}-{3}.p".format(self.current_task, self.memo_order, self.checkpoint_identifier_name, 
                                                                        self.current_epoch - self.checkpoint_save_frequency)
            old_checkpoint_file_path = os.path.join(self.experiment_checkpoints_dir_path, old_checkpoint_file_name)
            if os.path.exists(old_checkpoint_file_path):
                os.remove(old_checkpoint_file_path)

    def load_experiment_checkpoint(self):
        """
        Loads a checkpoint of the experiment if one is available
        :return: None
        """
        # Build checkpoint prefix based on experiment type
        if self.no_ordering:
            checkpoint_prefix = f"task-{self.current_task}_no_ordering"
        else:
            checkpoint_prefix = f"task-{self.current_task}_{self.memo_order}"

        # find the most recent checkpoint for this task
        if os.path.exists(self.experiment_checkpoints_dir_path):
            checkpoint_files = [f for f in os.listdir(self.experiment_checkpoints_dir_path) if f.startswith(checkpoint_prefix)]
            if checkpoint_files:
                checkpoint_epochs = [int(f.split('-')[-1].split('.')[0]) for f in checkpoint_files]
                latest_epoch = max(checkpoint_epochs)
                checkpoint_file_name = f"{checkpoint_prefix}_{self.checkpoint_identifier_name}-{latest_epoch}.p"
                checkpoint_file_path = os.path.join(self.experiment_checkpoints_dir_path, checkpoint_file_name)
                self._load_variables_from_checkpoint(checkpoint_file_path)
                print(f"Loaded checkpoint from task {self.current_task}, epoch {latest_epoch}")

    def _load_variables_from_checkpoint(self, file_path: str):
        """
        Loads the variables from a checkpoint
        :param file_path: the path to the checkpoint
        :return: (bool) if the variables were succesfully loaded
        """

        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        self.net.load_state_dict(checkpoint["model_weights"])
        self.optim.load_state_dict(checkpoint["optim_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        self.current_epoch = checkpoint["epoch_number"]
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else partial_results[k].to(self.device)

        if self.use_cbp:
            self.resgnt = checkpoint["resgnt"]

    # --------------------------------------- For storing summaries --------------------------------------- #
    def _store_training_summaries(self):
        # Calculate mathematically correct averages
        if self.current_running_avg_step < self.num_epochs and self.running_batches > 0:
            avg_loss = self.running_loss / self.running_batches
            avg_accuracy = self.running_accuracy / self.running_batches  # Use actual number of batches, not window size
            
            # Debug the calculation in detail
            if not (0.0 <= avg_accuracy <= 1.0):
                self._print(f"\t\tDEBUG: IMPOSSIBLE MATH DETECTED!")
                self._print(f"\t\t  running_accuracy = {self.running_accuracy:.6f}")
                self._print(f"\t\t  running_batches = {self.running_batches}")
                self._print(f"\t\t  running_samples = {self.running_samples}")
                self._print(f"\t\t  avg_accuracy = {self.running_accuracy:.6f} / {self.running_batches} = {avg_accuracy:.6f}")
                self._print(f"\t\t  window_size = {self.running_avg_window}")
                self._print(f"\t\tThis should be mathematically impossible if batch accuracies are in [0,1]!")
                # Don't cap - let it crash to find the real bug
                raise ValueError(f"Invalid accuracy calculation: {avg_accuracy:.6f}")
            
            # If we get here, the calculation is valid
            
            # Store the mathematically correct values
            self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] = avg_loss
            self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] = avg_accuracy
            
            self._print(f"\t\tOnline accuracy: {avg_accuracy:.4f} (from {self.running_samples} samples, {self.running_batches} batches)")
        
        # Reset running averages
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.running_samples = 0
        self.running_batches = 0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_data: DataLoader, val_data: DataLoader, epoch_number: int, epoch_runtime: float, train_samples_count: int):
        """ Computes test summaries and stores them in results dir """

        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32, device=self.device)
        self.results_dict["train_samples_per_epoch"][epoch_number] += torch.tensor(train_samples_count, dtype=torch.float32, device=self.device)

        self.net.eval()
        test_samples_count = 0
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            # evaluate on data
            evaluation_start_time = time.perf_counter()
            # Pass epoch and task info for test prediction tracking (only for test data)
            if data_name == "test":
                loss, accuracy = self.evaluate_network(data_loader, epoch=epoch_number, task_id=self.current_task)
            else:
                loss, accuracy = self.evaluate_network(data_loader)
            evaluation_time = time.perf_counter() - evaluation_start_time

            # count test samples (only for test data, not validation)
            if data_name == "test":
                for batch in data_loader:
                    test_samples_count += batch["image"].shape[0]

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_accuracy_model_parameters = deepcopy(self.net.state_dict())

            # store summaries
            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_time, dtype=torch.float32, device=self.device)
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy

            # print progress
            self._print("\t\t{0} accuracy: {1:.4f}".format(data_name, accuracy))

        self.results_dict["test_samples_per_epoch"][epoch_number] += torch.tensor(test_samples_count, dtype=torch.float32, device=self.device)
        self.net.train()
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def _save_individual_results(self):
        """Save results organized by task for easier analysis"""
        import os
        
        # Build experiment identifier
        if self.predetermined_sample_order:
            exp_id = f"predetermined_sample_{self.predetermined_sample_order}"
        elif self.no_ordering:
            exp_id = "no_ordering"
        else:
            exp_id = f"memo_ordered_{self.memo_order}"
        
        # Add scratch and epochs info
        if self.scratch:
            exp_id += "_scratch"
        
        if self.incremental_epochs:
            exp_id += f"_incremental_{self.epochs_per_task}"
        else:
            exp_id += f"_fixed_{self.epochs_per_task}"
        
        # Create results subdirectories organized by task
        results_base = os.path.join(self.results_dir, f"{exp_id}_experiment")
        os.makedirs(results_base, exist_ok=True)
        
        # Save results for each task that was completed
        tasks_completed = list(range(self.start_task, self.start_task + self.max_tasks))
        
        # Calculate cumulative epochs for each task (needed for incremental mode)
        cumulative_epochs = 0
        for task_idx, task_num in enumerate(tasks_completed):
            # Calculate epoch range for this task
            if self.incremental_epochs:
                task_epochs = self.epochs_per_task * (task_num + 1)
            else:
                task_epochs = self.epochs_per_task
            
            start_epoch = cumulative_epochs
            end_epoch = start_epoch + task_epochs
            cumulative_epochs = end_epoch
            
            # Extract results for this task's epochs
            task_results = {}
            for key, values in self.results_dict.items():
                if isinstance(values, torch.Tensor):
                    task_results[key] = values[start_epoch:end_epoch].cpu().numpy()
                else:
                    task_results[key] = values
            
            # Save individual files per task
            task_dir = os.path.join(results_base, f"task_{task_num}")
            os.makedirs(task_dir, exist_ok=True)
            
            # Save individual metrics for easy analysis
            metrics_to_save = [
                "train_loss_per_epoch", "train_accuracy_per_epoch",
                "test_loss_per_epoch", "test_accuracy_per_epoch", 
                "validation_loss_per_epoch", "validation_accuracy_per_epoch"
            ]
            
            for metric_name in metrics_to_save:
                if metric_name in task_results:
                    # Save both numpy and CSV formats
                    metric_path_npy = os.path.join(task_dir, f"{metric_name}_task_{task_num}.npy")
                    metric_path_csv = os.path.join(task_dir, f"{metric_name}_task_{task_num}.csv")
                    
                    np.save(metric_path_npy, task_results[metric_name])
                    
                    # Also save as CSV for easy analysis
                    import pandas as pd
                    metric_df = pd.DataFrame({'Value': task_results[metric_name]})
                    metric_df.to_csv(metric_path_csv, index=False)
            
            # Save all results for this task
            all_results_path = os.path.join(task_dir, f"all_results_task_{task_num}.npy")
            np.save(all_results_path, task_results)
            
            # Save task metadata
            metadata = {
                "task_number": task_num,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "classes_range": [1 + task_num * 5, 5 + task_num * 5],
                "total_classes": 5 + task_num * 5,
                "scratch_mode": self.scratch,
                "epochs_per_task": task_epochs,
                "gpu_id": getattr(self.device, 'index', 'unknown')
            }
            metadata_path = os.path.join(task_dir, f"task_{task_num}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Also save legacy format for compatibility - now save per task
        cumulative_epochs = 0
        for task_idx, task_num in enumerate(tasks_completed):
            if self.incremental_epochs:
                task_epochs = self.epochs_per_task * (task_num + 1)
            else:
                task_epochs = self.epochs_per_task
            
            start_epoch = cumulative_epochs
            end_epoch = start_epoch + task_epochs
            cumulative_epochs = end_epoch
            
            # Save results for this specific task
            task_results_to_save = {}
            for k, v in self.results_dict.items():
                if isinstance(v, torch.Tensor):
                    task_results_to_save[k] = v[start_epoch:end_epoch].cpu().numpy()
                else:
                    task_results_to_save[k] = v
            
            legacy_path = os.path.join(results_base, f"all_results_{exp_id}_task-{task_num}.npy")
            np.save(legacy_path, task_results_to_save)
        
        self._print(f"Results saved by task to {results_base}")

    def evaluate_network(self, test_data: DataLoader, epoch=None, task_id=None):
        """
        Evaluates the network on the test data
        :param test_data: a pytorch DataLoader object
        :param epoch: current epoch number for tracking (optional)
        :param task_id: current task ID for tracking (optional)
        :return: (torch.Tensor) test loss, (torch.Tensor) test accuracy
        """

        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        
        # Track per-sample test predictions if enabled
        if self.save_epoch_orders and epoch is not None and task_id is not None:
            test_predictions_data = []
        
        with torch.no_grad():
            for _, sample in enumerate(test_data):
                images = sample["image"].to(self.device)
                test_labels = sample["label"].to(self.device)
                test_predictions = self.net.forward(images)[:, self.all_classes[:self.current_num_classes]]

                # Calculate correctness for each sample
                correct_predictions = (test_predictions.argmax(axis=1) == test_labels.argmax(axis=1))
                
                # Track per-sample test predictions if enabled
                if self.save_epoch_orders and epoch is not None and task_id is not None:
                    # Get sample IDs if available
                    if 'index' in sample and sample['index'] is not None:
                        sample_indices = sample['index']
                        if hasattr(sample_indices, 'tolist'):
                            sample_ids = sample_indices.tolist()
                        elif hasattr(sample_indices, '__iter__'):
                            sample_ids = list(sample_indices)
                        else:
                            sample_ids = [int(sample_indices)] * len(images)  # Single index for whole batch
                    else:
                        # Fallback: estimate based on batch position
                        batch_size = images.shape[0]
                        base_idx = num_test_batches * batch_size
                        sample_ids = list(range(base_idx, base_idx + batch_size))
                    
                    # Store prediction results
                    for i, (sample_id, is_correct) in enumerate(zip(sample_ids, correct_predictions)):
                        test_predictions_data.append({
                            'epoch': epoch,
                            'task_id': task_id,
                            'sample_id': sample_id,
                            'is_correct': is_correct.item()
                        })

                # Handle loss calculation for both weighted and regular loss
                if self.use_memo_weighted_loss:
                    batch_loss = torch.mean(self.loss(test_predictions, test_labels))  # Average per-sample losses for evaluation
                else:
                    batch_loss = self.loss(test_predictions, test_labels)
                avg_loss += batch_loss
                avg_acc += torch.mean(correct_predictions.to(torch.float32))
                num_test_batches += 1

        # Save test predictions data if enabled
        if self.save_epoch_orders and epoch is not None and task_id is not None and test_predictions_data:
            self._save_test_predictions(test_predictions_data, epoch, task_id)

        return avg_loss / num_test_batches, avg_acc / num_test_batches

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):
        # load data
        training_data, training_dataloader = self.get_data(train=True, validation=False)
        val_data, val_dataloader = self.get_data(train=True, validation=True)
        test_data, test_dataloader = self.get_data(train=False)
        
        # Set adaptive running window based on training data size
        training_data_size = len(training_data.data["data"])
        self._set_adaptive_running_window(training_data_size)
        
        # load checkpoint if one is available
        self.load_experiment_checkpoint()
        # train network
        self.train(train_dataloader=training_dataloader, test_dataloader=test_dataloader, val_dataloader=val_dataloader,
                   test_data=test_data, training_data=training_data, val_data=val_data)
        # store results separately
        self._save_individual_results()

    def get_data(self, train: bool = True, validation: bool = False):
        """
        Loads the data set
        :param train: (bool) indicates whether to load the train (True) or the test (False) data
        :param validation: (bool) indicates whether to return the valid ation set. The validation set is made up of
                           50 examples of each class of whichever set was loaded
        :return: data set, data loader
        """

        """ Loads CIFAR data set """
        cifar_data = CifarDataSet(root_dir=self.data_path,
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

        if not validation:
            transformations.append(RandomHorizontalFlip(p=0.5))
            transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
            transformations.append(RandomRotator(degrees=(0,15)))

        cifar_data.set_transformation(transforms.Compose(transformations))

        if not train:
            # Use all test data without memo filtering for comprehensive evaluation
            # This allows us to measure generalization to the complete test distribution
            batch_size = self.batch_sizes["test"]
            # Create indices for test data (sequential since we don't need special ordering for test)
            test_indices = torch.arange(len(cifar_data))
            indexed_dataset = IndexedDatasetWrapper(cifar_data, test_indices)
            dataloader = DataLoader(indexed_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
            return cifar_data, dataloader

        train_indices, validation_indices = self.get_validation_and_train_indices(cifar_data)
        indices = validation_indices if validation else train_indices
        
        # Order indices by memorization score (only for training data, not validation)
        if validation:
            # For validation data, don't apply predetermined ordering - use standard split
            ordered_indices = indices
        else:
            # For training data, apply memorization-based or predetermined ordering
            # Skip reordering if using CSV-based predetermined order (indices already in correct order)
            if self.predetermined_sample_order and self.predetermined_sample_indices:
                ordered_indices = indices  # Already in CSV order from get_validation_and_train_indices
                self.predetermined_reordering_successful = True  # Set flag to indicate success
            else:
                ordered_indices = self.order_samples_by_memo_score(indices)
        
        # DEBUG: Check if ordering is working correctly
        if not self.no_ordering and len(indices) > 0:
            if self.predetermined_sample_order and self.predetermined_sample_indices:
                print(f"DEBUG: Using CSV-based predetermined order")
                print(f"DEBUG: CSV order indices[:10]: {indices[:10]}")
                print(f"DEBUG: No reordering needed (already in correct CSV order)")
            else:
                print(f"DEBUG: Original indices[:10]: {indices[:10]}")
                print(f"DEBUG: Ordered indices[:10]: {ordered_indices[:10]}")
                print(f"DEBUG: Indices same? {torch.equal(indices, ordered_indices)}")
        
        # Use all samples (unlike the low memo experiment which filters)
        subsample_cifar_data_set(sub_sample_indices=ordered_indices, cifar_data=cifar_data)
        
        batch_size = self.batch_sizes["validation"] if validation else self.batch_sizes["train"]
        # Use shuffle=True when:
        # 1. no_ordering is enabled (like original experiment), or 
        # 2. predetermined ordering failed to actually reorder samples
        shuffle_data = self.no_ordering or (self.predetermined_sample_order and not self.predetermined_reordering_successful)
        
        if shuffle_data and self.predetermined_sample_order and not self.predetermined_reordering_successful:
            print("WARNING: Predetermined sample ordering failed to reorder samples - enabling shuffle to prevent training failure")
        
        # Create DataLoader with indexed dataset wrapper for sample order tracking
        indexed_dataset = IndexedDatasetWrapper(cifar_data, ordered_indices)
        dataloader = DataLoader(indexed_dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=self.num_workers)
        
        # Store sample orders for both training and validation to create complete 50k CSV
        if not validation:
            # Extract actual training order from DataLoader (45k samples)
            actual_training_order = self._extract_actual_training_order(dataloader, ordered_indices, shuffle_data)
            # Store training order globally for later combination
            self._actual_training_order = actual_training_order
            
        else:
            # Store validation order (5k samples)
            self._actual_validation_order = ordered_indices
            
            # Now we have both training and validation orders - save complete 50k CSV
            if hasattr(self, '_actual_training_order'):
                # Combine: 45k training samples + 5k validation samples = 50k total
                complete_order = torch.cat([self._actual_training_order, self._actual_validation_order])
                
                # Load sample_class_map.csv to get class labels for all sample IDs
                try:
                    sample_map_path = "/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/sample_class_map.csv"
                    import pandas as pd
                    sample_df = pd.read_csv(sample_map_path)
                    
                    # Create lookup dictionary for class labels and memo scores
                    class_lookup = dict(zip(sample_df['sample_id'], sample_df['class_label']))
                    memo_lookup = dict(zip(sample_df['sample_id'], sample_df['memorization_score']))
                    
                    self.save_complete_50k_order(complete_order, class_lookup, memo_lookup, self.current_task, self.current_epoch)
                    
                except Exception as e:
                    print(f"Warning: Could not save complete 50k sample order: {e}")
        
        return cifar_data, dataloader

    def get_validation_and_train_indices(self, cifar_data: CifarDataSet):
        """
        Splits the cifar data into validation and train set and returns the indices of each set with respect to the
        original dataset. Uses CSV-based split if predetermined sample order is enabled.
        :param cifar_data: and instance of CifarDataSet
        :return: train and validation indices
        """
        # Check if we should use CSV-based splitting
        if self.predetermined_sample_order and self.predetermined_sample_indices:
            # CSV-based split: first 45k samples from CSV for training, remaining 5k for validation
            train_sample_ids = self.predetermined_sample_indices[:45000]  # First 45k
            val_sample_ids = self.predetermined_sample_indices[45000:]   # Remaining 5k
            
            # Convert sample IDs to tensor indices (convert floats to ints first)
            train_indices = torch.tensor([int(x) for x in train_sample_ids], dtype=torch.int32)
            validation_indices = torch.tensor([int(x) for x in val_sample_ids], dtype=torch.int32)
            
            print(f"CSV-based split: {len(train_indices)} training, {len(validation_indices)} validation samples")
            return train_indices, validation_indices
        
        else:
            # Original class-based approach
            num_val_samples_per_class = 50
            num_train_samples_per_class = 450
            validation_set_size = 5000
            train_set_size = 45000

            validation_indices = torch.zeros(validation_set_size, dtype=torch.int32)
            train_indices = torch.zeros(train_set_size, dtype=torch.int32)
            current_val_samples = 0
            current_train_samples = 0
            for i in range(self.num_classes):
                class_indices = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()
                validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] += class_indices[:num_val_samples_per_class]
                train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] += class_indices[num_val_samples_per_class:]
                current_val_samples += num_val_samples_per_class
                current_train_samples += num_train_samples_per_class

            return train_indices, validation_indices

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
              test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):

        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])
        val_data.select_new_partition(self.all_classes[:self.current_num_classes])
        self._save_model_parameters()

        for e in tqdm(range(self.current_epoch, self.num_epochs)):
            self._print(f"\t[GPU {self.device.index if hasattr(self.device, 'index') else 'N/A'}] [Task {self.current_task}] Epoch {e + 1} (Classes {self.current_num_classes})")
            self.set_lr()

            # Capture random seed state before epoch (for DataLoader shuffling)
            if self.save_epoch_orders:
                random_seed_state = {
                    'torch': torch.get_rng_state(),
                    'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'numpy': np.random.get_state(),
                    'python': random.getstate()
                }
                self.epoch_random_seeds[e] = random_seed_state
                self.epoch_sample_orders[e] = []  # Initialize list for this epoch's sample order

            epoch_start_time = time.perf_counter()
            train_samples_count = 0
            for step_number, sample in enumerate(train_dataloader):
                # sample observationa and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)
                train_samples_count += image.shape[0]

                # reset gradients
                for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

                # compute prediction and loss
                current_features = [] if self.use_cbp else None
                predictions = self.net.forward(image, current_features)[:, self.all_classes[:self.current_num_classes]]
                
                # Apply weighted loss if enabled (either memo-based or predetermined)
                if self.use_memo_weighted_loss or self.use_predetermined_weights:
                    loss_vec = self.loss(predictions, label)  # Per-sample losses
                    
                    # Get sample IDs - for now, use step_number * batch_size + within_batch_idx as proxy
                    batch_size = image.shape[0]
                    sample_weights = torch.ones(batch_size, device=self.device)
                    
                    # Try to get actual sample indices if available
                    if 'index' in sample and sample['index'] is not None:
                        sample_indices = sample['index']
                        if hasattr(sample_indices, 'tolist'):
                            sample_ids = sample_indices.tolist()
                        elif hasattr(sample_indices, '__iter__'):
                            sample_ids = list(sample_indices)
                        else:
                            sample_ids = [int(sample_indices)] * batch_size  # Single index for whole batch
                    else:
                        # Fallback: estimate sample IDs based on current training progress
                        base_idx = step_number * batch_size
                        sample_ids = list(range(base_idx, base_idx + batch_size))
                    
                    # Debug: Print sample dictionary keys and types occasionally
                    if self.use_predetermined_weights and step_number == 0 and e == 0:
                        print(f"DEBUG: Sample dictionary keys: {list(sample.keys())}")
                        for key, value in sample.items():
                            if hasattr(value, 'shape'):
                                print(f"DEBUG: sample['{key}'] shape: {value.shape}, type: {type(value)}")
                            else:
                                print(f"DEBUG: sample['{key}'] type: {type(value)}")
                    
                    # Save sample order for this epoch
                    if self.save_epoch_orders:
                        self.epoch_sample_orders[e].extend(sample_ids)
                    
                    # Calculate weights based on mode
                    if self.use_predetermined_weights:
                        # Use predetermined weights from CSV file with dramatic boosting formula
                        weights_applied = 0
                        for i, sample_id in enumerate(sample_ids):
                            if sample_id in self.predetermined_weights_dict:
                                memo_score = self.predetermined_weights_dict[sample_id]
                                # Convert memo score to weight boost with dramaticity: 1.0 + (dramaticity * memo_score)
                                # dramaticity=1.0: 0.25→1.25x, 0.9→1.9x, 1.0→2.0x (mild)
                                # dramaticity=2.0: 0.25→1.5x, 0.9→2.8x, 1.0→3.0x (moderate)
                                # dramaticity=5.0: 0.25→2.25x, 0.9→5.5x, 1.0→6.0x (extreme)
                                weight = 1.0 + (self.weight_dramaticity * memo_score)
                                sample_weights[i] = weight
                                weights_applied += 1
                        
                        # Debug logging every 50 steps to monitor weight application
                        if step_number % 50 == 0:
                            print(f"DEBUG: Epoch {e}, Step {step_number}: Applied predetermined weights to {weights_applied}/{batch_size} samples")
                            print(f"DEBUG: Sample IDs in batch: {sample_ids[:5]}...{sample_ids[-5:] if len(sample_ids) > 5 else sample_ids}")
                            if weights_applied > 0:
                                applied_weights = sample_weights[sample_weights > 1.0]
                                print(f"DEBUG: Weight range in this batch: {applied_weights.min().item():.2f}x to {applied_weights.max().item():.2f}x")
                            else:
                                print(f"DEBUG: No weights applied - sample IDs don't match CSV tr_idx values")
                        
                        # Show total stats once per epoch
                        if step_number == 0:
                            print(f"DEBUG: CSV contains {len(self.predetermined_weights_dict)} unique sample IDs")
                            csv_sample_ids = list(self.predetermined_weights_dict.keys())
                            print(f"DEBUG: CSV sample ID range: {min(csv_sample_ids)} to {max(csv_sample_ids)}")
                            print(f"DEBUG: Training batch sample ID range: {min(sample_ids)} to {max(sample_ids)}")
                    else:
                        # Use memorization-based threshold weighting
                        for i, sample_id in enumerate(sample_ids):
                            if sample_id in self.memo_scores_dict:
                                memo_score = self.memo_scores_dict[sample_id]
                                if memo_score > self.memo_threshold:
                                    sample_weights[i] = self.high_memo_weight
                    
                    # Apply weighted loss
                    unweighted_loss = loss_vec.mean()
                    current_reg_loss = (sample_weights * loss_vec).mean()
                    
                    # Debug: Show actual loss impact occasionally
                    if self.use_predetermined_weights and step_number % 100 == 0:
                        weight_impact = current_reg_loss / unweighted_loss if unweighted_loss > 0 else 1.0
                        print(f"DEBUG: Loss impact - Unweighted: {unweighted_loss:.4f}, Weighted: {current_reg_loss:.4f}, Ratio: {weight_impact:.3f}x")
                else:
                    current_reg_loss = self.loss(predictions, label)
                    
                    # Still track sample orders even when not using weighted loss
                    if self.save_epoch_orders:
                        batch_size = image.shape[0]
                        if 'index' in sample and sample['index'] is not None:
                            sample_indices = sample['index']
                            if hasattr(sample_indices, 'tolist'):
                                sample_ids = sample_indices.tolist()
                            elif hasattr(sample_indices, '__iter__'):
                                sample_ids = list(sample_indices)
                            else:
                                sample_ids = [int(sample_indices)] * batch_size  # Single index for whole batch
                        else:
                            base_idx = step_number * batch_size
                            sample_ids = list(range(base_idx, base_idx + batch_size))
                        self.epoch_sample_orders[e].extend(sample_ids)
                
                current_loss = current_reg_loss.detach().clone()

                # backpropagate and update weights
                current_reg_loss.backward()
                self.optim.step()
                if self.use_cbp: self.resgnt.gen_and_test(current_features)
                self.inject_noise()

                # store summaries
                self.running_loss += current_loss.item()
                
                # Calculate training accuracy with mathematically correct approach
                with torch.no_grad():
                    correct_predictions = (predictions.argmax(dim=1) == label.argmax(dim=1))
                    batch_accuracy = correct_predictions.float().mean().item()
                    # batch_accuracy is guaranteed to be in [0, 1] by definition
                    
                    # Debug if batch_accuracy is somehow invalid
                    if not (0.0 <= batch_accuracy <= 1.0):
                        self._print(f"\t\tDEBUG: Invalid batch_accuracy {batch_accuracy:.6f} from {correct_predictions.sum().item()}/{correct_predictions.numel()} correct")
                    
                self.running_accuracy += batch_accuracy
                self.running_samples += predictions.size(0)  # Track total samples for verification
                self.running_batches += 1  # Track total batches
                
                # Also accumulate for per-epoch metrics (separate from running avg that gets reset)
                self.epoch_loss_accumulator += current_loss.item()
                self.epoch_accuracy_accumulator += batch_accuracy
                self.epoch_batches_accumulator += 1
                
                if (step_number + 1) % self.running_avg_window == 0:
                    self._store_training_summaries()

            # Store per-epoch training metrics using epoch accumulators
            if self.epoch_batches_accumulator > 0:
                # Use mathematically correct calculation: average of batch accuracies/losses over the entire epoch
                epoch_online_accuracy = self.epoch_accuracy_accumulator / self.epoch_batches_accumulator
                epoch_online_accuracy_capped = min(1.0, max(0.0, epoch_online_accuracy))
                epoch_online_loss = self.epoch_loss_accumulator / self.epoch_batches_accumulator
                
                # Store per-epoch training metrics
                self.results_dict["train_accuracy_per_epoch"][self.current_epoch] = epoch_online_accuracy_capped
                self.results_dict["train_loss_per_epoch"][self.current_epoch] = epoch_online_loss
                
                self._print(f"\t\tEnd-of-epoch train accuracy: {epoch_online_accuracy_capped:.4f} (from {self.epoch_batches_accumulator} batches)")
                self._print(f"\t\tEnd-of-epoch train loss: {epoch_online_loss:.4f}")
                self._print(f"\t\tDEBUG: Stored train metrics for epoch {self.current_epoch}: acc={epoch_online_accuracy_capped:.6f}, loss={epoch_online_loss:.6f}")
                
                # Reset epoch accumulators for next epoch
                self.epoch_loss_accumulator = 0.0
                self.epoch_accuracy_accumulator = 0.0
                self.epoch_batches_accumulator = 0

            # update the current epoch
            self.current_epoch += 1
            epoch_runtime = time.perf_counter() - epoch_start_time

            # store test summaries
            self._store_test_summaries(test_dataloader, val_dataloader, self.current_epoch - 1, epoch_runtime, train_samples_count)

            # save model parameters every epoch
            self._save_model_parameters()

            # increase number of classes and create checkpoints
            should_stop = self.increase_classes(training_data, test_data, val_data)
            if should_stop:
                self._print("Training stopped: reached maximum tasks")
                break
            

            
            if (self.current_epoch % self.checkpoint_save_frequency) == 0:
                self.create_experiment_checkpoint()
                
                # In single task mode, also save individual metric files frequently for analysis
                if self.max_tasks == 1 and self.current_epoch % 10 == 0:  # Save every 10 epochs
                    self._save_individual_results()
                    self._print(f"Intermediate metrics saved at epoch {self.current_epoch}")

            # early stopping check
            if self.early_stopping and self.current_epoch >= 1000:
                recent_accuracies = self.results_dict["validation_accuracy_per_epoch"][self.current_epoch-100:self.current_epoch]
                if len(recent_accuracies) >= 100 and torch.std(recent_accuracies) < 0.001:
                    self._print("Early stopping triggered due to convergence")
                    break

        # Save final checkpoint and metrics (especially important in single task mode)
        if self.max_tasks == 1:
            self.create_experiment_checkpoint()
            self._save_individual_results()  # Save all metrics as individual files
            self._print(f"Final checkpoint and metrics saved for epoch {self.current_epoch}")

        # Save epoch sample orders and random seeds
        self._save_epoch_sample_orders()

        # Save final model
        if self.predetermined_sample_order:
            final_model_path = os.path.join(self.results_dir, "model_parameters", f"final_model_predetermined_{self.predetermined_sample_order}_task-{self.current_task}.pt")
        elif self.no_ordering:
            final_model_path = os.path.join(self.results_dir, "model_parameters", f"final_model_no_ordering_task-{self.current_task}.pt")
        else:
            final_model_path = os.path.join(self.results_dir, "model_parameters", f"final_model_{self.memo_order}_task-{self.current_task}.pt")
        torch.save(self.best_accuracy_model_parameters, final_model_path)
        self._print(f"Final model saved with best validation accuracy: {self.best_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, required=True, help="Path to config file")
    parser.add_argument("--results_dir", "-r", type=str, required=True, help="Path to results directory")
    parser.add_argument("--run_index", "-i", type=int, default=0, help="Run index")
    parser.add_argument("--memo_order", "-o", type=str, default="low_to_high", 
                        choices=["low_to_high", "high_to_low", "alternating_hardest_easiest", "middle_out"], 
                        help="Sample memorization score ordering strategy")
    parser.add_argument("--class_order", "-co", type=str, default="memo_low_to_high",
                        choices=["memo_low_to_high", "memo_high_to_low", "random", "sequential", "predetermined"],
                        help="Class introduction order for tasks")
    parser.add_argument("--within_task_class_order", type=str, default="task_order",
                        choices=["memo_low_to_high", "memo_high_to_low", "task_order"],
                        help="Class ordering within each task when using predetermined class order")
    parser.add_argument("--csv_file_path", type=str, default=None,
                        help="Path to CSV file containing predetermined class order")
    parser.add_argument("--gpu_id", "-g", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--no_ordering", action="store_true", help="Disable memo ordering (shuffle randomly like original experiment)")
    parser.add_argument("--scratch", action="store_true", help="Reinitialize network from scratch for each task")
    parser.add_argument("--epochs_per_task", "-e", type=int, default=200, help="Base epochs per task")
    parser.add_argument("--incremental_epochs", action="store_true", help="Use incremental epochs (base_epochs * task_number)")
    parser.add_argument("--start_task", type=int, default=0, help="Starting task number (for multi-GPU parallelization)")
    parser.add_argument("--max_tasks", type=int, default=20, help="Maximum number of tasks to run")
    parser.add_argument("--predetermined_sample_order", type=str, choices=["ascending", "descending"], help="Use predetermined sample order from CSV files")
    parser.add_argument("--predetermined_sample_csv_path", type=str, help="Custom path to CSV file for predetermined sample order")
    
    # Memorization-aware weighted loss arguments
    parser.add_argument("--use_memo_weighted_loss", action="store_true", help="Enable memorization-aware weighted loss")
    parser.add_argument("--memo_threshold", type=float, default=0.25, help="Memorization score threshold for high-weight samples")
    parser.add_argument("--high_memo_weight", type=float, default=3.0, help="Weight multiplier for high-memorization samples")
    parser.add_argument("--memo_csv_path", type=str, default="", help="Path to CSV file with memorization scores (empty = use sample_class_map.csv)")
    
    # Predetermined sample weights arguments
    parser.add_argument("--use_predetermined_weights", action="store_true", help="Enable predetermined sample weights from CSV")
    parser.add_argument("--predetermined_weights_csv_path", type=str, default="", help="Path to CSV file with predetermined sample weights")
    parser.add_argument("--weight_dramaticity", type=float, default=1.0, help="Dramaticity multiplier for weight boost: weight = 1.0 + (dramaticity * memo_score)")
    
    # Epoch sample order tracking arguments
    parser.add_argument("--save_epoch_orders", action="store_true", help="Save all epoch sample orders and random seeds for analysis")
    
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Set default paths if empty (like original experiment)
    file_path = os.path.dirname(os.path.abspath(__file__))
    if "data_path" not in config.keys() or config["data_path"] == "":
        config["data_path"] = os.path.join(file_path, "data")

    exp = IncrementalCIFARMemoOrderedExperiment(
        exp_params=config, 
        results_dir=args.results_dir, 
        run_index=args.run_index,
        memo_order=args.memo_order,
        gpu_id=args.gpu_id,
        no_ordering=args.no_ordering,
        scratch=args.scratch,
        epochs_per_task=args.epochs_per_task,
        incremental_epochs=args.incremental_epochs,
        start_task=args.start_task,
        max_tasks=args.max_tasks,
        class_order=args.class_order,
        within_task_class_order=args.within_task_class_order,
        csv_file_path=args.csv_file_path,
        predetermined_sample_order=args.predetermined_sample_order,
        predetermined_sample_csv_path=args.predetermined_sample_csv_path,
        use_memo_weighted_loss=args.use_memo_weighted_loss,
        memo_threshold=args.memo_threshold,
        high_memo_weight=args.high_memo_weight,
        memo_csv_path=args.memo_csv_path,
        save_epoch_orders=args.save_epoch_orders,
        use_predetermined_weights=args.use_predetermined_weights,
        predetermined_weights_csv_path=args.predetermined_weights_csv_path,
        weight_dramaticity=args.weight_dramaticity
    )
    exp.run()