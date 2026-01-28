# built-in libraries
import time
import os
import pickle
from copy import deepcopy
import json
import argparse
from functools import partialmethod

# third party libraries
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

# WandB for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import Logger from UPGD codebase for JSON logging
import sys
sys.path.insert(0, '/scratch/gautschi/shin283/upgd')
from core.logger import Logger

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

    cifar_data.data["data"] = cifar_data.data["data"][sub_sample_indices.numpy()]       # .numpy wasn't necessary with torch 2.0
    cifar_data.data["labels"] = cifar_data.data["labels"][sub_sample_indices.numpy()]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[sub_sample_indices.numpy()].tolist()
    cifar_data.current_data = cifar_data.partition_data()


class IncrementalCIFARExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # UPGD parameters
        self.use_upgd = access_dict(exp_params, "use_upgd", default=False, val_type=bool)
        self.upgd_beta_utility = access_dict(exp_params, "upgd_beta_utility", default=0.999, val_type=float)
        self.upgd_sigma = access_dict(exp_params, "upgd_sigma", default=0.001, val_type=float)
        self.upgd_beta1 = access_dict(exp_params, "upgd_beta1", default=0.9, val_type=float)
        self.upgd_beta2 = access_dict(exp_params, "upgd_beta2", default=0.999, val_type=float)
        self.upgd_eps = access_dict(exp_params, "upgd_eps", default=1e-5, val_type=float)
        self.upgd_use_adam_moments = access_dict(exp_params, "upgd_use_adam_moments", default=True, val_type=bool)
        self.upgd_gating_mode = access_dict(exp_params, "upgd_gating_mode", default="full", val_type=str)
        self.upgd_non_gated_scale = access_dict(exp_params, "upgd_non_gated_scale", default=0.5, val_type=float)

        """ Training constants """
        self.num_epochs = 4000
        self.current_num_classes = 5
        self.batch_sizes = {"train": 90, "test": 100, "validation":50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.num_images_per_class = 450

        """ Network set up """
        # initialize network
        self.net = build_resnet18(num_classes=self.num_classes, norm_layer=torch.nn.BatchNorm2d)
        self.net.apply(kaiming_init_resnet_module)

        # initialize optimizer
        if self.use_upgd:
            from lop.algos.upgd import UPGD
            self.optim = UPGD(
                self.net.parameters(),
                lr=self.stepsize,
                weight_decay=self.weight_decay,
                beta_utility=self.upgd_beta_utility,
                sigma=self.upgd_sigma,
                beta1=self.upgd_beta1,
                beta2=self.upgd_beta2,
                eps=self.upgd_eps,
                use_adam_moments=self.upgd_use_adam_moments,
                momentum=self.momentum,
                gating_mode=self.upgd_gating_mode,
                non_gated_scale=self.upgd_non_gated_scale
            )
            # Set parameter names for logging
            self.optim.set_param_names(self.net.named_parameters())
        else:
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
        self.class_increase_frequency = 200
        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_accuracy_model_parameters = {}

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added
        self.delete_old_checkpoints = True

        """ For summaries """
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()

        """ WandB tracking """
        self.use_wandb = access_dict(exp_params, "use_wandb", default=False, val_type=bool)
        self.wandb_initialized = False

        """ JSON logging setup """
        self.json_logger = Logger(log_dir="/scratch/gautschi/shin283/upgd/logs")
        optimizer_name = "upgd" if self.use_upgd else "sgd"
        optimizer_hps = {
            "lr": self.stepsize,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
        }
        if self.use_upgd:
            optimizer_hps.update({
                "beta_utility": self.upgd_beta_utility,
                "sigma": self.upgd_sigma,
                "gating_mode": self.upgd_gating_mode,
            })
        self.json_logger.initialize_log_path(
            task="incremental_cifar",
            learner=optimizer_name,
            network="resnet18",
            optimizer_hps=optimizer_hps,
            seed=self.random_seed,
            n_samples=self.num_epochs
        )

        # Step-level data collection for JSON
        self.json_step_data = {
            "train_loss_per_step": [],
            "train_accuracy_per_step": [],
            "weight_l2_per_step": [],
            "weight_l1_per_step": [],
            "grad_l2_per_step": [],
            "grad_l1_per_step": [],
            "utility_histogram_per_step": {},
            "global_max_utility_per_step": [],
        }
        self.json_step_counter = 0

    # ------------------------------ Methods for initializing the experiment ------------------------------#
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
        class_increase = 5
        number_of_image_per_task = self.num_images_per_class * class_increase
        bin_size = (self.running_avg_window * self.batch_sizes["train"])
        total_checkpoints = int(np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size))

        train_prototype_array = torch.zeros(total_checkpoints, device=self.device, dtype=torch.float32)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros_like(train_prototype_array)

        prototype_array = torch.zeros(self.num_epochs, device=self.device, dtype=torch.float32)
        self.results_dict["epoch_runtime"] = torch.zeros_like(prototype_array)
        # test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = torch.zeros_like(prototype_array)
        self.results_dict["class_order"] = self.all_classes

        # UPGD-specific summaries
        if self.use_upgd:
            self.results_dict["upgd_global_max_utility"] = torch.zeros_like(prototype_array)
            self.results_dict["upgd_mean_utility"] = torch.zeros_like(prototype_array)
            self.results_dict["upgd_utility_sparsity"] = torch.zeros_like(prototype_array)
            self.results_dict["upgd_mean_gate_value"] = torch.zeros_like(prototype_array)
            self.results_dict["upgd_active_fraction"] = torch.zeros_like(prototype_array)

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self):
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "model_weights": self.net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": partial_results
        }

        if self.use_cbp:
            checkpoint["resgnt"] = self.resgnt

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
        :param file_path: path to the experiment checkpoint
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
        # store train data
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_data: DataLoader, val_data: DataLoader, epoch_number: int, epoch_runtime: float):
        """ Computes test summaries and stores them in results dir """

        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)

        self.net.eval()
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            # evaluate on data
            evaluation_start_time = time.perf_counter()
            loss, accuracy = self.evaluate_network(data_loader)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_accuracy_model_parameters = deepcopy(self.net.state_dict())

            # store summaries
            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(evaluation_time, dtype=torch.float32)
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy

            # print progress
            self._print("\t\t{0} accuracy: {1:.4f}".format(data_name, accuracy))

        self.net.train()
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

        # Log to WandB
        if self.use_wandb and self.wandb_initialized:
            wandb_metrics = {
                "epoch": epoch_number + 1,
                "train/loss": float(self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step - 1]) if self.current_running_avg_step > 0 else 0.0,
                "train/accuracy": float(self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step - 1]) if self.current_running_avg_step > 0 else 0.0,
                "test/loss": float(self.results_dict["test_loss_per_epoch"][epoch_number]),
                "test/accuracy": float(self.results_dict["test_accuracy_per_epoch"][epoch_number]),
                "validation/loss": float(self.results_dict["validation_loss_per_epoch"][epoch_number]),
                "validation/accuracy": float(self.results_dict["validation_accuracy_per_epoch"][epoch_number]),
                "epoch_runtime": epoch_runtime,
                "current_num_classes": self.current_num_classes,
                "current_task": self.current_epoch // self.class_increase_frequency,
            }
            wandb.log(wandb_metrics, step=epoch_number)

        # Store UPGD utility statistics
        if self.use_upgd:
            utility_stats = self.optim.get_utility_stats()
            gating_stats = self.optim.get_gating_stats()

            self.results_dict["upgd_global_max_utility"][epoch_number] = torch.tensor(
                utility_stats.get('global_max_utility', 0.0), dtype=torch.float32
            )
            self.results_dict["upgd_mean_utility"][epoch_number] = torch.tensor(
                utility_stats.get('mean_utility', 0.0), dtype=torch.float32
            )
            self.results_dict["upgd_utility_sparsity"][epoch_number] = torch.tensor(
                utility_stats.get('utility_sparsity', 0.0), dtype=torch.float32
            )
            self.results_dict["upgd_mean_gate_value"][epoch_number] = torch.tensor(
                gating_stats.get('mean_gate_value', 0.0), dtype=torch.float32
            )
            self.results_dict["upgd_active_fraction"][epoch_number] = torch.tensor(
                gating_stats.get('active_fraction', 0.0), dtype=torch.float32
            )

            # Print utility statistics
            self._print("\t\tUPGD Gating Mode: {0}".format(gating_stats.get('gating_mode', 'full')))
            self._print("\t\tUPGD Gated Params: {0}, Non-Gated: {1}".format(
                gating_stats.get('gated_params', 0), gating_stats.get('non_gated_params', 0)
            ))
            self._print("\t\tUPGD Global Max Utility: {0:.6f}".format(utility_stats.get('global_max_utility', 0.0)))
            self._print("\t\tUPGD Mean Gate Value: {0:.4f}".format(gating_stats.get('mean_gate_value', 0.0)))
            self._print("\t\tUPGD Active Fraction: {0:.4f}".format(gating_stats.get('active_fraction', 0.0)))

            # Log UPGD stats to WandB
            if self.use_wandb and self.wandb_initialized:
                upgd_metrics = {
                    "upgd/global_max_utility": utility_stats.get('global_max_utility', 0.0),
                    "upgd/mean_utility": utility_stats.get('mean_utility', 0.0),
                    "upgd/utility_sparsity": utility_stats.get('utility_sparsity', 0.0),
                    "upgd/mean_gate_value": gating_stats.get('mean_gate_value', 0.0),
                    "upgd/active_fraction": gating_stats.get('active_fraction', 0.0),
                    "upgd/gated_params": gating_stats.get('gated_params', 0),
                    "upgd/non_gated_params": gating_stats.get('non_gated_params', 0),
                }
                
                # Add per-layer utility statistics
                layer_utilities = utility_stats.get('layer_utilities', {})
                for layer_name, layer_utility in layer_utilities.items():
                    short_name = layer_name.replace('.', '_')
                    upgd_metrics[f"utility_layer/{short_name}"] = layer_utility
                
                wandb.log(upgd_metrics, step=epoch_number)

        # Compute and log plasticity metrics (every 10 epochs to reduce overhead)
        if self.use_wandb and self.wandb_initialized and (epoch_number + 1) % 10 == 0:
            plasticity_metrics = self._compute_plasticity_metrics()
            wandb.log(plasticity_metrics, step=epoch_number)

    def _compute_plasticity_metrics(self):
        """Compute plasticity metrics: dead neurons, weight norms, stable rank, effective rank."""
        metrics = {}
        
        # Weight statistics per layer
        total_params = 0
        total_norm_sq = 0.0
        sranks = []
        eff_ranks = []
        
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                # Weight norms
                l2_norm = param.norm(2).item()
                mean_abs = param.abs().mean().item()
                short_name = name.replace('.', '_')
                metrics[f"weight_norm/{short_name}"] = l2_norm
                
                total_params += param.numel()
                total_norm_sq += l2_norm ** 2
                
                # Stable rank and effective rank for weight matrices
                if 'weight' in name and param.dim() >= 2:
                    try:
                        W = param.view(param.size(0), -1)
                        frobenius_sq = (W ** 2).sum().item()
                        
                        # SVD for stable rank and effective rank
                        U, S, V = torch.linalg.svd(W, full_matrices=False)
                        spectral_sq = (S[0] ** 2).item()
                        
                        # Stable rank
                        srank = frobenius_sq / spectral_sq if spectral_sq > 1e-10 else 0.0
                        metrics[f"srank/{short_name}"] = srank
                        sranks.append(srank)
                        
                        # Effective rank (entropy-based)
                        S_normalized = S / (S.sum() + 1e-10)
                        entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum().item()
                        eff_rank = np.exp(entropy)
                        metrics[f"effective_rank/{short_name}"] = eff_rank
                        eff_ranks.append(eff_rank)
                        
                    except Exception:
                        pass
        
        # Aggregate metrics
        metrics["weight_norm/total"] = np.sqrt(total_norm_sq)
        if sranks:
            metrics["srank/mean"] = np.mean(sranks)
            metrics["srank/min"] = np.min(sranks)
        if eff_ranks:
            metrics["effective_rank/mean"] = np.mean(eff_ranks)
        
        # Dead neurons detection (requires forward pass)
        try:
            dead_stats = self._compute_dead_neurons()
            metrics.update(dead_stats)
        except Exception:
            pass
        
        return metrics

    def _compute_dead_neurons(self, threshold=0.01):
        """Compute fraction of dead neurons per layer using activation hooks."""
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for linear/conv layers
        for name, module in self.net.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Forward pass with dummy input
        dummy_input = torch.randn(16, *self.image_dims[::-1], device=self.device)
        with torch.no_grad():
            _ = self.net(dummy_input)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Compute dead neuron ratios
        stats = {}
        total_dead = 0
        total_neurons = 0
        
        for name, act in activations.items():
            if act.dim() >= 2:
                mean_act = act.abs().mean(dim=0)
                while mean_act.dim() > 1:
                    mean_act = mean_act.mean(dim=-1)
                
                n_neurons = mean_act.numel()
                n_dead = (mean_act < threshold).sum().item()
                
                short_name = name.replace('.', '_')
                stats[f"dead_neurons/{short_name}"] = n_dead / n_neurons if n_neurons > 0 else 0.0
                total_dead += n_dead
                total_neurons += n_neurons
        
        stats["dead_neurons/total_ratio"] = total_dead / total_neurons if total_neurons > 0 else 0.0
        stats["dead_neurons/total_count"] = total_dead
        
        return stats

    def _save_json_checkpoint(self, final=False):
        """Save JSON checkpoint with all collected metrics."""
        # Convert results_dict tensors to lists
        json_data = {}

        # Per-epoch metrics from results_dict
        for key in ["test_loss_per_epoch", "test_accuracy_per_epoch",
                    "validation_loss_per_epoch", "validation_accuracy_per_epoch",
                    "train_loss_per_checkpoint", "train_accuracy_per_checkpoint",
                    "epoch_runtime"]:
            if key in self.results_dict:
                tensor = self.results_dict[key]
                if isinstance(tensor, torch.Tensor):
                    json_data[key] = tensor[:self.current_epoch].cpu().tolist()
                else:
                    json_data[key] = list(tensor)[:self.current_epoch]

        # UPGD-specific per-epoch metrics
        if self.use_upgd:
            for key in ["upgd_global_max_utility", "upgd_mean_utility",
                        "upgd_utility_sparsity", "upgd_mean_gate_value", "upgd_active_fraction"]:
                if key in self.results_dict:
                    tensor = self.results_dict[key]
                    if isinstance(tensor, torch.Tensor):
                        json_data[key] = tensor[:self.current_epoch].cpu().tolist()

        # Step-level data
        json_data.update(self.json_step_data)

        # Add metadata fields directly
        optimizer_name = "upgd" if self.use_upgd else "sgd"
        json_data["task"] = "incremental_cifar"
        json_data["learner"] = optimizer_name
        json_data["network"] = "resnet18"
        json_data["seed"] = self.random_seed
        json_data["current_epoch"] = self.current_epoch
        json_data["total_epochs"] = self.num_epochs
        json_data["current_num_classes"] = self.current_num_classes
        json_data["status"] = "completed" if final else "in_progress"
        if self.use_upgd:
            json_data["upgd_config"] = {
                "beta_utility": self.upgd_beta_utility,
                "sigma": self.upgd_sigma,
                "gating_mode": self.upgd_gating_mode,
            }

        # Save to logger
        self.json_logger.log(**json_data)

        if final:
            print(f"Final JSON log saved to: {self.json_logger.log_path}")

    def evaluate_network(self, test_data: DataLoader):
        """
        Evaluates the network on the test data
        :param test_data: a pytorch DataLoader object
        :return: (torch.Tensor) test loss, (torch.Tensor) test accuracy
        """

        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        with torch.no_grad():
            for _, sample in enumerate(test_data):
                images = sample["image"].to(self.device)
                test_labels = sample["label"].to(self.device)
                test_predictions = self.net.forward(images)[:, self.all_classes[:self.current_num_classes]]

                avg_loss += self.loss(test_predictions, test_labels)
                avg_acc += torch.mean((test_predictions.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))
                num_test_batches += 1

        return avg_loss / num_test_batches, avg_acc / num_test_batches

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):
        # load data
        training_data, training_dataloader = self.get_data(train=True, validation=False)
        val_data, val_dataloader = self.get_data(train=True, validation=True)
        test_data, test_dataloader = self.get_data(train=False)
        # load checkpoint if one is available
        self.load_experiment_checkpoint()
        # train network
        self.train(train_dataloader=training_dataloader, test_dataloader=test_dataloader, val_dataloader=val_dataloader,
                   test_data=test_data, training_data=training_data, val_data=val_data)
        # store results using exp.store_results()

    def get_data(self, train: bool = True, validation: bool = False):
        """
        Loads the data set
        :param train: (bool) indicates whether to load the train (True) or the test (False) data
        :param validation: (bool) indicates whether to return the validation set. The validation set is made up of
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
            batch_size = self.batch_sizes["test"]
            dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
            return cifar_data, dataloader

        train_indices, validation_indices = self.get_validation_and_train_indices(cifar_data)
        indices = validation_indices if validation else train_indices
        subsample_cifar_data_set(sub_sample_indices=indices, cifar_data=cifar_data)
        batch_size = self.batch_sizes["validation"] if validation else self.batch_sizes["train"]
        return cifar_data, DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

    def get_validation_and_train_indices(self, cifar_data: CifarDataSet):
        """
        Splits the cifar data into validation and train set and returns the indices of each set with respect to the
        original dataset
        :param cifar_data: and instance of CifarDataSet
        :return: train and validation indices
        """
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
            self._print("\tEpoch number: {0}".format(e + 1))
            self.set_lr()

            epoch_start_time = time.perf_counter()
            for step_number, sample in enumerate(train_dataloader):
                # sample observationa and target
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                # reset gradients
                for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

                # compute prediction and loss
                current_features = [] if self.use_cbp else None
                predictions = self.net.forward(image, current_features)[:, self.all_classes[:self.current_num_classes]]
                current_reg_loss = self.loss(predictions, label)
                current_loss = current_reg_loss.detach().clone()

                # backpropagate and update weights
                current_reg_loss.backward()
                self.optim.step()
                if self.use_cbp: self.resgnt.gen_and_test(current_features)
                self.inject_noise()

                # store summaries
                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()
                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

                # Collect step-level data for JSON (every 10 steps to reduce overhead)
                if self.json_step_counter % 10 == 0:
                    self.json_step_data["train_loss_per_step"].append(float(current_loss.item()))
                    self.json_step_data["train_accuracy_per_step"].append(float(current_accuracy.item()))

                    # Weight/gradient statistics
                    total_weight_l2 = 0.0
                    total_weight_l1 = 0.0
                    total_grad_l2 = 0.0
                    total_grad_l1 = 0.0
                    for param in self.net.parameters():
                        if param.requires_grad:
                            total_weight_l2 += param.norm(2).item() ** 2
                            total_weight_l1 += param.abs().sum().item()
                            if param.grad is not None:
                                total_grad_l2 += param.grad.norm(2).item() ** 2
                                total_grad_l1 += param.grad.abs().sum().item()
                    self.json_step_data["weight_l2_per_step"].append(float(np.sqrt(total_weight_l2)))
                    self.json_step_data["weight_l1_per_step"].append(float(total_weight_l1))
                    self.json_step_data["grad_l2_per_step"].append(float(np.sqrt(total_grad_l2)))
                    self.json_step_data["grad_l1_per_step"].append(float(total_grad_l1))

                    # UPGD-specific stats
                    if self.use_upgd and hasattr(self.optim, 'get_utility_stats'):
                        utility_stats = self.optim.get_utility_stats()
                        if 'global_max_utility' in utility_stats:
                            self.json_step_data["global_max_utility_per_step"].append(float(utility_stats['global_max_utility']))
                        # Collect histogram bins
                        for key, value in utility_stats.items():
                            if 'hist_' in key and isinstance(value, (int, float)):
                                if key not in self.json_step_data["utility_histogram_per_step"]:
                                    self.json_step_data["utility_histogram_per_step"][key] = []
                                self.json_step_data["utility_histogram_per_step"][key].append(float(value))

                self.json_step_counter += 1

            epoch_end_time = time.perf_counter()
            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=e,
                                       epoch_runtime=epoch_end_time - epoch_start_time)

            self.current_epoch += 1
            self.extend_classes(training_data, test_data, val_data)

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

            # Save JSON checkpoint every 50 epochs
            if self.current_epoch % 50 == 0:
                self._save_json_checkpoint()

        # Final JSON save at end of training
        self._save_json_checkpoint(final=True)


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
            for g in self.optim.param_groups:
                g['lr'] = current_stepsize
            self._print("\tCurrent stepsize: {0:.5f}".format(current_stepsize))

    def inject_noise(self):
        """
        Adds a small amount of random noise to the parameters of the network
        """
        if not self.perturb_weights_indicator: return

        with torch.no_grad():
            for param in self.net.parameters():
                param.add_(torch.randn(param.size(), device=param.device) * self.noise_std)

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet):
        """
        Adds one new class to the data set with certain frequency
        """
        if (self.current_epoch % self.class_increase_frequency) == 0:
            self._print("Best accuracy in the task: {0:.4f}".format(self.best_accuracy))
            if self.early_stopping:
                self.net.load_state_dict(self.best_accuracy_model_parameters)
            self.best_accuracy = torch.zeros_like(self.best_accuracy)
            self.best_accuracy_model_parameters = {}
            self._save_model_parameters()

            if self.current_num_classes == self.num_classes: return

            increase = 5
            self.current_num_classes += increase
            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])

            self._print("\tNew class added...")
            if self.reset_head:
                kaiming_init_resnet_module(self.net.fc)
            if self.reset_network:
                self.net.apply(kaiming_init_resnet_module)

    def _save_model_parameters(self):
        """ Stores the parameters of the model, so it can be evaluated after the experiment is over """

        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = "index-{0}_epoch-{1}.pt".format(self.run_index, self.current_epoch)
        file_path = os.path.join(model_parameters_dir_path, file_name)

        store_object_with_several_attempts(self.net.state_dict(), file_path, storing_format="torch", num_attempts=10)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action="store", type=str,
                        default='./incremental_cifar/cfg/base_deep_learning_system.json',
                        help="Path to the file containing the parameters for the experiment.")
    parser.add_argument("--experiment-index", action="store", type=int, default=0,
                        help="Index for the run; this will determine the random seed and the name of the results.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Whether to print extra information about the experiment as it's running.")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Whether to log to Weights & Biases.")
    parser.add_argument("--wandb-project", action="store", type=str, default="upgd-incremental-cifar",
                        help="WandB project name.")
    parser.add_argument("--wandb-entity", action="store", type=str, default=None,
                        help="WandB entity (team) name.")
    parser.add_argument("--wandb-run-name", action="store", type=str, default=None,
                        help="WandB run name.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        experiment_parameters = json.load(config_file)

    file_path = os.path.dirname(os.path.abspath(__file__))
    if "data_path" not in experiment_parameters.keys() or experiment_parameters["data_path"] == "":
        experiment_parameters["data_path"] = os.path.join(file_path, "data")
    if "results_dir" not in experiment_parameters.keys() or experiment_parameters["results_dir"] == "":
        experiment_parameters["results_dir"] = os.path.join(file_path, "results")
    if "experiment_name" not in experiment_parameters.keys() or experiment_parameters["experiment_name"] == "":
        experiment_parameters["experiment_name"] = os.path.splitext(os.path.basename(args.config_file))

    # Add use_wandb to experiment parameters
    experiment_parameters["use_wandb"] = args.wandb and WANDB_AVAILABLE

    # Initialize WandB if requested
    if experiment_parameters["use_wandb"]:
        run_name = args.wandb_run_name or f"{experiment_parameters.get('experiment_name', 'incr_cifar')}_idx{args.experiment_index}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=experiment_parameters,
            name=run_name,
            save_code=True,
        )
        print(f"WandB initialized: {wandb.run.url}")

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(experiment_parameters["results_dir"], experiment_parameters["experiment_name"]),
                                     run_index=args.experiment_index,
                                     verbose=args.verbose)
    
    # Set WandB initialized flag
    if experiment_parameters["use_wandb"]:
        exp.wandb_initialized = True

    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))

    # Finish WandB run
    if experiment_parameters["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
