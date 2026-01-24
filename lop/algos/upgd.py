import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np


class UPGD(torch.optim.Optimizer):
    """
    Utility-based Perturbed Gradient Descent (UPGD) optimizer for continual learning.

    This optimizer tracks parameter utility and applies selective gating to balance
    plasticity and stability. It supports both basic UPGD and adaptive variants with
    Adam-style momentum.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        weight_decay: weight decay coefficient (default: 0.0)
        beta_utility: exponential decay rate for utility tracking (default: 0.999)
        sigma: noise scale for perturbation (default: 0.001)
        beta1: exponential decay rate for first moment (Adam-style, default: 0.9)
        beta2: exponential decay rate for second moment (Adam-style, default: 0.999)
        eps: term for numerical stability (default: 1e-5)
        use_adam_moments: whether to use Adam-style moments (default: True)
        momentum: SGD-style momentum (only used if use_adam_moments=False, default: 0.0)
        gating_mode: layer-selective gating mode (default: 'full')
            - 'full': Apply gating to all layers
            - 'output_only': Apply gating only to final FC layer (fc.weight, fc.bias)
            - 'hidden_only': Apply gating to all layers except final FC layer
        non_gated_scale: scaling factor for non-gated layers (default: 0.5)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        beta_utility=0.999,
        sigma=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-5,
        use_adam_moments=True,
        momentum=0.0,
        gating_mode='full',
        non_gated_scale=0.5
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta_utility < 1.0:
            raise ValueError(f"Invalid beta_utility value: {beta_utility}")
        if sigma < 0.0:
            raise ValueError(f"Invalid sigma value: {sigma}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if gating_mode not in ['full', 'output_only', 'hidden_only']:
            raise ValueError(f"Invalid gating_mode: {gating_mode}. Must be 'full', 'output_only', or 'hidden_only'")
        if non_gated_scale < 0.0:
            raise ValueError(f"Invalid non_gated_scale value: {non_gated_scale}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            use_adam_moments=use_adam_moments,
            momentum=momentum,
            gating_mode=gating_mode,
            non_gated_scale=non_gated_scale
        )
        super(UPGD, self).__init__(params, defaults)

        # Track parameter names for logging
        self.param_names = {}

    def set_param_names(self, named_params):
        """
        Set parameter names for logging.

        Args:
            named_params: iterator of (name, param) tuples from model.named_parameters()
        """
        self.param_names = {id(param): name for name, param in named_params}

    def _should_apply_gating(self, param_name, gating_mode):
        """
        Determine if utility gating should be applied to this parameter.

        Args:
            param_name: Name of the parameter
            gating_mode: Gating mode ('full', 'output_only', 'hidden_only')

        Returns:
            bool: True if gating should be applied, False otherwise
        """
        if gating_mode == 'full':
            return True
        elif gating_mode == 'output_only':
            # ResNet18 final layer: 'fc.weight', 'fc.bias'
            return param_name.startswith('fc.')
        elif gating_mode == 'hidden_only':
            # All layers except final FC
            return not param_name.startswith('fc.')
        else:
            raise ValueError(f"Unknown gating_mode: {gating_mode}")

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # First pass: compute global max utility for normalization
        global_max_util = torch.tensor(-torch.inf, device=self.param_groups[0]['params'][0].device)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['avg_utility'] = torch.zeros_like(p.data)
                    if group['use_adam_moments']:
                        state['first_moment'] = torch.zeros_like(p.data)
                        state['sec_moment'] = torch.zeros_like(p.data)
                    else:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                state['step'] += 1

                # Update utility: u_t = β_u * u_{t-1} + (1 - β_u) * (-∇L * θ)
                avg_utility = state['avg_utility']
                avg_utility.mul_(group['beta_utility']).add_(
                    -p.grad.data * p.data, alpha=1 - group['beta_utility']
                )

                # Update moments if using Adam-style
                if group['use_adam_moments']:
                    first_moment = state['first_moment']
                    sec_moment = state['sec_moment']
                    first_moment.mul_(group['beta1']).add_(p.grad.data, alpha=1 - group['beta1'])
                    sec_moment.mul_(group['beta2']).add_(p.grad.data ** 2, alpha=1 - group['beta2'])
                else:
                    # SGD-style momentum
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(group['momentum']).add_(p.grad.data)

                # Track global max utility
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max

        # Second pass: update parameters using normalized utilities
        for group in self.param_groups:
            gating_mode = group['gating_mode']
            non_gated_scale = group['non_gated_scale']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Get parameter name for layer-selective gating
                param_id = id(p)
                param_name = self.param_names.get(param_id, f"param_{param_id}")

                # Check if gating should be applied to this parameter
                apply_gating = self._should_apply_gating(param_name, gating_mode)

                # Generate noise
                noise = torch.randn_like(p.grad) * group['sigma']

                # Compute update direction
                if group['use_adam_moments']:
                    # Adam-style update with bias correction
                    bias_correction_beta1 = 1 - group['beta1'] ** state['step']
                    bias_correction_beta2 = 1 - group['beta2'] ** state['step']

                    exp_avg = state['first_moment'] / bias_correction_beta1
                    exp_avg_sq = state['sec_moment'] / bias_correction_beta2

                    if apply_gating:
                        # Apply utility-based gating
                        # Bias correction for utility
                        bias_correction_utility = 1 - group['beta_utility'] ** state['step']
                        corrected_utility = state['avg_utility'] / bias_correction_utility

                        # Normalize utility by global max and apply sigmoid gating
                        # scaled_utility ∈ [0, 1], where 1 means "preserve" and 0 means "update"
                        if global_max_util > 0:
                            scaled_utility = torch.sigmoid(corrected_utility / global_max_util)
                        else:
                            scaled_utility = torch.zeros_like(corrected_utility)

                        # Update direction: adaptive gradient + noise, both gated by (1 - utility)
                        update = (exp_avg * (1 - scaled_utility)) / (exp_avg_sq.sqrt() + group['eps']) + \
                                 noise * (1 - scaled_utility)
                    else:
                        # Use fixed scaling for non-gated layers
                        update = (exp_avg * non_gated_scale) / (exp_avg_sq.sqrt() + group['eps']) + \
                                 noise * non_gated_scale
                else:
                    # SGD-style update with momentum
                    if apply_gating:
                        # Apply utility-based gating
                        bias_correction_utility = 1 - group['beta_utility'] ** state['step']
                        corrected_utility = state['avg_utility'] / bias_correction_utility

                        if global_max_util > 0:
                            scaled_utility = torch.sigmoid(corrected_utility / global_max_util)
                        else:
                            scaled_utility = torch.zeros_like(corrected_utility)

                        update = state['momentum_buffer'] * (1 - scaled_utility) + noise * (1 - scaled_utility)
                    else:
                        # Use fixed scaling for non-gated layers
                        update = state['momentum_buffer'] * non_gated_scale + noise * non_gated_scale

                # Apply weight decay (L2 regularization)
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Apply parameter update
                # Factor of 2.0 for compatibility with original UPGD formulation
                p.data.add_(update, alpha=-2.0 * group['lr'])

        return loss

    def get_utility_stats(self, bins=9):
        """
        Get utility statistics for logging.

        Args:
            bins: number of bins for histogram (default: 9)

        Returns:
            dict containing utility statistics:
                - 'utility_histograms': dict mapping param names to histogram arrays
                - 'utility_norms': dict mapping param names to L2 norms
                - 'global_max_utility': maximum utility across all parameters
                - 'mean_utility': mean utility across all parameters
                - 'utility_sparsity': fraction of utilities near zero
        """
        stats = {
            'utility_histograms': {},
            'utility_norms': {},
            'global_max_utility': -torch.inf,
            'mean_utility': 0.0,
            'utility_sparsity': 0.0,
        }

        all_utilities = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0 or 'avg_utility' not in state:
                    continue

                # Get parameter name
                param_id = id(p)
                param_name = self.param_names.get(param_id, f"param_{param_id}")

                # Get utility
                utility = state['avg_utility']

                # Compute histogram
                hist, _ = np.histogram(utility.cpu().numpy().flatten(), bins=bins, range=(0, 1))
                stats['utility_histograms'][param_name] = hist.tolist()

                # Compute L2 norm
                stats['utility_norms'][param_name] = float(torch.norm(utility).item())

                # Track for global statistics
                all_utilities.append(utility.flatten())

                # Update global max
                current_max = float(utility.max().item())
                if current_max > stats['global_max_utility']:
                    stats['global_max_utility'] = current_max

        # Compute global statistics
        if all_utilities:
            all_utilities_tensor = torch.cat(all_utilities)
            stats['mean_utility'] = float(all_utilities_tensor.mean().item())

            # Sparsity: fraction of utilities below 0.01
            stats['utility_sparsity'] = float((all_utilities_tensor.abs() < 0.01).float().mean().item())

        return stats

    def get_gating_stats(self):
        """
        Get statistics about the gating mechanism, including layer-selective gating info.

        Returns:
            dict containing gating statistics:
                - 'mean_gate_value': mean of (1 - scaled_utility) across all params
                - 'active_fraction': fraction of parameters with gate > 0.5
                - 'inactive_fraction': fraction of parameters with gate < 0.5
                - 'gated_params': number of parameters with gating applied
                - 'non_gated_params': number of parameters without gating
                - 'gating_mode': current gating mode
        """
        stats = {
            'mean_gate_value': 0.0,
            'active_fraction': 0.0,
            'inactive_fraction': 0.0,
            'gated_params': 0,
            'non_gated_params': 0,
            'gating_mode': 'full',  # Default
        }

        all_gates = []
        gated_count = 0
        non_gated_count = 0

        # Get gating mode from first group
        if self.param_groups:
            stats['gating_mode'] = self.param_groups[0]['gating_mode']

        # Recompute gating values (same as in step())
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0 or 'avg_utility' not in state:
                    continue

                avg_utility = state['avg_utility']
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max

        for group in self.param_groups:
            gating_mode = group['gating_mode']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0 or 'avg_utility' not in state:
                    continue

                # Get parameter name and check if gating applies
                param_id = id(p)
                param_name = self.param_names.get(param_id, f"param_{param_id}")
                apply_gating = self._should_apply_gating(param_name, gating_mode)

                # Count gated vs non-gated parameters
                if apply_gating:
                    gated_count += p.numel()
                else:
                    non_gated_count += p.numel()

                bias_correction_utility = 1 - group['beta_utility'] ** state['step']
                corrected_utility = state['avg_utility'] / bias_correction_utility

                if global_max_util > 0:
                    scaled_utility = torch.sigmoid(corrected_utility / global_max_util)
                else:
                    scaled_utility = torch.zeros_like(corrected_utility)

                gate = 1 - scaled_utility  # Gate value (1 = update, 0 = preserve)
                all_gates.append(gate.flatten())

        stats['gated_params'] = gated_count
        stats['non_gated_params'] = non_gated_count

        if all_gates:
            all_gates_tensor = torch.cat(all_gates)
            stats['mean_gate_value'] = float(all_gates_tensor.mean().item())
            stats['active_fraction'] = float((all_gates_tensor > 0.5).float().mean().item())
            stats['inactive_fraction'] = float((all_gates_tensor <= 0.5).float().mean().item())

        return stats
