# Loss of Plasticity in Continual ImageNet
This directory contains the implementation of the continual binary ImageNet classification problem.

The first step to replicate the results is to download the data. The data can be downloaded [here](https://drive.google.com/file/d/1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z/view?usp=sharing).
Create a directory named `data` and extract the downloaded file in `data`
```sh
cd lop/imagenet/
mkdir data
```

The next step is to test a deep convolution network that uses backprop to learn.
The network is specified in [`../nets/conv_net.py`](../nets/conv_net.py)
This command produces 30 temporary cfg files in `temp_cfg`.

```sh
python3.8 multi_param_expr.py -c cfg/cbp.json 
```

Each of the new temporary cfg files can then be used to do one run of backprop. Each run takes about 12 hours on an A-100 GPU.
```sh
python3.8 expr.py -c temp_cfg/0.json 
```

Finally, after completing all the runs, the command below can be used to generate
the plot for the network specified in `cfg/sgd/bp.json`.

```sh
python3.8 bp_plot.py --c cfg/snp.json 
```

The command above will generate the plot on the right below.
The results below are averaged for 30 runs, and accuracy is binned into bins of size 50.


![](bp_imagenet.png "BP on Continual ImageNet")

Similarly, you can use the cfg files for l2 regularization, shrink-and-perturb, and continual backpropagation
to evaluate the peformance of these methods. The results of these methods can be plotted using the
following command.

```sh
python3.8 all_plot.py --c cfg/bp.json 
```

The results for all the methods are presented below.

![](all_methods_imagenet.png "All methods on CIBC")



## Boonam: ImageNet Experiment Metrics (`single_expr_curv.py`)

This document details the metrics calculated by the `single_expr_curv.py` script during the ImageNet continual learning experiment.

### Overview

The script has been enhanced to compute and save a comprehensive set of metrics at the end of each training epoch for every task. This allows for a detailed analysis of the model's behavior and learning dynamics over time, similar to the analyses performed for the Permuted MNIST and Incremental CIFAR experiments.

### Calculated Metrics

The following metrics are calculated and stored in the final output `.pkl` file:

1.  **`train_accuracies`**: The average training accuracy for the current task.
2.  **`test_accuracies`**: The average test accuracy for the current task.
3.  **`curvatures`**: An approximation of the Hessian (loss landscape curvature) calculated using finite differences. This metric helps understand the flatness or sharpness of the loss surface.
4.  **`weight_magnitudes`**: The average absolute magnitude of all weights in the network. This can be an indicator of weight growth or decay.
5.  **`dormant_proportions`**: The proportion of neurons in the hidden layers whose activations are close to zero (below a threshold of 0.01) for a given batch of data. This helps identify "dead" or inactive units.
6.  **`effective_ranks`**: The effective rank of the final hidden layer's activation matrix. This is an entropy-based measure that indicates the dimensionality of the learned representations.
7.  **`stable_ranks`**: The stable rank (or nuclear norm / Frobenius norm) of the final hidden layer's activation matrix. It provides another perspective on the dimensionality of the representations.

### Implementation Details

- The metrics are computed after each training epoch completes.
- All results are aggregated in tensors and saved to a single pickle file specified in the experiment's configuration.
- The metric calculation functions were adapted from `lop/permuted_mnist/plots/bp_metrics_curv.py` and `lop/incremental_cifar/post_run_analysis_modified2.py` to be compatible with the `ConvNet` and `ConvNet2` architectures used in this experiment.
