# ============================================================================
# PER IMAGE CURVATURE ANALYSIS SCRIPT
# ============================================================================
# This script analyzes the loss landscape curvature of neural networks on a per-image basis.
# It helps researchers understand how different images affect the model's behavior.
# ============================================================================

# ======= IMPORTS SECTION =======
# Standard library imports - fundamental Python packages
import os  # For file and directory operations
import torch  # PyTorch deep learning framework
import numpy as np  # NumPy for numerical operations
from tqdm import tqdm  # Progress bar for tracking long operations
from torch.utils.data import DataLoader  # For efficient data loading

# Custom imports for data handling and model architecture
# CifarDataSet - A class that manages CIFAR dataset operations
from mlproj_manager.problems import CifarDataSet
# Transformation utilities to prepare images for neural networks
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize
# PyTorch's transformation package
from torchvision import transforms

# Fix the import path for the custom ResNet model
import sys
import os

# Add the project root directory to sys.path to find 'lop.nets' module
# The module is at C:\Users\bnshi\OneDrive - purdue.edu\loss-of-plasticity\lop\nets\torchvision_modified_resnet.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Now the import should work
from lop.nets.torchvision_modified_resnet import build_resnet18


def compute_curvature_per_image(net, data_loader, h=1e-3, niter=10, temp=1.0, save_dir=None):
    """
    Compute the curvature and directional eigenvalues for each image in the dataset.
    
    What this function does:
    - For each image, it measures how "curved" the loss landscape is
    - Higher curvature means the model's predictions change more dramatically with small input changes
    - This helps identify which images cause instability in the neural network
    
    Technical explanation:
    - We approximate the Hessian (second derivative) of the loss function
    - We use random direction vectors and finite difference approximation
    - This gives us insight into the local geometry of the loss landscape
    
    Args:
        net: Neural network model (ResNet18) - the trained model to analyze
        data_loader: DataLoader containing the dataset - provides batches of images
        h: Step size for finite difference approximation (default: 1e-3) - how much to perturb the image
        niter: Number of random directions to average over (default: 10) - more directions = more accurate but slower
        temp: Temperature parameter for softmax (default: 1.0) - controls the "sharpness" of probability distribution
        save_dir: Directory to save results (default: None) - if provided, results will be saved here
    
    Returns:
        tuple: (curvature_tensor, directional_eig_tensor, labels_tensor)
            - curvature_tensor: Measure of overall curvature for each image
            - directional_eig_tensor: Eigenvalues in random directions for each image
            - labels_tensor: Class labels for each image
    """
    # Check if the dataset is empty
    if len(data_loader) == 0:
        print("WARNING: Empty dataset. No data to analyze.")
        return None, None, None
        
    # Get device of the model (CPU or GPU) - ensures computations happen on the same device as the model
    device = next(net.parameters()).device
    net.eval()  # Set model to evaluation mode - disables dropout, uses running stats for batch norm
    criterion = torch.nn.CrossEntropyLoss()  # Define loss function - measures difference between predictions and labels

    # Initialize lists to store results - will be filled and concatenated later
    curvature_list = []  # Will store curvature values
    directional_eig_list = []  # Will store directional eigenvalues
    label_list = []  # Will store corresponding class labels

    # Limit the number of samples to process for faster analysis
    max_samples = 100000  # Adjust this number based on your needs
    
    # Convert DataLoader to iterable list for tqdm progress bar
    data_list = list(data_loader)
    total_batches = min(len(data_list), max_samples // data_loader.batch_size + 1)
    print(f"Processing {total_batches} batches ({min(max_samples, len(data_list) * data_loader.batch_size)} samples)...")
    
    # Process each batch of images with progress bar
    for batch_idx, (images, labels) in enumerate(tqdm(data_list[:total_batches], desc="Computing curvature")):
        # Move data to appropriate device (CPU or GPU) - ensures efficient computation
        batch_data = images.to(device)  # Images
        batch_labels = labels.to(device)  # Labels

        num_samples = batch_data.shape[0]  # Number of images in this batch
        
        # Initialize tensors to accumulate results across iterations
        regr = torch.zeros(num_samples, device=device)  # For curvature (gradient norm)
        eigs = torch.zeros(num_samples, device=device)  # For eigenvalues (directional derivatives)

        # Iterate over multiple random directions to get more reliable estimates
        for _ in range(niter):
            # Generate random direction vector with elements ±1
            # This creates random perturbations to apply to each image
            v = torch.randint_like(batch_data, high=2, device=device) * 2 - 1  # Values are either -1 or 1
            v = h * v  # Scale the direction vector by step size h

            with torch.enable_grad():
                # Enable gradient computation for input - needed to track how loss changes with input
                batch_data = batch_data.detach().requires_grad_(True)
                
                # Forward pass with perturbed input (original image + small random noise)
                outputs_pos = net(batch_data + v)  # Model predictions for perturbed images
                outputs_orig = net(batch_data)  # Model predictions for original images
                
                # Compute loss for both original and perturbed inputs
                loss_pos = criterion(outputs_pos / temp, batch_labels)  # Loss for perturbed images
                loss_orig = criterion(outputs_orig / temp, batch_labels)  # Loss for original images
                
                # Compute gradient of loss difference with respect to the input
                # This tells us how the loss change (due to perturbation) varies with the input
                grad_diff = torch.autograd.grad(loss_pos - loss_orig, batch_data, create_graph=False)[0]

            # Accumulate curvature metrics
            # The gradient norm measures overall curvature - how much the loss changes in any direction
            regr += grad_diff.reshape(num_samples, -1).norm(dim=1)  # Reshape to (batch_size, flattened_image_size) and compute norm
            
            # Directional eigenvalue estimates how much the loss curves specifically in the direction of v
            # A dot product between the perturbation and the gradient difference
            eigs += (v.reshape(num_samples, -1) * grad_diff.reshape(num_samples, -1)).sum(dim=1)

            # Clear gradients to prepare for next iteration
            # This prevents gradient accumulation across iterations
            net.zero_grad()  # Clear gradients in the model
            if batch_data.grad is not None:
                batch_data.grad.zero_()  # Clear gradients of the input data

        # Average over iterations to get more stable estimates
        regr /= niter  # Average curvature
        eigs /= niter  # Average directional eigenvalues

        # Store results from this batch
        curvature_list.append(regr.cpu())  # Move results to CPU for storage
        directional_eig_list.append(eigs.cpu())
        label_list.append(batch_labels.cpu())

    # Check if any data was processed
    if not curvature_list:
        print("WARNING: No data to process for this epoch/task. Skipping.")
        return None, None, None

    # Concatenate results from all batches into single tensors
    curvature_tensor = torch.cat(curvature_list)  # Shape: (num_total_samples,)
    directional_eig_tensor = torch.cat(directional_eig_list)  # Shape: (num_total_samples,)
    labels_tensor = torch.cat(label_list)  # Shape: (num_total_samples,)

    # Save results if a directory is provided
    if save_dir:
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save numpy arrays for later analysis
        np.save(os.path.join(save_dir, "per_image_curvature.npy"), curvature_tensor.numpy())
        np.save(os.path.join(save_dir, "per_image_directional_eig.npy"), directional_eig_tensor.numpy())
        np.save(os.path.join(save_dir, "per_image_labels.npy"), labels_tensor.numpy())
        print(f"Saved per-image curvature/eig/labels to {save_dir}")

    # Return the computed tensors
    return curvature_tensor, directional_eig_tensor, labels_tensor


def load_cifar_data(data_path: str, train: bool = True, allowed_classes=None) -> DataLoader:
    """
    Load and prepare CIFAR-100 dataset with appropriate transformations, optionally filtering to allowed_classes.
    
    What this function does:
    - Sets up the CIFAR-100 dataset with proper preprocessing
    - Applies necessary transformations to make images suitable for neural networks
    - Creates a DataLoader that efficiently provides batches of images
    
    Args:
        data_path: Path to CIFAR dataset - directory where the dataset is stored
        train: Whether to load training data (default: True) - set to False for test data
    
    Returns:
        DataLoader: Configured data loader for CIFAR-100 - provides batches of (image, label) pairs
    """
    # Use torchvision's built-in CIFAR100 instead of custom CifarDataSet for more reliable loading
    import torchvision
    from torchvision.datasets import CIFAR100
    from torch.utils.data import TensorDataset
    
    # Use torchvision's standard transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    # Load the CIFAR-100 dataset using torchvision
    try:
        dataset = CIFAR100(root=os.path.dirname(data_path), train=train, download=True, transform=transform)
    except RuntimeError:
        # If there's an issue with the download, try with download=False
        dataset = CIFAR100(root=os.path.dirname(data_path), train=train, download=False, transform=transform)
        
    # Extract images and labels
    all_images = []
    all_labels = []
    for img, label in dataset:
        all_images.append(img)
        all_labels.append(label)
    
    # Convert to tensors
    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)
    
    # Create a TensorDataset
    cifar_data = TensorDataset(images_tensor, labels_tensor)

    # Debug: Print unique class IDs and sample counts in the dataset
    labels = labels_tensor.numpy()
    unique_classes = sorted(np.unique(labels).tolist())
    class_counts = {cls: np.sum(labels == cls).item() for cls in unique_classes}
    
    print(f"DEBUG: Unique class IDs in dataset: {unique_classes}")
    print(f"DEBUG: Number of samples per class: {class_counts}")
    print(f"DEBUG: Total number of samples: {len(labels)}")
    
    # Filter by allowed classes if specified
    if allowed_classes is not None:
        from torch.utils.data import Subset
        allowed_classes_set = set(allowed_classes)
        valid_indices = [i for i, label in enumerate(labels) if label in allowed_classes_set]
        
        if not valid_indices:
            print(f"WARNING: No samples found for allowed_classes={allowed_classes}.")
            # Return an empty dataset to avoid errors but allow the script to continue
            return DataLoader(
                TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)),
                batch_size=1
            )
            
        print(f"Filtered dataset to {len(valid_indices)} samples matching allowed classes.")
        filtered_dataset = Subset(cifar_data, valid_indices)
    else:
        filtered_dataset = cifar_data

    return DataLoader(
        filtered_dataset,  # The dataset to load data from
        batch_size=50,  # Smaller batch size for more frequent updates and better progress tracking
        shuffle=False,  # Don't shuffle the data (maintain order for analysis)
        num_workers=4  # Number of parallel processes to use for loading data
    )


def run_per_image_curvature_analysis(data_path, model_path, save_dir, allowed_classes=None):
    """
    Main function to run curvature analysis on a trained model.
    Optionally restricts analysis to allowed_classes (for incremental learning).
    
    What this function does:
    - Loads a pre-trained neural network model
    - Prepares the dataset
    - Runs the curvature analysis on all images
    - Saves the results
    
    Args:
        data_path: Path to CIFAR dataset - where the images are stored
        model_path: Path to saved model parameters - the weights of the trained network
        save_dir: Directory to save results - where analysis outputs will be stored
    """
    # Initialize ResNet18 model with batch normalization
    # ResNet18 is a standard neural network architecture with 18 layers
    # BatchNorm helps stabilize and accelerate training
    net = build_resnet18(
        num_classes=100,  # CIFAR-100 has 100 different classes
        norm_layer=torch.nn.BatchNorm2d  # Use batch normalization
    )
    
    # Check if CUDA is available and set device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model parameters from file with proper device mapping
    # This loads the weights that were learned during training and maps them to the current device
    net.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move model to the appropriate device and set to evaluation mode
    # eval() disables dropout and uses running stats for batch normalization
    net.eval().to(device)

    # Load dataset using the helper function defined above
    data_loader = load_cifar_data(data_path, train=True, allowed_classes=allowed_classes)
    
    # Run curvature analysis on the loaded model and dataset
    # Results will be saved to the specified directory
    compute_curvature_per_image(
        net=net,  # The neural network model to analyze
        data_loader=data_loader,  # The dataset to perform analysis on
        h=1e-3,  # Step size for finite difference approximation
        niter=10,  # Number of random directions to average over
        temp=1.0,  # Temperature parameter for softmax
        save_dir=save_dir  # Directory to save the results
    )


# ======= SCRIPT ENTRY POINT =======
# This block runs when the script is executed directly (not imported)
if __name__ == "__main__":
    # Command-line interface setup
    # This allows users to provide parameters when running the script from terminal
    import argparse  # Library for parsing command-line arguments
    parser = argparse.ArgumentParser(description="Analyze per-image curvature of a neural network model")
    
    # Define required command-line arguments
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the CIFAR dataset directory")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the saved model weights file (.pth)")
    parser.add_argument("--save_dir", type=str, required=True, 
                        help="Directory to save analysis results")
    parser.add_argument("--allowed_classes", type=str, default=None, 
                        help="Comma-separated list of allowed class IDs (e.g. '0,1,2,3,4,5')")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    allowed_classes = None
    if args.allowed_classes:
        allowed_classes = [int(x) for x in args.allowed_classes.split(",") if x.strip()]

    # Run analysis with command-line arguments
    run_per_image_curvature_analysis(
        data_path=args.data_path,  # Path to dataset
        model_path=args.model_path,  # Path to model weights
        save_dir=args.save_dir,  # Path to save results
        allowed_classes=allowed_classes
    )

# Example usage in batch script:
# For task t (0-based), allowed_classes = list(range((t+1)*5))
# run_per_image_curvature_analysis(data_path, model_path, save_dir, allowed_classes=list(range((task_id+1)*5)))
