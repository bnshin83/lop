#!/usr/bin/env python3
import numpy as np
import csv
import os
import argparse
import glob
from pathlib import Path

def convert_npz_to_csv(npz_file_path, output_dir=None):
    """Convert a single .npz file to CSV format."""
    try:
        # Load the numpy archive
        archive = np.load(npz_file_path)
        
        # Determine base output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(npz_file_path).replace('.npz', '')
        else:
            base_name = npz_file_path.replace('.npz', '')
        
        print(f"\nProcessing NPZ: {npz_file_path}")
        print(f"  Contains {len(archive.files)} arrays: {list(archive.files)}")
        
        # Check if all arrays have the same first dimension (indicating tabular data)
        array_shapes = {name: archive[name].shape for name in archive.files}
        first_dims = [shape[0] if len(shape) > 0 else 1 for shape in array_shapes.values()]
        
        if len(set(first_dims)) == 1 and len(first_dims) > 1:
            # All arrays have same first dimension - treat as tabular data
            print(f"  Detected tabular data with {first_dims[0]} rows")
            
            if output_dir:
                csv_file = os.path.join(output_dir, f"{base_name}_combined.csv")
            else:
                csv_file = f"{base_name}_combined.csv"
            
            print(f"  Creating combined CSV: {csv_file}")
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header with array names
                header = []
                arrays = {}
                for name in sorted(archive.files):  # Sort for consistent column order
                    array = archive[name]
                    arrays[name] = array
                    if len(array.shape) == 1:
                        header.append(name)
                    else:
                        # For multi-dimensional arrays, create multiple columns
                        if len(array.shape) == 2:
                            for j in range(array.shape[1]):
                                header.append(f"{name}_{j}")
                        else:
                            # Flatten higher dimensions
                            flat_size = np.prod(array.shape[1:])
                            for j in range(flat_size):
                                header.append(f"{name}_{j}")
                
                writer.writerow(header)
                
                # Write data rows
                num_rows = first_dims[0]
                for i in range(num_rows):
                    row = []
                    for name in sorted(archive.files):
                        array = arrays[name]
                        if len(array.shape) == 1:
                            if array.dtype == object:
                                row.append(str(array[i]))
                            else:
                                row.append(array[i])
                        else:
                            # Handle multi-dimensional arrays
                            if len(array.shape) == 2:
                                for j in range(array.shape[1]):
                                    if array.dtype == object:
                                        row.append(str(array[i, j]))
                                    else:
                                        row.append(array[i, j])
                            else:
                                # Flatten higher dimensions
                                flat_data = array[i].flatten()
                                for val in flat_data:
                                    if array.dtype == object:
                                        row.append(str(val))
                                    else:
                                        row.append(val)
                    writer.writerow(row)
            
            archive.close()
            print(f"  ✅ Combined NPZ conversion complete!")
            return [csv_file]
        
        else:
            # Arrays have different shapes - create separate files
            print(f"  Arrays have different shapes - creating separate files")
            converted_files = []
            
            # Process each array in the archive
            for array_name in archive.files:
                array = archive[array_name]
                
                # Create output filename for this array
                if output_dir:
                    csv_file = os.path.join(output_dir, f"{base_name}_{array_name}.csv")
                else:
                    csv_file = f"{base_name}_{array_name}.csv"
                
                print(f"  Array '{array_name}':")
                print(f"    Shape: {array.shape}")
                print(f"    Type: {array.dtype}")
                print(f"    Memory size: {array.nbytes / (1024**3):.2f} GB")
                print(f"    Writing to: {csv_file}")
                
                # Write array to CSV file
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # If array is 0D (scalar)
                    if len(array.shape) == 0:
                        writer.writerow(['Value'])
                        if array.dtype == object:
                            writer.writerow([str(array.item())])
                        else:
                            writer.writerow([array.item()])
                    
                    # If array is 1D
                    elif len(array.shape) == 1:
                        writer.writerow([array_name])
                        for value in array:
                            if array.dtype == object:
                                writer.writerow([str(value)])
                            else:
                                writer.writerow([value])
                    
                    # If array is 2D
                    elif len(array.shape) == 2:
                        for row in array:
                            if array.dtype == object:
                                writer.writerow([str(item) for item in row])
                            else:
                                writer.writerow(row)
                    
                    # If array has more dimensions
                    else:
                        array_flat = array.reshape(array.shape[0], -1)
                        for row in array_flat:
                            if array.dtype == object:
                                writer.writerow([str(item) for item in row])
                            else:
                                writer.writerow(row)
                
                converted_files.append(csv_file)
                print(f"    ✅ Array '{array_name}' converted!")
            
            archive.close()
            print(f"  ✅ NPZ conversion complete! Created {len(converted_files)} CSV files.")
            return converted_files
        
    except Exception as e:
        print(f"  ❌ Error processing {npz_file_path}: {e}")
        return None

def convert_npy_to_csv(npy_file_path, output_dir=None):
    """Convert a single .npy file to CSV format."""
    try:
        # Load the numpy array (allow pickle for object arrays)
        array = np.load(npy_file_path, allow_pickle=True)
        
        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            csv_file = os.path.join(output_dir, os.path.basename(npy_file_path).replace('.npy', '.csv'))
        else:
            csv_file = npy_file_path.replace('.npy', '.csv')
        
        # Print information about the array
        print(f"\nProcessing: {npy_file_path}")
        print(f"  Array shape: {array.shape}")
        print(f"  Array type: {array.dtype}")
        print(f"  Memory size: {array.nbytes / (1024**3):.2f} GB")
        
        # Write array to CSV file
        print(f"  Writing to: {csv_file}")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # If array is 0D (scalar)
            if len(array.shape) == 0:
                writer.writerow(['Value'])
                # Handle scalar object arrays
                if array.dtype == object:
                    writer.writerow([str(array.item())])
                else:
                    writer.writerow([array.item()])
            
            # If array is 1D
            elif len(array.shape) == 1:
                writer.writerow(['Value'])
                for value in array:
                    # Handle object arrays by converting to string
                    if array.dtype == object:
                        writer.writerow([str(value)])
                    else:
                        writer.writerow([value])
            
            # If array is 2D
            elif len(array.shape) == 2:
                # Write header for 2D arrays (optional)
                # writer.writerow([f'Sample_{i}' for i in range(array.shape[1])])
                for row in array:
                    # Handle object arrays by converting each element to string
                    if array.dtype == object:
                        writer.writerow([str(item) for item in row])
                    else:
                        writer.writerow(row)
            
            # If array has more dimensions
            else:
                array_flat = array.reshape(array.shape[0], -1)
                for row in array_flat:
                    # Handle object arrays by converting each element to string
                    if array.dtype == object:
                        writer.writerow([str(item) for item in row])
                    else:
                        writer.writerow(row)
        
        print(f"  ✅ Conversion complete!")
        return csv_file
        
    except Exception as e:
        print(f"  ❌ Error processing {npy_file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert .npy and .npz files to CSV format")
    parser.add_argument("input", nargs="*", help="Input .npy/.npz file(s) or directory containing numpy files")
    parser.add_argument("--directory", "-d", type=str, help="Directory containing .npy/.npz files to convert")
    parser.add_argument("--output", "-o", type=str, help="Output directory for CSV files (default: same as input)")
    parser.add_argument("--pattern", "-p", type=str, default="*", help="File pattern to match (default: *)")
    
    args = parser.parse_args()
    
    # Collect all numpy files to process
    numpy_files = []
    
    # If specific files are provided
    if args.input:
        for item in args.input:
            if os.path.isfile(item) and (item.endswith('.npy') or item.endswith('.npz')):
                numpy_files.append(item)
            elif os.path.isdir(item):
                npy_files = glob.glob(os.path.join(item, "*.npy"))
                npz_files = glob.glob(os.path.join(item, "*.npz"))
                numpy_files.extend(npy_files + npz_files)
    
    # If directory is specified
    if args.directory:
        if os.path.isdir(args.directory):
            npy_files = glob.glob(os.path.join(args.directory, "*.npy"))
            npz_files = glob.glob(os.path.join(args.directory, "*.npz"))
            numpy_files.extend(npy_files + npz_files)
        else:
            print(f"❌ Directory not found: {args.directory}")
            return
    
    # If no input specified, prompt for input directory
    if not numpy_files:
        print("No input files or directories specified.")
        print("Please enter the directory containing .npy/.npz files to convert:")
        
        # Show some common directories as suggestions
        default_dir = '/scratch/gautschi/shin283/loss-of-plasticity/lop/incremental_cifar/results2/progression_cache'
        print(f"Common options:")
        print(f"  1. Progression cache: {default_dir}")
        print(f"  2. Current directory: {os.getcwd()}")
        print(f"  3. Custom path (enter below)")
        print()
        
        # Get user input
        try:
            user_input = input("Enter directory path (or press Enter for progression cache): ").strip()
            
            # If empty, use default
            if not user_input:
                user_input = default_dir
                print(f"Using default: {user_input}")
            
            # Expand ~ and relative paths
            user_input = os.path.expanduser(user_input)
            user_input = os.path.abspath(user_input)
            
            if os.path.isdir(user_input):
                npy_files = glob.glob(os.path.join(user_input, "*.npy"))
                npz_files = glob.glob(os.path.join(user_input, "*.npz"))
                found_files = npy_files + npz_files
                numpy_files.extend(found_files)
                print(f"✅ Using directory: {user_input}")
                print(f"   Found {len(found_files)} .npy/.npz files")
            else:
                print(f"❌ Directory not found: {user_input}")
                print("Usage examples:")
                print("  python npy_to_csv.py file1.npy file2.npz")
                print("  python npy_to_csv.py --directory /path/to/numpy/files")
                print("  python npy_to_csv.py /path/to/directory")
                return
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user.")
            return
        except EOFError:
            # Fallback to default if running in non-interactive environment
            user_input = default_dir
            if os.path.isdir(user_input):
                npy_files = glob.glob(os.path.join(user_input, "*.npy"))
                npz_files = glob.glob(os.path.join(user_input, "*.npz"))
                numpy_files.extend(npy_files + npz_files)
                print(f"Using default directory: {user_input}")
            else:
                print("❌ No input files specified and default directory not found.")
                return
    
    # Remove duplicates and sort
    numpy_files = sorted(list(set(numpy_files)))
    
    if not numpy_files:
        print("❌ No .npy/.npz files found!")
        return
    
    print(f"Found {len(numpy_files)} numpy file(s) to convert:")
    for f in numpy_files:
        print(f"  - {f}")
    
    # Process each file
    converted_files = []
    failed_files = 0
    
    for numpy_file in numpy_files:
        if numpy_file.endswith('.npz'):
            csv_files = convert_npz_to_csv(numpy_file, args.output)
            if csv_files:
                converted_files.extend(csv_files)
            else:
                failed_files += 1
        else:  # .npy file
            csv_file = convert_npy_to_csv(numpy_file, args.output)
            if csv_file:
                converted_files.append(csv_file)
            else:
                failed_files += 1
    
    print(f"\n🎉 Conversion Summary:")
    print(f"  Processed: {len(numpy_files)} files")
    print(f"  Successful: {len(converted_files)} CSV files created")
    print(f"  Failed: {failed_files} files")
    
    if converted_files:
        print(f"\nConverted files:")
        for f in converted_files:
            print(f"  ✅ {f}")

if __name__ == "__main__":
    main()
