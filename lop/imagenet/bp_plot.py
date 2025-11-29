import sys
import json
import pickle
import argparse
from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


def add_cfg_performance(cfg='', setting_idx=0, m=2*10*1000, num_runs=30):
    with open(cfg, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []
    loaded_runs_count = 0 # Keep track of how many runs were successfully loaded
    for idx in range(num_runs):
        file_path = params['data_dir'] + str(setting_idx) + '/' + str(idx)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            # Online performance
            per_param_setting_performance.append(np.array(bin_m_errs(errs=data['test_accuracies'][:, -1].flatten()*100, m=m)))
            loaded_runs_count += 1
        except FileNotFoundError:
            print(f"Warning: Data file not found and skipped: {file_path}")
        except Exception as e: # Catch other potential errors during loading/processing
            print(f"Warning: Error processing file {file_path}: {e}. Skipping this run.")

    if not per_param_setting_performance:
        setting_name = "Unknown Setting"
        if setting_idx < len(param_settings):
            setting_name = param_settings[setting_idx]
        print(f"Warning: No data files successfully loaded for setting '{setting_name}' (index {setting_idx}). Plotting may be affected.")
        return np.array([]) 
    
    mean_performance = np.array(per_param_setting_performance).mean()
    setting_name = "Unknown Setting"
    if setting_idx < len(param_settings):
        setting_name = param_settings[setting_idx]
    print(f"Setting: '{setting_name}' (Index: {setting_idx}), Loaded Runs: {loaded_runs_count}/{num_runs}, Mean Performance: {mean_performance}")
    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_file', help="Path of the file containing the parameters of the experiment", type=str,
                            default='cfg/bp.json')
    args = parser.parse_args(arguments)
    cfg_file = args.cfg_file

    with open(cfg_file, 'r') as f:
        params = json.load(f)
    list_params, param_settings = get_configurations(params=params)

    performances = []
    m = 50
    num_runs = params['num_runs']
    num_settings = len(param_settings)

    indices = [i for i in range(num_settings)]
    for i in indices:
        performances.append(add_cfg_performance(cfg=cfg_file, setting_idx=i, m=m, num_runs=num_runs))

    performances.append(0.771 * np.ones(performances[-1].shape if performances else (0,)) * 100) # Handle case where performances might be empty
    param_settings.append('linear')
    indices.append(-1)

    # Convert all param_settings to strings for consistent labeling
    string_labels = [str(ps) for ps in param_settings]

    yticks = [70, 75, 80, 85, 90]
    generate_online_performance_plot(
        performances=performances,
        colors=['C3', 'C1', 'C2'],
        yticks=yticks,
        xticks=[0, 1000, 2000],
        xticks_labels=['0', '1k', '2k'],
        m=m,
        fontsize=18,
        labels=np.array(string_labels)[indices],
    )


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

