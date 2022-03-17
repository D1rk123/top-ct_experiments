import numpy as np
from folder_locations import get_data_folder, get_results_folder
import ct_experiment_utils as ceu


def list_subfolders(experiment_folder):
    subfolders = []
    for child in experiment_folder.iterdir():
        if child.is_dir() and child.name != "scripts" and child.name != "projections":
            subfolders.append(child.name)
    subfolders.sort()
    return subfolders

    
def make_metrics_overview(lo, hi, experiment_folder, result_file_name = "metrics_overview.csv"):
    variant_names = list_subfolders(experiment_folder)
    
    header = ",PSNR,std.dev,SSIM,std.dev,Iterations,std.dev"
    
    with open(experiment_folder / result_file_name, "w") as result_file:
        result_file.write(header+"\n")
        for i, name in enumerate(variant_names):
            metrics = np.loadtxt(experiment_folder / name / "metrics.csv", delimiter=",", skiprows=1)
            psnr_mean = np.mean(metrics[lo:hi, 1])
            psnr_std = np.std(metrics[lo:hi, 1], ddof=1)
            ssim_mean = np.mean(metrics[lo:hi, 4])
            ssim_std = np.std(metrics[lo:hi, 4], ddof=1)
            
            iterations_path = experiment_folder / name / "optimal_iters.txt"
            if iterations_path.is_file():
                iterations = np.loadtxt(iterations_path)
                if len(iterations.shape) == 0:
                    iters_mean = iterations
                    iters_std = 0
                else:
                    iters_mean = np.mean(iterations[lo:hi])
                    iters_std = np.std(iterations[lo:hi], ddof=1)
            else:
                iters_mean = 0
                iters_std = 0
            metrics_str = f"{name},{psnr_mean},{psnr_std},{ssim_mean},{ssim_std},{iters_mean},{iters_std}"
            result_file.write(metrics_str+"\n")

if __name__ == "__main__":
    #experiment_folder = get_results_folder() / "2021-12-22_68_objects_25000_photons_3"
    #experiment_folder = get_results_folder() / "2022-01-04_68_objects_25000_photons_att_parameter_tuning_1"
    #experiment_folder = get_results_folder() / "2022-01-05_38_objects_25000_photons_together_att_parameter_tuning_1"
    #experiment_folder = get_results_folder() / "2022-01-10_38_objects_25000_photons_together_att_parameter_tuning_1"
    experiment_folder = get_results_folder() / "2022-01-12_68_objects_25000_photons_method_comparison_1"

    make_metrics_overview(lo=9, hi=59, experiment_folder=experiment_folder)
