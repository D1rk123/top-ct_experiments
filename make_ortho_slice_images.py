from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from folder_locations import get_data_folder, get_results_folder
import ct_experiment_utils as ceu

def to_rgb(img, range_low, range_high):
    range_img = np.clip((img-range_low)/(range_high-range_low), 0, 1)
    return np.tile(range_img[:, :, None], (1, 1, 3))

def save_ortho_slices(stack, save_folder):
    if not isinstance(stack, np.ndarray):
        stack = ceu.load_stack(stack)
    save_folder.mkdir(parents=True)

    plt.imsave(save_folder / "axis1.png", to_rgb(stack[237//2, :, :], 0, 0.06))
    plt.imsave(save_folder / "axis2.png", to_rgb(np.flipud(stack[:, 255//2, :]), 0, 0.06))
    plt.imsave(save_folder / "axis3.png", to_rgb(np.flipud(stack[:, :, 255//2]), 0, 0.06))
    #plt.imsave(save_folder / "axis1.png", to_rgb(stack[237//2, :, :], 0, 0.025))
    #plt.imsave(save_folder / "axis2.png", to_rgb(stack[:, 255//2, :], 0, 0.025))
    #plt.imsave(save_folder / "axis3.png", to_rgb(stack[:, :, 255//2], 0, 0.025))

if __name__ == "__main__":
    #experiments_path = Path("/run/media/dirkschut/dirk_ext/PhD/Experiments/overlapping_projections_paper")
    experiments_path = get_results_folder()
    input_folder = get_results_folder() / "2022-01-12_68_objects_25000_photons_method_comparison_1"
    
    experiment_name = "ortho_slices_method_comparison"
    experiment_folder = ceu.make_new_experiment_folder(experiments_path, name=experiment_name)
    
    save_ortho_slices(input_folder / "together" / "recon_31", experiment_folder / "together")
    save_ortho_slices(input_folder / "ignore" / "recon_31", experiment_folder / "ignore")
    save_ortho_slices(input_folder / "subtract" / "recon_31", experiment_folder / "subtract")
    save_ortho_slices(input_folder / "submatrix" / "recon_31", experiment_folder / "submatrix")
    save_ortho_slices(input_folder / "ignore_100_iters" / "recon_31", experiment_folder / "ignore_100_iters")

