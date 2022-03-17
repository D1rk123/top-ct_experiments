from pathlib import Path
import torch
import astra
import numpy as np
from matplotlib import pyplot as plt

import tomosipo as ts
from tomosipo.qt import animate

import ct_experiment_utils as ceu
from multi_operator import MultiOperator
from folder_locations import get_data_folder, get_results_folder
from init_value_sirt import sirt
from carousel_simulation_experiment import make_delayed_vgs, make_overlapping_vgs_after, make_overlapping_vgs_before_after, make_cylinder_mask

# Preprocesses the projection data without making copies
def preprocess_in_place(y, dark, flat):
    dark = dark[:, None, :]
    flat = flat[:, None, :]
    y -= dark
    y /= (flat - dark)
    torch.log_(y)
    y *= -1

if __name__ == "__main__":
    detector_shape = np.array((500, 956))
    pixel_size = 0.149600
    full_num_proj = 2400
    full_new_object_delay = 400
    skip_proj = 1
    num_proj = full_num_proj // skip_proj
    new_object_delay = full_new_object_delay // skip_proj
    num_objs = 23
    sirt_iterations = 100
    num_gpus = 1
    
    car_mag = 954.145000
    car_tra = 33.22569524099723
    car_rad = 64
    obj_size = 64
    src_det_dist = 1098
    start_angle = -0.15

    
    # Use multiple GPUs if available
    astra.set_gpu_index(list(range(num_gpus)))
    
    scan_path = get_data_folder() / "mandarin_carousel_roi"

    experiment_name = "mandarin_carousel_recons"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name=experiment_name)    

    
    # set up a cone beam projection geometry
    pg = ts.cone(
        shape = detector_shape,
        size = detector_shape*pixel_size,
        src_orig_dist = car_mag,
        src_det_dist = src_det_dist
    )
    # set up a volume geometry
    # This will be moved over the trajectory for each mandarin
    vg = ts.volume(
        shape = (256, 256, 256),
        size = (obj_size, obj_size, obj_size)
    ).to_vec()
    # Set up the trajectory
    tra = ts.translate((0, 0, car_tra+car_rad))
    rot = ts.rotate(pos=(0, 0, car_tra), axis=(-1, 0, 0), angles = np.linspace(start_angle, 2*np.pi+start_angle, num_proj))
    T = rot * tra
    
    # According to the reconstruction strategies together, subtract and
    # submatrix set up the volume geometries needed for the projection operator B
    vgs_full = make_delayed_vgs(vg, T, new_object_delay, num_objs)
    vgs_after = make_overlapping_vgs_after(vg, T, num_proj, new_object_delay)
    vgs_ba = make_overlapping_vgs_before_after(vg, T, num_proj, new_object_delay)
    
    # Make an animation of the geometries and save it
    #s = ts.scale(0.008)
    #animation = animate(*[s * vg for vg in vgs_full], s * pg)
    #animation.save(experiment_folder / "geometry_video.mp4")
    
    # Make projection operators for: one object (A), the together method
    # (B_full), the subtract method (B_after) and the submatrix method (B_ba)
    A = ts.operator(rot * tra * vg, pg)
    B_full = MultiOperator(vgs_full, [pg], detector_supersampling=2, voxel_supersampling=2)
    B_after = MultiOperator(vgs_after, [pg], detector_supersampling=2, voxel_supersampling=2)
    B_ba = MultiOperator(vgs_ba, [pg], detector_supersampling=2, voxel_supersampling=2)

    # Apply flat fielding and log transform the measurements so a linear model
    # can be used in reconstruction
    dark_field = torch.mean(torch.from_numpy(ceu.load_stack(scan_path, dtype=np.float32, prefix="di")), axis=0)
    flat_field = torch.mean(torch.from_numpy(ceu.load_stack(scan_path, dtype=np.float32, prefix="io")), axis=0)
    y = torch.from_numpy(ceu.load_stack(scan_path, stack_axis=1, dtype=np.float32, prefix="scan_",
        range_stop=full_num_proj+((num_objs-1)*full_new_object_delay), range_step=skip_proj))
    preprocess_in_place(y, dark_field, flat_field)
    mask = make_cylinder_mask(256, 256)

    
    # Reconstruct with together method
    recon_together = sirt(B_full, y[None, ...], sirt_iterations, positivity_constraint=True, volume_mask=mask[None, ...])
    for i in range(num_objs):
        ceu.save_stack(experiment_folder / "together" / f"recon_sirt_{i:02d}", recon_together[i, ...], stack_axis=0, parents=True)
    del recon_together
        
    # Reconstruct with submatrix method
    for i in range(num_objs):
        recon_submatrix = sirt(B_ba, y[None, :, i*new_object_delay:i*new_object_delay+num_proj, :], sirt_iterations, positivity_constraint=True, volume_mask=mask[None, ...])
        ceu.save_stack(experiment_folder / "submatrix" / f"recon_sirt_{i:02d}", recon_submatrix[recon_submatrix.shape[0]//2, ...], stack_axis=0, parents=True)
    del recon_submatrix
    
    # Reconstruct with subtract method
    y_curr = y.clone()
    for i in range(num_objs):
        recon_subtract = sirt(B_after, y[None, :, i*new_object_delay:i*new_object_delay+num_proj, :], sirt_iterations, positivity_constraint=True, volume_mask=mask[None, ...])
        ceu.save_stack(experiment_folder / "subtract" / f"recon_sirt_{i:02d}", recon_subtract[0, ...], stack_axis=0, parents=True)
        y_curr[:, i*new_object_delay:i*new_object_delay+num_proj, :] -= A(recon_subtract[0, ...])
    del recon_subtract, y_curr

