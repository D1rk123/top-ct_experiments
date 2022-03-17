import numpy as np
import torch
import astra
import tomosipo as ts
import tomosipo.torch_support
from matplotlib import pyplot as plt
import math
import ct_experiment_utils as ceu
from multi_operator import MultiOperator
from folder_locations import get_data_folder, get_results_folder
from init_value_sirt import sirt, optimal_iters_sirt

import ct_experiment_utils as ceu
from evaluation import calc_metrics
from make_metrics_overview import make_metrics_overview

def make_carousel_geometries(obj_size, carousel_radius, src_coc_z_dist, obj_resolution, obj_ver_resolution, dtr_relative_resolution, dtr_height, num_angles):
    half_obj_size = obj_size/2
    carousel_outer_radius = carousel_radius+half_obj_size
    
    src_coc_dist = math.sqrt(src_coc_z_dist**2+half_obj_size**2)
    cone_angle = math.asin(carousel_outer_radius/src_coc_dist)+math.asin(half_obj_size/src_coc_dist)
    half_cone_angle = cone_angle/2
    
    src_dtr_dist = math.cos(half_cone_angle-math.acos(src_coc_z_dist/src_coc_dist))*src_coc_dist + carousel_outer_radius
    half_dtr_width = src_dtr_dist * math.tan(half_cone_angle)
    dtr_width = half_dtr_width*2

    print(f"cone_angle = {math.degrees(cone_angle)}")
    print(f"src_dtr_dist = {src_dtr_dist}")
    print(f"dtr_width = {dtr_width}")
    
    trajectory_length = 2*obj_size+np.pi*carousel_radius
    trajectory_range = np.linspace(0, trajectory_length, num_angles)
    
    pre_tra_range = trajectory_range[trajectory_range<=obj_size]
    rot_range = (trajectory_range[np.logical_and(
        trajectory_range > obj_size,
        trajectory_range <= trajectory_length-obj_size
    )]-obj_size)/carousel_radius
    post_tra_range = trajectory_length - trajectory_range[trajectory_range>trajectory_length-obj_size]
    print(f"minimum amount of spacing required = {math.ceil(len(rot_range)/((np.pi/2) / math.atan(obj_size/(2*carousel_radius))))}")
    
    angles_range = np.linspace(0, np.pi, num_angles)
    
    vg = ts.volume(shape=(obj_ver_resolution, obj_resolution, obj_resolution), size=(obj_size*(obj_ver_resolution/obj_resolution), obj_size, obj_size))
    vg_transform_pre_tra = ts.translate((0, src_coc_z_dist-carousel_radius, -half_obj_size)) * \
        ts.translate((0, 0, 1), alpha=pre_tra_range)
    vg_transform_rot = ts.translate((0, src_coc_z_dist, half_obj_size)) * \
        ts.rotate(pos=(0,0,0), axis=(1, 0, 0), angles=rot_range) * \
        ts.translate((0,-carousel_radius,0))
    vg_transform_post_tra = ts.translate((0, src_coc_z_dist+carousel_radius, -half_obj_size)) * \
        ts.translate((0, 0, 1), alpha=post_tra_range) * \
        ts.rotate(pos=(0,0,0), axis=(1, 0, 0), angles=np.pi)
    vg_transform = ts.concatenate([
        vg_transform_pre_tra, vg_transform_rot, vg_transform_post_tra
    ])
        
    vg = vg.to_vec()
    
    dtr_shape = np.ceil(np.array([dtr_height, dtr_width])*dtr_relative_resolution).astype(int)+np.array([0, 2])
    pg = ts.rotate(pos=0, axis=(1, 0, 0), angles=(-half_cone_angle,)) * \
        ts.cone(shape=dtr_shape, size=dtr_shape/dtr_relative_resolution, src_orig_dist=0, src_det_dist=src_dtr_dist).to_vec()
    return vg, pg, vg_transform
    
def make_delayed_vgs(vg, T, delay, num):
    vgs = []
    for i in range(num):
        pre_delay_T = ts.translate(np.ones((delay*i, 3))*-100)
        post_delay_T = ts.translate(np.ones((delay*(num-i-1), 3))*-100)
        vgs.append(ts.concatenate([pre_delay_T, T, post_delay_T]) * vg)
    return vgs
    
def make_overlapping_vgs_after(vg, T, num_angles, delay):
    vgs = []
    num_other_objs = math.ceil(num_angles / delay)
    for i in range(num_other_objs):
        pre_delay_length = delay*i
        pre_delay_T = ts.translate(np.ones((pre_delay_length, 3))*-100)
        vgs.append(ts.concatenate([pre_delay_T, T[:num_angles-pre_delay_length]]) * vg)
    return vgs
    
def make_overlapping_vgs_before_after(vg, T, num_angles, delay):
    vgs = []
    num_other_objs = math.ceil(num_angles / delay)
    for i in reversed(range(1, num_other_objs)):
        post_delay_length = delay*i
        post_delay_T = ts.translate(np.ones((post_delay_length, 3))*-100)
        vgs.append(ts.concatenate([T[post_delay_length:], post_delay_T]) * vg)
    for i in range(num_other_objs):
        pre_delay_length = delay*i
        pre_delay_T = ts.translate(np.ones((pre_delay_length, 3))*-100)
        vgs.append(ts.concatenate([pre_delay_T, T[:num_angles-pre_delay_length]]) * vg)
    return vgs

def load_apples(num_objects):
    data_folder = get_data_folder() / "preprocessed_apples"

    selected_indices = np.arange(0, num_objects)+1#np.random.choice(111, num_objects, replace=False)+1
    x = np.zeros((num_objects, 237, 256, 256), dtype=np.float32)
    segmentations = np.zeros((num_objects, 237, 256, 256), dtype=np.bool_)
    
    for i in range(num_objects):
        x[i, ...] = ceu.load_stack(
            data_folder / "downsampled_256_recons" / f"apple{selected_indices[i]:03d}",
            prefix="output",
            dtype=np.float32
        )
        segmentations[i, ...] = ceu.load_stack(
            data_folder / "downsampled_256_masks" / f"mask{selected_indices[i]:03d}",
            prefix="output",
            dtype=np.bool_
        )
        
    x *= segmentations
    np.clip(x, a_min=0, a_max=None, out=x)
    return torch.from_numpy(x), torch.from_numpy(segmentations)

def reconstruct_together(B, y, x, mask, cost_mask, cost_function, num_iterations, early_stop):
    recon, cost = optimal_iters_sirt(
        B,
        y,
        ground_truth=x,
        cost_function=cost_function,
        cost_mask=cost_mask,
        num_iterations=num_iterations,
        early_stop=early_stop,
        positivity_constraint=True,
        verbose=True,
        volume_mask=mask[None, ...]
    )
    #print(f"Finished SIRT, with optimal cost after {np.argmin(cost)+1} iterations")
    #plt.plot(range(len(cost)), cost)
    #plt.show()
    return recon, [np.argmin(cost)]
    
def reconstruct_subtract(A, B_after, y, x, mask, cost_mask, cost_function, domain_shape, num_angles, new_object_delay, num_iterations, early_stop):
    y_curr = y.clone()
    recon = torch.zeros(domain_shape, dtype=torch.float32)
    num_objects = domain_shape[0]
    optimal_iters = []
    
    def cost_function_first(x_cur, ground_truth, mask):
        return cost_function(x_cur[0, ...], ground_truth, mask)
        
    for i in range(num_objects):
        result, cost = optimal_iters_sirt(
            B_after,
            y_curr[:, :, i*new_object_delay:i*new_object_delay+num_angles, :],
            ground_truth=x[i, ...],
            cost_function=cost_function_first,
            cost_mask=cost_mask[i, ...],
            num_iterations=num_iterations,
            early_stop=early_stop,
            positivity_constraint=True,
            verbose=True,
            volume_mask=mask[None, ...]
        )
        recon[i, ...] = result[0, ...]
        y_curr[:, :, i*new_object_delay:i*new_object_delay+num_angles, :] -= A(recon[i, ...])
        optimal_iters.append(np.argmin(cost))
    return recon, optimal_iters


def reconstruct_submatrix(B_ba, y, x, mask, cost_mask, cost_function, domain_shape, num_angles, new_object_delay, num_iterations, early_stop):
    recon = torch.zeros(domain_shape, dtype=torch.float32)
    num_objects = domain_shape[0]
    optimal_iters = []
    
    def cost_function_middle(x_cur, ground_truth, mask):
        return cost_function(x_cur[B_ba.domain_shape[0]//2, ...], ground_truth, mask)
    
    for i in range(num_objects):
        result, cost = optimal_iters_sirt(
            B_ba,
            y[:, :, i*new_object_delay:i*new_object_delay+num_angles, :],
            ground_truth=x[i, ...],
            cost_function=cost_function_middle,
            cost_mask=cost_mask[i, ...],
            num_iterations=num_iterations,
            early_stop=early_stop,
            positivity_constraint=True,
            verbose=True,
            volume_mask=mask[None, ...]
        )
        #only use the middle element of each SIRT result
        recon[i, ...] = result[result.shape[0]//2, ...]
        optimal_iters.append(np.argmin(cost))
    return recon, optimal_iters
    
def reconstruct_ignore(A, y, x, mask, cost_mask, cost_function, domain_shape, num_angles, new_object_delay, num_iterations, early_stop):
    recon = torch.zeros(domain_shape, dtype=torch.float32)
    num_objects = domain_shape[0]
    optimal_iters = []
    
    for i in range(num_objects):
        result, cost = optimal_iters_sirt(
            A,
            y[0, :, i*new_object_delay:i*new_object_delay+num_angles, :],
            ground_truth=x[i, ...],
            cost_function=cost_function,
            cost_mask=cost_mask[i, ...],
            num_iterations=num_iterations,
            early_stop=early_stop,
            positivity_constraint=True,
            verbose=True,
            volume_mask=mask
        )
        recon[i, ...] = result
        optimal_iters.append(np.argmin(cost))
    return recon, optimal_iters
    
def reconstruct_ignore_100_iters(A, y, mask, domain_shape, num_angles, new_object_delay):
    recon = torch.zeros(domain_shape, dtype=torch.float32)
    num_objects = domain_shape[0]
    
    for i in range(num_objects):
        result = sirt(
            A,
            y[0, :, i*new_object_delay:i*new_object_delay+num_angles, :],
            num_iterations=100,
            positivity_constraint=True,
            verbose=True,
            volume_mask=mask
        )
        recon[i, ...] = result
    return recon


def add_poisson_noise(img, photon_count, attenuation_factor=1):
    img = img * attenuation_factor
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    img *= photon_count
    # Add poisson noise and retain scale by dividing by photon_count
    img = np.random.poisson(img)
    img = img / photon_count
    img[img == 0] = 1
    # Redo log transform and scale img to range [0, img_max] +- some noise.
    img = -np.log(img, **opt)
    return img / attenuation_factor
    
def make_cylinder_mask(diameter, height):
    xx, yy = np.mgrid[:diameter, :diameter]
    
    center_dist = (xx - diameter/2)**2 + (yy - diameter/2)**2
    circle = center_dist <= ((diameter/2)+0.5)**2
    
    cylinder = np.tile(circle, (height, 1, 1))
    return torch.from_numpy(cylinder)


def MSE_loss(x, y, mask):
    return torch.mean(((x-y)*mask)**2)
    
def write_optimal_iters(path, optimal_iters):
    with open(path, "w") as costs_file:
        for optimal_iter in optimal_iters:
            costs_file.write(f"{optimal_iter}\n")

if __name__ == "__main__":
    #obj = object, src = source, dtr = detector, coc = center of carousel
    obj_size = 1
    carousel_radius = 2
    src_coc_z_dist = 8
    
    obj_resolution = 256
    obj_ver_resolution = 237
    dtr_relative_resolution = 256
    dtr_height = 2
    num_angles = 400
    new_object_delay = 48
    num_objects = 68
    num_iterations = 1000
    early_stop = 30
    num_gpus = 4
    
    overview_include_low = 9
    overview_include_high = 59
    
    photon_count=25000
    attenuation_factor=90/(2**1.5)
    
    experiment_name = "68_objects_25000_photons_method_comparison"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name=experiment_name)
    
    astra.set_gpu_index(list(range(num_gpus)))


    vg, pg, T = make_carousel_geometries(obj_size, carousel_radius, src_coc_z_dist, obj_resolution, obj_ver_resolution, dtr_relative_resolution, dtr_height, num_angles)
    
    vgs_full = make_delayed_vgs(vg, T, new_object_delay, num_objects)
    vgs_after = make_overlapping_vgs_after(vg, T, num_angles, new_object_delay)
    vgs_ba = make_overlapping_vgs_before_after(vg, T, num_angles, new_object_delay)
    
    # Create an operator from the transformed geometries
    print("making operators")
    A = ts.operator(T*vg, pg)
    B_projections = MultiOperator(vgs_full, [pg], detector_supersampling=3, voxel_supersampling=3)
    B_full = MultiOperator(vgs_full, [pg], detector_supersampling=2, voxel_supersampling=2)
    B_after = MultiOperator(vgs_after, [pg], detector_supersampling=2, voxel_supersampling=2)
    B_ba = MultiOperator(vgs_ba, [pg], detector_supersampling=2, voxel_supersampling=2)
    
    vg_low, pg_low, T_low = make_carousel_geometries(obj_size, carousel_radius, src_coc_z_dist, obj_resolution, obj_ver_resolution, dtr_relative_resolution, dtr_height, new_object_delay)
    A_low = ts.operator(T_low*vg_low, pg_low)
    
    print("making masks")
    x, segmentations = load_apples(num_objects)
    cyl_mask = make_cylinder_mask(obj_resolution, obj_ver_resolution)

    print("making projections")
    y = B_projections(x)
    
    print("adding noise")
    y = add_poisson_noise(y, photon_count=photon_count, attenuation_factor=attenuation_factor)
    y = torch.from_numpy(y)
    ceu.save_stack(experiment_folder / "projections", y[0, ...], stack_axis=1, exist_ok=True)
    
    #y = ceu.load_stack(get_results_folder() / "2021-12-22_68_objects_25000_photons_3" / "projections", stack_axis=1, dtype="float")
    #y = torch.from_numpy(y[None, ...])
    
    data_range = float(torch.max(x))

    recon, optimal_iters = reconstruct_together(B_full, y, x, cyl_mask, segmentations, cost_function=MSE_loss, num_iterations=num_iterations, early_stop=early_stop)
    for i in range(num_objects):
        ceu.save_stack(experiment_folder / "together" / f"recon_{i}", recon[i, ...], stack_axis=0, exist_ok=True, parents=True)
    calc_metrics(
        reference=x.numpy(),
        reconstruction=recon.numpy(),
        mask=segmentations.numpy(),
        data_range=data_range,
        save_path=experiment_folder / "together" / "metrics.csv")
    write_optimal_iters(experiment_folder / "together" / "optimal_iters.txt", optimal_iters)
    
    recon, optimal_iters = reconstruct_subtract(A, B_after, y, x, cyl_mask, segmentations, MSE_loss, B_full.domain_shape, num_angles, new_object_delay, num_iterations=num_iterations, early_stop=early_stop)
    for i in range(num_objects):
        ceu.save_stack(experiment_folder / "subtract" / f"recon_{i}", recon[i, ...], stack_axis=0, exist_ok=True, parents=True)
    calc_metrics(
        reference=x.numpy(),
        reconstruction=recon.numpy(),
        mask=segmentations.numpy(),
        data_range=data_range,
        save_path=experiment_folder / "subtract" / "metrics.csv")
    write_optimal_iters(experiment_folder / "subtract" / "optimal_iters.txt", optimal_iters)

    recon, optimal_iters = reconstruct_submatrix(B_ba, y, x, cyl_mask, segmentations, MSE_loss, B_full.domain_shape, num_angles, new_object_delay, num_iterations=num_iterations, early_stop=early_stop)
    for i in range(num_objects):
        ceu.save_stack(experiment_folder / "submatrix" / f"recon_{i}", recon[i, ...], stack_axis=0, exist_ok=True, parents=True)
    calc_metrics(
        reference=x.numpy(),
        reconstruction=recon.numpy(),
        mask=segmentations.numpy(),
        data_range=data_range,
        save_path=experiment_folder / "submatrix" / "metrics.csv")
    write_optimal_iters(experiment_folder / "submatrix" / "optimal_iters.txt", optimal_iters)
        
    recon, optimal_iters = reconstruct_ignore(A, y, x, cyl_mask, segmentations, MSE_loss, B_full.domain_shape, num_angles, new_object_delay, num_iterations=num_iterations, early_stop=early_stop)
    for i in range(num_objects):
        ceu.save_stack(experiment_folder / "ignore" / f"recon_{i}", recon[i, ...], stack_axis=0, exist_ok=True, parents=True)
    calc_metrics(
        reference=x.numpy(),
        reconstruction=recon.numpy(),
        mask=segmentations.numpy(),
        data_range=data_range,
        save_path=experiment_folder / "ignore" / "metrics.csv")
    write_optimal_iters(experiment_folder / "ignore" / "optimal_iters.txt", optimal_iters)
    
    recon = reconstruct_ignore_100_iters(A, y, cyl_mask, B_full.domain_shape, num_angles, new_object_delay)
    for i in range(num_objects):
        ceu.save_stack(experiment_folder / "ignore_100_iters" / f"recon_{i}", recon[i, ...], stack_axis=0, exist_ok=True, parents=True)
    calc_metrics(
        reference=x.numpy(),
        reconstruction=recon.numpy(),
        mask=segmentations.numpy(),
        data_range=data_range,
        save_path=experiment_folder / "ignore_100_iters" / "metrics.csv")
    
    make_metrics_overview(lo=overview_include_low, hi=overview_include_high, experiment_folder=experiment_folder)
