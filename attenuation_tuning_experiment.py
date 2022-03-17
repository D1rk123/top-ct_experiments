import numpy as np
import torch
import astra
import tomosipo as ts
import tomosipo.torch_support
from tomosipo.qt import animate
import ts_algorithms as tsa
from matplotlib import pyplot as plt
import math
import ct_experiment_utils as ceu
from multi_operator import MultiOperator
from folder_locations import get_data_folder, get_results_folder
from init_value_sirt import sirt, optimal_iters_sirt

import ct_experiment_utils as ceu
from evaluation import calc_metrics
from carousel_simulation_experiment import make_carousel_geometries, make_delayed_vgs, make_cylinder_mask, load_apples, add_poisson_noise, reconstruct_together, MSE_loss, write_optimal_iters

def reconstruct_seperate_projections(A, x, mask, cost_mask, cost_function, domain_shape, num_iterations, early_stop, photon_count, attenuation_factor):
    recon = torch.zeros(domain_shape, dtype=torch.float32)
    num_objects = domain_shape[0]
    optimal_iters = []
    
    for i in range(num_objects):
        yi = A(x[i, ...]).numpy()
        yi = add_poisson_noise(yi, photon_count=photon_count, attenuation_factor=attenuation_factor)
        yi = torch.from_numpy(yi)
        result, cost = optimal_iters_sirt(
            A,
            yi,
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
    num_objects = 38
    num_iterations = 1000
    early_stop = 30
    num_gpus = 4
    
    photon_count=25000
    
    experiment_name = "38_objects_25000_photons_parameter_tuning"
    experiment_folder = ceu.make_new_experiment_folder(get_results_folder(), name=experiment_name)
    
    astra.set_gpu_index(list(range(num_gpus)))


    vg, pg, T = make_carousel_geometries(obj_size, carousel_radius, src_coc_z_dist, obj_resolution, obj_ver_resolution, dtr_relative_resolution, dtr_height, num_angles)
    
    vgs_full = make_delayed_vgs(vg, T, new_object_delay, num_objects)
    
    # Make an animation of the geometries and save it
    #s = ts.scale(0.4)
    #scaled_vgs = [s * vg for vg in vgs]
    #animation = animate(*scaled_vgs, s * pg)
    #animation.save("geometry_video.mp4")
    # Create an operator from the transformed geometries
    print("making operators")
    A = ts.operator(T*vg, pg)
    B_projections = MultiOperator(vgs_full, [pg], detector_supersampling=3, voxel_supersampling=3)
    B_full = MultiOperator(vgs_full, [pg], detector_supersampling=2, voxel_supersampling=2)
    
    vg_low, pg_low, T_low = make_carousel_geometries(obj_size, carousel_radius, src_coc_z_dist, obj_resolution, obj_ver_resolution, dtr_relative_resolution, dtr_height, new_object_delay)
    A_low = ts.operator(T_low*vg_low, pg_low)
    
    print("making masks")
    x, segmentations = load_apples(num_objects)
    cyl_mask = make_cylinder_mask(obj_resolution, obj_ver_resolution)
    data_range = float(torch.max(x))

    for exponent in np.linspace(-5, 2, num=15):
        attenuation_factor = 90*(2**exponent)
        print(f"attenuation_factor = {attenuation_factor}")
        round_att_fac = round(attenuation_factor)
        y = B_projections(x)
        
        y = add_poisson_noise(y, photon_count=photon_count, attenuation_factor=attenuation_factor)
        y = torch.from_numpy(y)
        #ceu.save_stack(experiment_folder / "projections", y[0, ...], stack_axis=1, exist_ok=True)

        recon, optimal_iters = reconstruct_together(B_full, y, x, cyl_mask, segmentations, cost_function=MSE_loss, num_iterations=num_iterations, early_stop=early_stop)
        for i in range(num_objects):
            ceu.save_stack(experiment_folder / f"together_att{round_att_fac}" / f"recon_{i}", recon[i, ...], stack_axis=0, exist_ok=True, parents=True)
        calc_metrics(
            reference=x.numpy(),
            reconstruction=recon.numpy(),
            mask=segmentations.numpy(),
            data_range=data_range,
            save_path=experiment_folder / f"together_att{round_att_fac}" / "metrics.csv")
        write_optimal_iters(experiment_folder / f"together_att{round_att_fac}" / "optimal_iters.txt", optimal_iters)

    for exponent in np.linspace(-5, 2, num=15):
        attenuation_factor = 90*(2**exponent)
        print(f"attenuation_factor = {attenuation_factor}")
        round_att_fac = round(attenuation_factor)
        recon, optimal_iters = reconstruct_seperate_projections(A_low, x, mask=cyl_mask, cost_mask=segmentations, cost_function=MSE_loss, domain_shape=B_full.domain_shape, num_iterations=num_iterations, early_stop=early_stop, photon_count=photon_count, attenuation_factor=attenuation_factor)
        for i in range(num_objects):
            ceu.save_stack(experiment_folder / f"separate_projections_{new_object_delay}_att{round_att_fac}" / f"recon_{i}", recon[i, ...], stack_axis=0, exist_ok=True, parents=True)
        calc_metrics(
            reference=x.numpy(),
            reconstruction=recon.numpy(),
            mask=segmentations.numpy(),
            data_range=data_range,
            save_path=experiment_folder / f"separate_projections_{new_object_delay}_att{round_att_fac}" / "metrics.csv")
        write_optimal_iters(experiment_folder / f"separate_projections_{new_object_delay}_att{round_att_fac}" / "optimal_iters.txt", optimal_iters)
    
