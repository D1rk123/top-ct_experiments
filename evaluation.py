from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from skimage.util.arraycrop import crop

def extend_cbar_labels(cbar):
    ticks = [t.get_text() for t in cbar.ax.get_yticklabels()]
    ticks[0] = "<" + ticks[0]
    ticks[-1] = ">" + ticks[-1]
    cbar.ax.set_yticklabels(ticks)
    
def masked_ssim(reference, reconstruction, mask, data_range):
    _, ssim_image = structural_similarity(
        reference,
        reconstruction,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        full = True
    )
    # crop 5 pixels from boundary of image to avoid edge effects caused by the 11x11 gausian filter
    # this is consistent with skimage
    crop_ssim_image = crop(ssim_image, 5)
    crop_mask = crop(mask, 5)
    inv_crop_mask = 1 - crop_mask
    
    total_ssim = np.mean(crop_ssim_image)
    in_mask_ssim = np.sum(crop_ssim_image*crop_mask)/np.sum(crop_mask)
    out_mask_ssim = np.sum(crop_ssim_image*inv_crop_mask)/np.sum(inv_crop_mask)
    return total_ssim, in_mask_ssim, out_mask_ssim
    
def mean_masked_slice_ssim(reference, reconstruction, mask, data_range):
    total_ssim_sum = 0
    in_mask_ssim_sum = 0
    out_mask_ssim_sum = 0
    crop_mask_sum = 0
    inv_crop_mask_sum = 0
    num_slices = reference.shape[0]
    
    for i in range(num_slices):
        _, ssim_image = structural_similarity(
            reference[i, ...],
            reconstruction[i, ...],
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            full = True
        )
        # crop 5 pixels from boundary of image to avoid edge effects caused by the 11x11 gausian filter
        # this is consistent with skimage
        crop_ssim_image = crop(ssim_image, 5)
        crop_mask = crop(mask[i, ...], 5)
        inv_crop_mask = 1 - crop_mask
        
        total_ssim_sum += np.mean(crop_ssim_image)
        in_mask_ssim_sum += np.sum(crop_ssim_image*crop_mask)
        out_mask_ssim_sum += np.sum(crop_ssim_image*inv_crop_mask)
        crop_mask_sum += np.sum(crop_mask)
        inv_crop_mask_sum += np.sum(inv_crop_mask)
    
    return total_ssim_sum/num_slices, in_mask_ssim_sum/crop_mask_sum, out_mask_ssim_sum/inv_crop_mask_sum
    
def masked_psnr(reference, reconstruction, mask, data_range):
    squared_error = (reference - reconstruction)**2
    inv_mask = 1 - mask
    
    total_psnr = 10 * np.log10((data_range ** 2) / np.mean(squared_error))
    in_mask_psnr = 10 * np.log10((data_range ** 2) / (np.sum(squared_error*mask)/np.sum(mask)))
    out_mask_psnr = 10 * np.log10((data_range ** 2) / (np.sum(squared_error*inv_mask)/np.sum(inv_mask)))
    return total_psnr, in_mask_psnr, out_mask_psnr
    
def calc_metrics(reference, reconstruction, mask, data_range, save_path):
    total_ssim_list = []
    in_mask_ssim_list = []
    out_mask_ssim_list = []
    for i in range(reference.shape[0]):
        total_ssim, in_mask_ssim, out_mask_ssim = mean_masked_slice_ssim(reference[i, ...], reconstruction[i, ...], mask[i, ...], data_range)
        
        total_ssim_list.append(total_ssim)
        in_mask_ssim_list.append(in_mask_ssim)
        out_mask_ssim_list.append(out_mask_ssim)
    
    total_psnr_list = []
    in_mask_psnr_list = []
    out_mask_psnr_list = []
    for i in range(reference.shape[0]):
        total_psnr, in_mask_psnr, out_mask_psnr = masked_psnr(reference[i, ...], reconstruction[i, ...], mask[i, ...], data_range)

        total_psnr_list.append(total_psnr)
        in_mask_psnr_list.append(in_mask_psnr)
        out_mask_psnr_list.append(out_mask_psnr)
    
    
    with open(save_path, "w") as metrics_file:
        metrics_file.write("total_psnr,in_mask_psnr,out_mask_psnr,total_ssim,in_mask_ssim,out_mask_ssim\n")
        for i in range(reference.shape[0]):
            metrics_file.write(f"{total_psnr_list[i]},{in_mask_psnr_list[i]},{out_mask_psnr_list[i]},{total_ssim_list[i]},{in_mask_ssim_list[i]},{out_mask_ssim_list[i]}\n")

