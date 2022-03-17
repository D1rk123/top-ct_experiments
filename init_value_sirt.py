# Copied from an older version of the Tomosipo Algorithms library
# (ts_algorithms) and then modified.
# The changes made in this file have been made available in a slightly modified
# form in the newest version of ts_algorithms, so using that version is
# recommended.
# https://github.com/ahendriksen/ts_algorithms

import tomosipo as ts
import torch
import math
import tqdm


def sirt(A, y, num_iterations, x_init=None, positivity_constraint=False, verbose=True, volume_mask=None):
    """Execute the SIRT algorithm

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    :param A: `tomosipo.operator`
    :param y: `torch.tensor`
    :param num_iterations: `int`

    :returns:
    :rtype:

    """
    dev = y.device

    # Compute C
    y_tmp = torch.ones(A.range_shape, device=dev)
    C = A.T(y_tmp)
    C[C < ts.epsilon] = math.inf
    C.reciprocal_()
    # Compute R
    x_tmp = torch.ones(A.domain_shape, device=dev)
    R = A(x_tmp)
    R[R < ts.epsilon] = math.inf
    R.reciprocal_()
    
    if x_init is None:
        x_cur = torch.zeros(A.domain_shape, device=dev)
    else:
        with torch.cuda.device_of(y):
            x_cur = x_init.clone().detach()
            
    if volume_mask is not None:
        x_cur *= volume_mask
        C *= volume_mask

    if verbose:
        iter_range = tqdm.trange(num_iterations)
    else:
        iter_range = range(num_iterations)
    for _ in iter_range:
        A(x_cur, out=y_tmp)
        y_tmp -= y
        y_tmp *= R
        A.T(y_tmp, out=x_tmp)
        x_tmp *= C
        x_cur -= x_tmp
        if positivity_constraint:
            torch.clamp(x_cur, min=0, out=x_cur)

    return x_cur
    
def optimal_iters_sirt(A, y, ground_truth, cost_function, num_iterations, early_stop=None, x_init=None, overwrite_x_init=False, positivity_constraint=False, verbose=True, volume_mask=None, cost_mask=None):
    """Execute the SIRT algorithm

    If `y` is located on GPU, the entire algorithm is executed on a single GPU.

    IF `y` is located in RAM (CPU in PyTorch parlance), then only the
    foward and backprojection are executed on GPU.

    :param A: `tomosipo.operator`
    :param y: `torch.tensor`
    :param num_iterations: `int`

    :returns:
    :rtype:

    """
    dev = y.device

    # Compute C
    y_tmp = torch.ones(A.range_shape, device=dev)
    C = A.T(y_tmp)
    C[C < ts.epsilon] = math.inf
    C.reciprocal_()
    # Compute R
    x_tmp = torch.ones(A.domain_shape, device=dev)
    R = A(x_tmp)
    R[R < ts.epsilon] = math.inf
    R.reciprocal_()
    
    if x_init is None:
        x_cur = torch.zeros(A.domain_shape, device=dev)
    else:
        if overwrite_x_init:
            x_cur = x_init
        else:
            with torch.cuda.device_of(y):
                x_cur = x_init.clone().detach()

    if verbose:
        iter_range = tqdm.trange(num_iterations)
    else:
        iter_range = range(num_iterations)
    
    if early_stop is None:
        early_stop = num_iterations
        
    if volume_mask is not None:
        x_cur *= volume_mask
        C *= volume_mask
        
    cost = []
    x_best = None
    no_improvement = 0
    for _ in iter_range:
        A(x_cur, out=y_tmp)
        y_tmp -= y
        y_tmp *= R
        A.T(y_tmp, out=x_tmp)
        x_tmp *= C
        x_cur -= x_tmp
        if positivity_constraint:
            torch.clamp(x_cur, min=0, out=x_cur)
        
        cost.append(cost_function(x_cur, ground_truth, cost_mask))
        
        if len(cost) > 1 and cost[-1] > cost[-2]:
            no_improvement += 1
            if no_improvement >= early_stop:
                break
        else:
            no_improvement = 0
            
        if cost[-1] == min(cost):
            #print("new best")
            x_best = x_cur.clone()

    return x_best, cost

