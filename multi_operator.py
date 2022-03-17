import tomosipo as ts
import tomosipo.torch_support
import torch
import numpy as np
from tomosipo.Operator import BackprojectionOperator

class MultiOperator:
    """
    Class for handling matrix operations for CT scanning geometries with
    overlapping projections.
    
    The interface is nearly identical to the interface of a tomosipo.Operator.
    After creating a MultiOperator A, a Torch tensor or numpy array containing
    multiple volumes can be projected onto multiple detectors, resulting in an
    object of the same type as the input containing the projection data.
    Because MultiOperator supports multiple objects and multiple detectors, an
    extra dimension is added at the beginning of both the volume and the
    projection data indicating the volume or detector.
    
    Forward projections are applied as follows:
    y = A(x) -or- A(x, out=y)
    Backprojections are applied as follows:
    b = A.T(y) -or- A.T(y, out=b)
    """

    def __init__(self, vgs, pgs, voxel_supersampling=1, detector_supersampling=1, additive=False):
        self._vgs = vgs
        self._pgs = pgs
        self._voxel_supersampling = voxel_supersampling
        self._detector_supersampling = detector_supersampling
        self._additive=additive
        
        self._setup_operators()
        self._determine_range_shape()
        self._determine_domain_shape()
        
        self._transpose = BackprojectionOperator(self)
        
    def _setup_operators(self):
        self._ops = []
        for pg in self._pgs:
            ops = []
            for vg in self._vgs:
                ops.append(ts.operator(vg, pg, voxel_supersampling=self._voxel_supersampling, detector_supersampling=self._detector_supersampling, additive=True))
            self._ops.append(ops)

    def _determine_domain_shape(self):
        self._domain_shape = (len(self._vgs), ) + self._vgs[0].shape
        all_same = True
        
        for op_list in self._ops:
            for op in op_list:
                for i in range(3):
                    if self._domain_shape[i+1] != op.domain_shape[i]:
                        all_same = False
                        if self._domain_shape[i+1] < op.domain_shape[i]:
                            self._domain_shape[i+1] = op.domain_shape[i]
                            
        if not all_same:
            print("Warning: not all internal operators have the same domain.")
            print("This will work, but the MultiOperator will use the amount")
            print("of memory of the largest domain for all internal operators.")
            print(f"The resulting domain shape is {self._domain_shape}")
            
    def _determine_range_shape(self):
        self._range_shape = (len(self._pgs), ) + self._ops[0][0].range_shape
        all_same = True
        
        for op_list in self._ops:
            for op in op_list:
                for i in range(3):
                    if self._range_shape[i+1] != op.range_shape[i]:
                        all_same = False
                        if self._range_shape[i+1] < op.range_shape[i]:
                            self._range_shape[i+1] = op.range_shape[i]
                
        if not all_same:
            print("Warning: not all internal operators have the same range.")
            print("This will work, but the MultiOperator will use the amount")
            print("of memory of the largest range for all internal operators.")
            print(f"The resulting range shape is {self._range_shape}")
    
    def _make_valid_out(self, x, out, shape):
        if out is None:
            if isinstance(x, torch.Tensor):
                out = torch.zeros(shape, dtype=x.dtype, device=x.device)
            else:
                out = np.zeros(shape, dtype=x.dtype)
        elif not self._additive:
            if isinstance(out, torch.Tensor):
                out.fill_(0)
            else:
                out.fill(0)
        return out
        
    def _fp(self, x, out=None):
        out = self._make_valid_out(x, out, self._range_shape)
        
        #note that all operators are created to be additive, so all results are added
        for i, op_list in enumerate(self._ops):    
            for j, op in enumerate(op_list):
                op(x[j, :op.domain_shape[0], :op.domain_shape[1], :op.domain_shape[2]],
                   out[i, :op.range_shape[0], :op.range_shape[1], :op.range_shape[2]])
        return out
        
    def _bp(self, y, out=None):
        out = self._make_valid_out(y, out, self._domain_shape)
        
        #note that all operators are created to be additive, so all results are added
        for i, op_list in enumerate(self._ops):    
            for j, op in enumerate(op_list):
                op.T(y[i, :op.range_shape[0], :op.range_shape[1], :op.range_shape[2]],
                     out[j, :op.domain_shape[0], :op.domain_shape[1], :op.domain_shape[2]])
        return out
    
    def get_operator(self, pg_num, vg_num):
        return self._ops[pg_num][vg_num]
    
    def __call__(self, volume, out=None):
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        return self.transpose()

    @property
    def domain(self):
        return self._vgs
        
    @property
    def range(self):
        return self._pgs

    @property
    def domain_shape(self):
        return self._domain_shape

    @property
    def range_shape(self):
        return self._range_shape
    
    
