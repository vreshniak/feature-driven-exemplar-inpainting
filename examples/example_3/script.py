from pathlib import Path

from skimage.color import grey2rgb
from skimage.io import imread, imsave
from skimage.morphology import binary_dilation
from skimage import img_as_uint, img_as_float

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from   inpainting import Inpainting
import inpainting.utils as op


#############################################################################
# create directory for the output


# create directory for the output
Path("output/").mkdir(parents=True, exist_ok=True)


#############################################################################
# patch shape and weight

patch_shape = (15,15)
patch_weight = op.gauss_weight(patch_shape,patch_sigma=5).reshape(patch_shape)
TOL=1.e-4


#############################################################################
# image and mask

image = img_as_float(imread('./data/image.png')[:,:,0:3])
mask  = imread("./data/mask.png",as_gray=True).astype(np.bool)

#############################################################################
# example setup

nonlocal_target_mask = binary_dilation( binary_dilation( mask, selem=np.ones((2,2)) ), selem=np.ones(patch_shape) )
nonlocal_source_mask = nonlocal_target_mask.copy()
nonlocal_source_mask[patch_shape[0]//2:-patch_shape[0]//2,patch_shape[0]//2:-patch_shape[0]//2]  = np.logical_not(nonlocal_target_mask[patch_shape[0]//2:-patch_shape[0]//2,patch_shape[0]//2:-patch_shape[0]//2])
image_setup = image.copy()
# image_setup[mask,0] = 0.95
# image_setup[mask,1] = 0.80
image_setup[mask,0] = 0.75
image_setup[mask,1] = 0.0
image_setup[mask,2] = 0.0
image_setup[nonlocal_source_mask,0] = 0.29
image_setup[nonlocal_source_mask,1] = 0.65
image_setup[nonlocal_source_mask,2] = 0.12
alpha = 0.5
image_setup[mask,:]                 = ((1-alpha)*image + alpha*image_setup)[mask,:]
image_setup[nonlocal_source_mask,:] = ((1-alpha)*image + alpha*image_setup)[nonlocal_source_mask,:]
imsave("./output/example_setup.png", op.add_patch(image_setup, patch_weight))

masked_chair = image.copy()
# masked_chair[mask,:] = ((1-alpha)*image + alpha*mask.astype(np.float)[:,:,np.newaxis])[mask,:]
masked_chair[mask,0] = (1-alpha)*image[mask,0] + alpha*mask.astype(np.float)[mask]
imsave("./output/masked_chair.png", masked_chair)

inp_region = np.zeros_like(image)
inp_region[...] = 0.75
inp_region[mask,:] = 1
imsave("./output/chair_inp_region.png", inp_region)

conv_mask = binary_dilation( mask, selem=np.ones(patch_shape) )
nonlocal_target_mask = binary_dilation( conv_mask, selem=np.ones(patch_shape) )
ext_inp_region = np.zeros_like(image)
ext_inp_region[...] = 0.75
ext_inp_region[nonlocal_target_mask,:] = 0.
ext_inp_region[nonlocal_target_mask,2] = 0.9
ext_inp_region[conv_mask,:] = 0
ext_inp_region[conv_mask,1] = 0.9
ext_inp_region[mask,:] = 0.5
ext_inp_region[mask,0] = 0.95
ext_inp_region[20:50,20:50,:] = 0.5
ext_inp_region[20:50,20:50,0] = 0.95
ext_inp_region[60:90,20:50,:] = 0
ext_inp_region[60:90,20:50,1] = 0.9
ext_inp_region[100:130,20:50,:] = 0
ext_inp_region[100:130,20:50,2] = 0.9
imsave("./output/chair_ext_inp_region.png", ext_inp_region)
# exit()


#############################################################################
# make lambda from edges

asympt_val = 0.1
decay_time = 10

edges = imread("./data/edges.png",as_gray=True).astype(np.bool)
edge_coef  = (1-asympt_val) * np.exp(-ndi.distance_transform_edt(1-edges)/decay_time) + asympt_val
edge_coef  = 1 - edge_coef
edge_coef /= np.amax(edge_coef)
edge_coef[edge_coef<1.e-6] = 1.e-4
imsave("./output/edges_coef.png",img_as_float(edge_coef))

color_edges = image.copy()
color_edges[mask,:]  = 1
color_edges[edges,0] = 0
color_edges[edges,1] = 0.4
color_edges[edges,2] = 1
color_edges[np.logical_and(edges,mask),0] = 1
color_edges[np.logical_and(edges,mask),1] = 0
color_edges[np.logical_and(edges,mask),2] = 0
imsave("./output/color_edges.png", color_edges)
# exit()


#############################################################################
# Patch nonlocal means

kernels = [[[1]]]
lambdas = [1]
problem = Inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight)
result  = problem.process(num_scales=1, initialization='harmonic', TOL=TOL)
imsave("./output/means.png", op.add_patch(result, patch_weight))
# exit()


#############################################################################
# Local PDE inpainting

harmonic = image.copy()
Inpainting.inpaint_PDE(None, np.moveaxis(harmonic,-1,0), mask, 'harmonic')
imsave("./output/harmonic.png", harmonic)
# exit()

# with edge completion
harmonic_edges = image.copy()
Inpainting.inpaint_PDE(None, np.moveaxis(harmonic_edges,-1,0), mask, 'harmonic', edge_coef)
imsave("./output/harmonic_edges.png", harmonic_edges)
# exit()


#############################################################################
# Patch nonlocal Poisson

kernels = op.grad_kernels("forward")
lambdas = [1.0,1.0]
problem = Inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight)
result  = problem.process(num_scales=1, initialization=harmonic, TOL=TOL)
imsave("./output/nonloc_harmonic.png", op.add_patch(result, patch_weight))
# exit()

# with edge completion
kernels = op.grad_kernels("forward")
lambdas = [edge_coef,edge_coef]
problem = Inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight)
result  = problem.process(num_scales=1, initialization=harmonic_edges, TOL=TOL)
imsave("./output/nonloc_harmonic_edges.png", op.add_patch(result, patch_weight))
# exit()