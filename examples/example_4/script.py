import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from skimage.exposure  import rescale_intensity
from skimage.color     import grey2rgb, rgb2grey
from skimage.color import grey2rgb
from skimage.io import imread, imsave
from skimage.morphology import binary_dilation
from skimage import img_as_uint, img_as_float

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

import inpainting as inp
import features   as feat
import utils      as op



from scipy.signal import correlate2d, convolve2d
def mollify_kernels(kernels, sigma):
	if sigma==0: return kernels
	gauss = op.gauss2d(sigma=(sigma,sigma), order=(0,0), angle=0, nstd=0.67, normalize=True)
	return [ correlate2d(np.array(ker), np.array(gauss), mode='full') for ker in kernels ]





#############################################################################
# create directory for the output

output_dir = "output/"

# create directory for the output
Path(output_dir).mkdir(parents=True, exist_ok=True)


#############################################################################
# patch shape and weight

patch_shape = (15,15)
patch_weight = op.gauss_weight(patch_shape,patch_sigma=5).reshape(patch_shape)
TOL = 1.e-5


#############################################################################
# feature kernels

kernels = op.grad_kernels("forward") + [[[1]]]


#############################################################################
# image and mask

image   = img_as_float(imread('./data/image.png')[:,:,0:3])
mask    = imread("./data/mask.png",as_gray=True).astype(np.bool)
house   = imread("./data/house.png",as_gray=True).astype(np.bool)
terrain = np.logical_not(house)


#############################################################################
# example setup

nonlocal_target_mask = binary_dilation( binary_dilation( mask, selem=np.ones((2,2)) ), selem=np.ones(patch_shape) )
nonlocal_source_mask = nonlocal_target_mask.copy()
nonlocal_source_mask[patch_shape[0]//2:,patch_shape[0]//2:-patch_shape[0]//2]  = np.logical_not(nonlocal_target_mask[patch_shape[0]//2:,patch_shape[0]//2:-patch_shape[0]//2])
image_setup = image.copy()
# image_setup[mask,0] = 0.95
# image_setup[mask,1] = 0.80
# image_setup[mask,2] = 0.0
image_setup[mask,0] = 0.75
image_setup[mask,1] = 0.0
image_setup[mask,2] = 0.0
image_setup[nonlocal_source_mask,0] = 0.29
image_setup[nonlocal_source_mask,1] = 0.65
image_setup[nonlocal_source_mask,2] = 0.12
alpha = 0.5
image_setup[mask,:]                 = ((1-alpha)*image + alpha*image_setup)[mask,:]
image_setup[nonlocal_source_mask,:] = ((1-alpha)*image + alpha*image_setup)[nonlocal_source_mask,:]
# image_setup[mask,:] = 1.0
imsave(output_dir+"example_setup.png", op.add_patch(image_setup, patch_weight))
# exit()


betas_mask = image.copy()
betas_mask[house,0] = 1.0
betas_mask[house,1] = 0.0
betas_mask[house,2] = 0.0
betas_mask[terrain,0] = 0.26
betas_mask[terrain,1] = 0.31
betas_mask[terrain,2] = 0.71
alpha = 0.6
# betas_mask[mask,:] = ((1-alpha)*betas_mask + alpha*grey2rgb(mask))[mask,:]
imsave(output_dir+"betas_mask.png", betas_mask)
# exit()


#############################################################################
# make lambda from edges

asympt_val = 0.1
decay_time = 10.0

edges = imread("./data/edges.png",as_gray=True).astype(np.bool)
edge_coef  = (1-asympt_val) * np.exp(-ndi.distance_transform_edt(1-edges)/decay_time) + asympt_val
edge_coef  = 1 - edge_coef
edge_coef /= np.amax(edge_coef)
edge_coef[edge_coef<1.e-6] = 1.e-4
imsave(output_dir+"edges_coef.png",img_as_float(edge_coef))

color_edges = image.copy()
color_edges[mask,:]  = 1
color_edges[edges,0] = 0
color_edges[edges,1] = 0.4
color_edges[edges,2] = 1
color_edges[np.logical_and(edges,mask),0] = 1
color_edges[np.logical_and(edges,mask),1] = 0
color_edges[np.logical_and(edges,mask),2] = 0
imsave(output_dir+"color_edges.png", color_edges)
# exit()



#############################################################################
# Local PDE inpainting with edge completion

img = image.copy()
inp.inpainting.inpaint_PDE(None, np.moveaxis(img,-1,0), mask, 'harmonic', edge_coef)
imsave(output_dir+"harmonic_edges.png", img)
# exit()

init_image = img_as_float(imread('./output/harmonic_edges.png')[:,:,0:3])


############################################################################
# Patch nonlocal means

problem = inp.inpainting(image, mask, as_gray=False, kernels=[[[1]]], lambdas=[1.0], patch_shape=patch_shape, patch_weight=patch_weight)
result  = problem.process(num_scales=1, initialization=init_image, TOL=TOL, debug=False)
imsave(output_dir+"nlmeans.png", op.add_patch(result, patch_weight))
# exit()


# disjoint models
problem = inp.inpainting(image, mask, as_gray=False, kernels=[[[1]]])
problem.add_feature(terrain, terrain, lambdas=[1.0], beta=1.0, patch_shape=patch_shape, patch_weight=patch_weight)
problem.add_feature(house,   house,   lambdas=[1.0], beta=1.0, patch_shape=patch_shape, patch_weight=op.gauss_weight(patch_shape,patch_sigma=100))
result = problem.process(num_scales=1, initialization=init_image, TOL=TOL, debug=False)
imsave(output_dir+"feat_nlmeans.png", op.add_patch(result, patch_weight))
# exit()


############################################################################
# Patch nonlocal Poisson

for moll_sig in [0]:
	ker   = mollify_kernels(op.grad_kernels(), moll_sig)
	fname = output_dir+"sm"+str(moll_sig)+"_grad_x"
	imsave(fname.replace('.','')+".png", rescale_intensity(op.apply_kernel(rgb2grey(image),ker[0],"channels_last"), in_range=(-1,1)))

	for lmd in [0.2]:
		kernels = mollify_kernels(op.grad_kernels("forward"), moll_sig) + [[[1]]]

		problem = inp.inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=[edge_coef, edge_coef, lmd], patch_shape=patch_shape, patch_weight=patch_weight)
		result  = problem.process(num_scales=1, initialization=init_image, TOL=TOL, debug=False)
		fname   = output_dir+"/sm"+str(moll_sig)+"_nlpoisson_edges_lmd"+str(lmd)
		imsave(fname.replace('.','')+".png", op.add_patch(result, patch_weight))

		# disjoint models
		problem = inp.inpainting(image, mask, as_gray=False, kernels=kernels)
		problem.add_feature(terrain, terrain, lambdas=[0.0, 0.0, 1.0],     beta=1.0, patch_shape=patch_shape, patch_weight=patch_weight)
		problem.add_feature(house,   house,   lambdas=[edge_coef]*2+[lmd], beta=1.0, patch_shape=patch_shape, patch_weight=op.gauss_weight(patch_shape,patch_sigma=100))
		result = problem.process(num_scales=1, initialization=init_image, TOL=TOL, debug=False)
		fname  = output_dir+"/sm"+str(moll_sig)+"_feat_nlpoisson_edges_lmd"+str(lmd)
		imsave(fname.replace('.','')+".png", op.add_patch(result, patch_weight))