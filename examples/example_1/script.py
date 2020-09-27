# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from pathlib import Path

from skimage.color      import grey2rgb
from skimage.transform  import resize
from skimage.morphology import binary_dilation
from skimage.util       import montage, pad
from skimage.io         import imread, imsave
from skimage            import img_as_uint, img_as_float

import scipy.ndimage as ndi

import matplotlib.pyplot as plt
import numpy as np

from   inpainting import Inpainting
import inpainting.utils as op


#############################################################################
# create directory for the output
Path("output/").mkdir(parents=True, exist_ok=True)



#############################################################################
# patch shape and weight

patch_shape  = (15,15)
patch_weight = op.gauss_weight(patch_shape,patch_sigma=10).reshape(patch_shape)
# imsave("./output/patch_weight.png", patch_weight.reshape(patch_shape)/np.amax(patch_weight))


#############################################################################
# exact nearest neighbors field

nnf_field  = np.ones((204, 204))
im_h, im_w = nnf_field.shape
for i in range(im_w):
	row = 24 + np.arange(im_h)%12
	col = i%12 + (24 if i<=im_w//2 else im_w-36)
	nnf_field[:,i] = col + row * im_w
nnf_field = nnf_field.ravel().astype(np.int)


#############################################################################
# make mask for the given patch shape

def make_mask(patch_shape, kernels=[[[1]]]):
	max_ker_size_0 = 0
	for ker in kernels:
		max_ker_size_0 = max( max_ker_size_0, np.array(ker).shape[0] ) if isinstance(ker, list) else max( max_ker_size_0, ker.shape[0] )
	offset_x = 12*5
	offset_y = 2*(max_ker_size_0//2 + patch_shape[0]//2)
	mask = np.ones(image.shape).astype('bool')
	mask[:offset_y,:] = mask[-offset_y:,:] = mask[:,:offset_x] = mask[:,-offset_x:] = False
	return mask


#############################################################################
# make image

cell = np.ones((12,12))
cell[3:9,3:9] = resize( imread("./data/cell.png", as_gray=True), (6,6), anti_aliasing=True, anti_aliasing_sigma=4 )
image = montage([cell]*(17*17), grid_shape=(17,17))

step_image = np.zeros_like(image)
step_image[:,:step_image.shape[1]//2] = 1

alpha = 0.7
image = (1-alpha)*image + alpha*step_image
imsave("./data/image.png", image)
imsave("./data/mask.png",  make_mask(patch_shape).astype(np.float))
# exit()


#############################################################################
# example setup

nonlocal_target_mask = binary_dilation( binary_dilation( make_mask(patch_shape=(21,21)), selem=np.ones((1,1)) ), selem=np.ones((21,21)) )
nonlocal_source_mask = nonlocal_target_mask.copy()
nonlocal_source_mask[10:-10,10:-10]  = np.logical_not(nonlocal_target_mask[10:-10,10:-10])
image_setup = grey2rgb(image)
# image_setup[make_mask((21,21)),0]   = 0.95
# image_setup[make_mask((21,21)),1]   = 0.80
# image_setup[make_mask((21,21)),2]   = 0.0
image_setup[make_mask((21,21)),0]   = 0.75
image_setup[make_mask((21,21)),1]   = 0.0
image_setup[make_mask((21,21)),2]   = 0.0
image_setup[nonlocal_source_mask,0] = 0.29
image_setup[nonlocal_source_mask,1] = 0.65
image_setup[nonlocal_source_mask,2] = 0.12
alpha = 0.6
image_setup[make_mask((21,21)),:]   = ((1-alpha)*grey2rgb(image) + alpha*image_setup)[make_mask((21,21)),:]
image_setup[nonlocal_source_mask,:] = ((1-alpha)*grey2rgb(image) + alpha*image_setup)[nonlocal_source_mask,:]
# image_setup[make_mask((21,21)),:] = 1
imsave("./output/example_setup.png", op.add_patch(image_setup, op.gauss_weight((21,21),patch_sigma=10).reshape((21,21))))
# exit()


#############################################################################
# Local PDE inpainting

img = image.copy()
Inpainting.inpaint_PDE(None, img, make_mask(patch_shape), 'harmonic')
imsave("./output/harmonic_PDE.png", img)

img = image.copy()
Inpainting.inpaint_PDE(None, img, make_mask(patch_shape), 'biharmonic')
imsave("./output/biharmonic_PDE.png", img)
# exit()



#############################################################################
# Patch nonlocal means

kernels = [[[1]]]
lambdas = [1]


problem = Inpainting(step_image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='random' )
imsave("./output/step_means.png", op.add_patch(result, patch_weight))

problem = Inpainting(image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='biharmonic' )
imsave("./output/means.png", op.add_patch(result, patch_weight))
# exit()


#############################################################################
# Patch nonlocal Poisson

lmb = 0.0

kernels = op.grad_kernels("forward") + [[[1]]]
lambdas = [1.0,1.0,0.0]


problem = Inpainting(step_image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='random' )
imsave("./output/step_poiss.png", op.add_patch(result, patch_weight))

problem = Inpainting(image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='biharmonic' )
imsave("./output/poiss.png", op.add_patch(result, patch_weight))
# exit()


#############################################################################
# Biharmonic exemplar algorithm

kernels = op.laplacian_kernel() + [[[1]]]
lambdas = [1.0,0.0]

problem = Inpainting(step_image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, patch_shape=patch_shape, patch_weight=patch_weight, lambdas=lambdas, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='biharmonic' )
imsave("./output/step_biharm.png", op.add_patch(result, patch_weight))

problem = Inpainting(image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, patch_shape=patch_shape, patch_weight=patch_weight, lambdas=lambdas, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='biharmonic' )
imsave("./output/biharm.png", op.add_patch(result, patch_weight))
# exit()



#############################################################################
# Patch nonlocal Poisson with anisotropic lambda

def make_coef(edges):
	asympt_val = 0.1
	decay_time = 10

	coef  = (1-asympt_val) * np.exp(-ndi.distance_transform_edt(1-edges)/decay_time) + asympt_val
	coef  = 1 - coef
	coef /= np.amax(coef)
	coef[coef<1.e-6] = 1.e-4
	return coef

im_h, im_w = image.shape

#########################
# straigt line interface
edges = np.zeros_like(image)
edges[:,im_w//2-1:im_w//2+1] = 1
coef = make_coef(edges)
imsave("./output/edges_coef.png",img_as_float(coef))

color_edges = image.copy()
mask = make_mask((15,15))
color_edges[mask] = 1.0
color_edges = grey2rgb(color_edges)
color_edges[edges==1,:] = 0.0
color_edges[edges==1,2] = 1.0
color_edges[np.logical_and(edges==1,mask),2] = 0.0
color_edges[np.logical_and(edges==1,mask),0] = 1.0
imsave("./output/color_edges.png", color_edges)

img = image.copy()
Inpainting.inpaint_PDE(None, img, make_mask(patch_shape), 'harmonic', aniso_coef=coef)
imsave("./output/step_harmonic_edges.png", img)


kernels = op.grad_kernels("forward")
lambdas = [coef,coef]
problem = Inpainting(image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='harmonic', init_coef=coef )
imsave("./output/step_nonloc_harmonic_edges.png", op.add_patch(result, patch_weight))
# exit()

#########################
# curved line interface
edges = imread("./data/edges2.png",as_gray=True).astype(np.bool)
coef  = make_coef(edges)
imsave("./output/edges_coef_2.png",img_as_float(coef))

color_edges = image.copy()
mask = make_mask((15,15))
color_edges[mask] = 1.0
color_edges = grey2rgb(color_edges)
color_edges[edges==1,:] = 0.0
color_edges[edges==1,2] = 1.0
color_edges[np.logical_and(edges==1,mask),2] = 0.0
color_edges[np.logical_and(edges==1,mask),0] = 1.0
imsave("./output/color_edges_2.png", color_edges)

img = image.copy()
Inpainting.inpaint_PDE(None, img, make_mask(patch_shape), 'harmonic', aniso_coef=coef)
imsave("./output/step_harmonic_edges_2.png", img)


kernels = op.grad_kernels("forward")
lambdas = [coef,coef]
problem = Inpainting(image, make_mask(patch_shape,kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight, nnf_field=nnf_field)
result  = problem.process(num_scales=1, initialization='harmonic', init_coef=coef )
imsave("./output/step_nonloc_harmonic_edges_2.png", op.add_patch(result, patch_weight))
# exit()

#############################################################################
# Patch nonlocal gamma-Poisson

for ker_size in [31]:
	for s in [-1,-0.5,0,0.5]:
		kernels = op.nonlocal_grad_kernels(size=ker_size, s=s)
		lambdas = [1.0]*len(kernels)

		problem = Inpainting(image, make_mask((1,1),kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=(1,1), nnf_field=None)
		result  = problem.process(num_scales=1, initialization='biharmonic' )
		imsave("./output/k"+str(ker_size)+"_s"+str(s).replace('.','')+".png", op.add_patch(result, patch_weight))

	for sigma in [100,10,5,3]:
		kernels = op.nonlocal_grad_kernels(size=ker_size, sigma=sigma)
		lambdas = [1.0]*len(kernels)

		problem = Inpainting(image, make_mask((1,1),kernels), as_gray=True, kernels=kernels, lambdas=lambdas, patch_shape=(1,1), nnf_field=None)
		result  = problem.process(num_scales=1, initialization='biharmonic' )
		imsave("./output/k"+str(ker_size)+"_sig"+str(sigma)+".png", op.add_patch(result, patch_weight))
# exit()