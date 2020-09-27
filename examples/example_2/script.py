from pathlib import Path

from skimage.exposure  import rescale_intensity
from skimage.color     import grey2rgb, rgb2grey
from skimage.io        import imread, imsave
from skimage.util      import montage, pad
from skimage.transform import resize, pyramid_reduce, pyramid_expand
from skimage           import img_as_uint, img_as_float

import matplotlib.pyplot as plt
import numpy as np

from   inpainting import Inpainting
import inpainting.utils as op



#############################################################################
# create directory for the output

output_dir = "output/"

Path(output_dir).mkdir(parents=True, exist_ok=True)


#############################################################################


# coo = [[69, 206], [120, 267], [41, 407], [209, 435], [286, 130], [333, 75], [372, 274], [453, 250], [439, 315]]
# coo = [[69, 206], [56, 407], [209, 435], [286, 130], [372, 274]]
coo = [[69, 206], [209, 435], [372, 274]]
def make_patch_collage(collage, patches=2):
	patch_collage = []
	p = 25
	for x, y in coo:
		for i in range(-patches,0,1):
			patch_collage.append( collage[i][y-p:y+p+1,x-p:x+p+1,:] )
	return montage( patch_collage, grid_shape=(len(coo),len(patch_collage)//len(coo)), padding_width=2, fill=[1.0]*3, multichannel=True )


from scipy.signal import correlate2d, convolve2d
def mollify_kernels(kernels, sigma):
	if sigma==0: return kernels
	gauss = op.gauss2d(sigma=(sigma,sigma), order=(0,0), angle=0, nstd=0.67, normalize=True)
	return [ correlate2d(np.array(ker), np.array(gauss), mode='full') for ker in kernels ]


#############################################################################

# damaged image and mask
example_setup = img_as_float(imread('./data/image.png')[:,:,0:3])
mask = np.zeros_like(example_setup[...,0]).astype(np.bool)
for x, y in coo:
	mask[y-15:y+16,x-15:x+16] = True
example_setup[mask,:] = 1.0
imsave(output_dir+"example_setup.png", example_setup )
# imsave("./data/mask.png", mask.astype(np.float) )
# exit()

# initialization
image = img_as_float(imread('./data/image.png')[:,:,0:3])
init_image = image.copy()
init_image[mask,:] = resize( pyramid_expand(pyramid_reduce(image,10,multichannel=True),10,multichannel=True), image.shape )[mask,:]
true_init = make_patch_collage([image, image, init_image], 3); w1 = true_init.shape[1]
true_init = make_patch_collage([image, init_image], 2);        w2 = true_init.shape[1]
true_init = pad(true_init, ((0,0),((w1-w2)//2,(w1-w2)//2+1),(0,0)), mode='edge')
imsave(output_dir+"true_init.png", true_init)
# exit()


# # 3rd party solutions
# im3rdparty = []
# # im3rdparty.append(img_as_float(imread('output/3rdparty/fedorov_means_p15.png')))
# im3rdparty.append(img_as_float(imread('output/3rdparty/fedorov_poisson_p15_lmd001.png')))
# im3rdparty.append(img_as_float(imread('output/3rdparty/crim_p9.png')))
# im3rdparty.append(img_as_float(imread('output/3rdparty/newson.png')))
# im3rdparty.append(img_as_float(imread('output/3rdparty/cntx_att_places2.png')))
# im3rdparty.append(img_as_float(imread('output/3rdparty/ec_places2.png')))
# im3rdparty.append(img_as_float(imread('output/3rdparty/gmcnn_places2.png')))
# imsave(output_dir+"3rdparty/3rdparty.png", make_patch_collage(im3rdparty, len(im3rdparty)) )
# exit()


#############################################################################
# common parameters

patch_shape = (15,15)
TOL = 1.e-4
initialization = init_image

sigmas      = [3,10,100]
moll_sigmas = [0,5]


collage = []


#############################################################################
# Patch nonlocal means

for sigma in sigmas:
	patch_weight = op.gauss_weight(patch_shape,patch_sigma=sigma).reshape(patch_shape)
	problem = Inpainting(image, mask, as_gray=False, kernels=[[[1]]], lambdas=[1], patch_shape=patch_shape, patch_weight=patch_weight)
	nlmeans = problem.process(num_scales=1, initialization=initialization, TOL=TOL)
	collage.append(op.add_patch(nlmeans,patch_weight))
	# imsave(output_dir+"means_sig"+str(sigma)+".png", op.add_patch(result,patch_weight))
imsave(output_dir+"nlmeans.png", make_patch_collage(collage, len(sigmas)) )
# exit()


#############################################################################
# Patch nonlocal Poisson
for moll_sig in moll_sigmas:

	ker   = mollify_kernels(op.grad_kernels(), moll_sig)
	fname = output_dir+"sm"+str(moll_sig)+"_grad_x"
	imsave(fname.replace('.','')+".png", rescale_intensity(op.apply_kernel(rgb2grey(image),ker[0],"channels_last"), in_range=(-1,1)))

	collage = [collage[i] for i in range(len(sigmas))]

	kernels = mollify_kernels(op.grad_kernels("forward"), moll_sig)
	lambdas = [1,1]
	for sigma in sigmas:
		patch_weight = op.gauss_weight(patch_shape,patch_sigma=sigma).reshape(patch_shape)
		problem = Inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight)
		result  = problem.process(num_scales=1, initialization=initialization, TOL=TOL, max_iters=30)
		collage.append(op.add_patch(result,patch_weight))
		# imsave(output_dir+"poisson_sig"+str(sigma)+".png", op.add_patch(result,patch_weight))
	fname = output_dir+"sm"+str(moll_sig)+"_nlpoisson"
	imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )

	for lmd in [0.01]:
		collage = [collage[i] for i in range(2*len(sigmas))]

		kernels = mollify_kernels(op.grad_kernels("forward"), moll_sig) + [[[1]]]
		lambdas = [1-lmd, 1-lmd, lmd]
		for sigma in sigmas:
			patch_weight = op.gauss_weight(patch_shape,patch_sigma=sigma).reshape(patch_shape)
			problem = Inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=lambdas, patch_shape=patch_shape, patch_weight=patch_weight)
			result  = problem.process(num_scales=1, initialization=initialization, TOL=TOL)
			collage.append(op.add_patch(result,patch_weight))
			# imsave(output_dir+"poisson_means_sig"+str(sigma)+".png", result)
		fname = output_dir+"sm"+str(moll_sig)+"_nlpoisson_lmd"+str(lmd)
		imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )

		for sigma in sigmas:
			patch_weight = op.gauss_weight(patch_shape,patch_sigma=sigma).reshape(patch_shape)
			problem = Inpainting(image, mask, as_gray=False, kernels=kernels)
			problem.add_feature(mask, None, lambdas=[1]*(len(kernels)-1)+[0.0], beta=1.-lmd, patch_shape=patch_shape, patch_weight=patch_weight)
			problem.add_feature(mask, None, lambdas=[0]*(len(kernels)-1)+[1.0], beta=lmd,    patch_shape=patch_shape, patch_weight=patch_weight)
			result  = problem.process(num_scales=1, initialization=initialization, TOL=TOL)
			collage.append(op.add_patch(result,patch_weight))
			# imsave(output_dir+"poisson_means_sig"+str(sigma)+".png", result)
		fname = output_dir+"sm"+str(moll_sig)+"_nlpoisson_feat_lmd"+str(lmd)
		imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )
		# exit()

		# Patch nonlocal Poisson and means average
		collage = collage + [ lmd*collage[len(sigmas)-i] + (1-lmd)*collage[2*len(sigmas)-1] for i in range(len(sigmas),0,-1) ]
		fname = output_dir+"sm"+str(moll_sig)+"_avg_poisson_means"
		imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )
		for i in range(len(sigmas)):
			fname = output_dir+"sm"+str(moll_sig)+"_avg_vs_feat_nlpoisson_sig"+str(sigmas[i])+"_lmd"+str(lmd)
			imsave(fname.replace('.','')+".png", make_patch_collage( [collage[i],collage[len(sigmas)+i]] + [collage[j] for j in [i-3*len(sigmas),i-2*len(sigmas),i-len(sigmas)]], 5) )
		# exit()
# exit()




#############################################################################
# Patch nonlocal gamma-Poisson
for moll_sig in moll_sigmas:
	collage = [collage[i] for i in range(len(sigmas))]

	for sigma in sigmas:
		kernels = mollify_kernels(op.nonlocal_grad_kernels(size=patch_shape[0], sigma=sigma), moll_sig)
		lambdas = [1]*len(kernels)
		problem = Inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=lambdas, patch_shape=(1,1))
		result  = problem.process(num_scales=1, initialization=initialization, TOL=TOL, max_iters=30)
		collage.append(result)
		# imsave(output_dir+"gamma_poisson_sig"+str(sigma)+".png", result)
	fname = output_dir+"sm"+str(moll_sig)+"_gamma_poisson"
	imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )
	# exit()


	for lmd in [0.1]:
		collage = [collage[i] for i in range(2*len(sigmas))]

		for sigma in sigmas:
			kernels = mollify_kernels(op.nonlocal_grad_kernels(size=patch_shape[0], sigma=sigma), moll_sig) + [[[1]]]
			lambdas = [1-lmd]*(len(kernels)-1) + [lmd]
			problem = Inpainting(image, mask, as_gray=False, kernels=kernels, lambdas=lambdas, patch_shape=(1,1))
			result  = problem.process(num_scales=1, initialization=initialization, TOL=TOL)
			collage.append(result)
			# imsave(output_dir+"gamma_poisson_means_sig"+str(sigma)+".png", result)
		fname = output_dir+"sm"+str(moll_sig)+"_gamma_poisson_lmd"+str(lmd)
		imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )

		for sigma in sigmas:
			kernels = mollify_kernels(op.nonlocal_grad_kernels(size=patch_shape[0], sigma=sigma), moll_sig) + [[[1]]]
			problem = Inpainting(image, mask, as_gray=False, kernels=kernels)
			problem.add_feature(mask, None, lambdas=[1]*(len(kernels)-1)+[0.0], beta=1.-lmd, patch_shape=(1,1))
			problem.add_feature(mask, None, lambdas=[0]*(len(kernels)-1)+[1.0], beta=lmd,    patch_shape=patch_shape, patch_weight=patch_weight)
			result  = problem.process(num_scales=1, initialization=initialization, TOL=TOL)
			collage.append(result)
			# imsave(output_dir+"gamma_poisson_means_sig"+str(sigma)+".png", result)
		fname = output_dir+"sm"+str(moll_sig)+"_gamma_poisson_feat_lmd"+str(lmd)
		imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )
		# exit()

		# Patch nonlocal gamma-Poisson and means average
		collage = collage + [ lmd*collage[len(sigmas)-i] + (1-lmd)*collage[2*len(sigmas)-1] for i in range(len(sigmas),0,-1) ]
		fname = output_dir+"sm"+str(moll_sig)+"_avg_gamma_poisson_means"
		imsave(fname.replace('.','')+".png", make_patch_collage(collage, len(sigmas)) )
		for i in range(len(sigmas)):
			fname = output_dir+"sm"+str(moll_sig)+"_avg_vs_feat_gamma_poisson_sig"+str(sigmas[i])+"_lmd"+str(lmd)
			imsave(fname.replace('.','')+".png", make_patch_collage( [collage[i],collage[len(sigmas)+i]] + [collage[j] for j in [i-3*len(sigmas),i-2*len(sigmas),i-len(sigmas)]], 5) )
		# exit()