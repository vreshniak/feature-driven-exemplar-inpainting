import os
import inspect
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from scipy.signal import convolve2d, correlate2d, fftconvolve, oaconvolve
from skimage import img_as_ubyte, img_as_float
from skimage.transform import resize, pyramid_reduce, pyramid_expand, pyramid_gaussian
from skimage.morphology import binary_dilation, binary_erosion
import skimage.morphology as morph
from skimage.io import imread, imsave
from skimage.color import rgb2grey, rgba2rgb, grey2rgb


import inpainting.utils    as op
import inpainting.features as ft


_im_dtype = np.float64


# TODO: global params dict


def read_image(img, as_gray):
	if as_gray:
		if isinstance(img, str):
			image = imread(img, as_gray=True)
		elif img.ndim==2:
			image = img.copy()
		elif img.ndim<=4:
			image = rgb2grey(img)
		else:
			assert True, "wrong number of image dimensions for image with shape "+str(image.shape)
		image = image[:,:,np.newaxis]
	else:
		if isinstance(img, str):
			image = imread(img, as_gray=False)
			if image.ndim==4:
				image = rgba2rgb(image)
		elif img.ndim==2:
			image = grey2rgb(img)
		elif img.ndim==3:
			image = img.copy()
		elif img.ndim==4:
			image = rgba2rgb(image)
		else:
			assert True, "wrong number of image dimensions for image with shape "+str(image.shape)
	return img_as_float(image).astype(_im_dtype)



class Inpainting:

	# def __init__(self, image_file_name, mask_file_name, source_mask_file_name=None, as_gray=False, kernels=None, lambdas=None, patch_shape=(3,3), patch_weight=None, patch_sigma=None):
	def __init__(self, image, mask, source_mask=None, as_gray=False, kernels=None, lambdas=None, patch_shape=(3,3), patch_weight=None, patch_sigma=None, nn_algorithm='PatchMatch', nnf_field=None):
		self.caller_path = os.path.dirname(os.path.abspath(inspect.getmodule(inspect.stack()[1][0]).__file__))

		# note that image has three dimensions and we use CHANNELS FIRST
		# if as_gray:
		# 	if isinstance(image, str):
		# 		self.image = imread(image,as_gray=True).astype(_im_dtype)[np.newaxis,:,:]
		# 	elif image.ndim==2:
		# 		self.image = image.copy()[np.newaxis,:,:]
		# 	elif image.ndim<=4:
		# 		self.image = rgb2grey(image)[np.newaxis,:,:]
		# 	else:
		# 		assert True, "wrong number of image dimensions"
		# 	self.image_shape = self.image.shape
		# else:
		# 	# self.image = np.moveaxis( imread(image_file_name,as_gray=False).astype(_im_dtype)[:,:,0:3], -1, 0 ) / 255.0
		# 	# self.image = imread(image,as_gray=False).astype(_im_dtype)[:,:,0:3] / 255.0
		# 	if isinstance(image, str):
		# 		self.image = imread(image,as_gray=False)
		# 		if self.image.ndim==4:
		# 			self.image = rgba2rgb(self.image)
		# 		self.image = img_as_float(self.image).astype(_im_dtype)
		# 	elif image.ndim==2:
		# 		self.image = grey2rgb(image)
		# 	elif image.ndim<=4:
		# 		self.image = rgb2grey(image)[np.newaxis,:,:]
		# 	else:
		# 		assert True, "wrong number of image dimensions"
		# 	self.image_shape = ( self.image.shape[2], self.image.shape[0], self.image.shape[1] )
		# from skimage.util import random_noise
		# self.image = random_noise(self.image,var=0.002)

		# if isinstance(mask, str):
		# 	self.mask = imread(mask, as_gray=True).astype(np.bool)
		# else:
		# 	self.mask = mask.astype(np.bool)


		self.image = read_image(image, as_gray)
		self.mask  = read_image(mask, as_gray=True).squeeze().astype(np.bool)

		self.image_shape = ( self.image.shape[2], self.image.shape[0], self.image.shape[1] )

		self.num_filters = 1 if kernels is None else len(kernels)
		self.filters     = [np.array([[1]],dtype=_im_dtype)] if kernels is None else [ np.array(kernel,dtype=_im_dtype) for kernel in kernels ]

		self.num_features = 0
		self.features     = []
		if lambdas is not None:
			# lambdas = [1]*self.num_filters if lambdas is None else lambdas
			self.add_feature(mask, source_mask, patch_shape, patch_weight, lambdas=lambdas, nn_algorithm=nn_algorithm, nnf_field=nnf_field)

		self.curr_image = None


	def build_image_pyramids(self, num_scales=10, coarsest_scale=0.1, mask_threshold=0.1):
		"""
		Build image and mask pyramids

		"""
		# from skimage.transform   import pyramid_gaussian

		# def pyramid(im,multichannel=False):
		# 	return pyramid_gaussian(im, max_layer=num_scales-1, downscale=downscale, multichannel=multichannel)

		# if num_scales>1:
		# 	downscale = coarsest_scale**(-1.0/(num_scales-1))
		# 	self.image_pyramid     = [ np.moveaxis(im,-1,0) for im in pyramid(np.moveaxis(self.image,0,-1),True) ] if self.image.ndim==3 else [ im for im in pyramid(self.image,False) ]
		# 	self.mask_pyramid      = [ im > mask_threshold  for im in pyramid(self.mask) ]
		# 	self.conf_mask_pyramid = list( pyramid(conf_mask) )
		# else:
		# 	self.image_pyramid     = [ self.image ]
		# 	self.mask_pyramid      = [ self.mask > mask_threshold ]
		# 	self.conf_mask_pyramid = [ conf_mask ]

		# build mask pyramids for each feature
		for feat in self.features:
			feat.build_mask_pyramids(num_scales, coarsest_scale, mask_threshold)


	def get_image_at_scale(self, image, scale, num_scales=10, coarsest_scale=0.1):
		im = image.copy()
		if num_scales>1:
			downscale = coarsest_scale**(-1.0/(num_scales-1))
			for s in range(scale):
				im = pyramid_reduce(im, downscale, multichannel=True) if im.ndim==3 else pyramid_reduce(im, downscale, multichannel=False)
		if im.ndim==3:
			return np.moveaxis(im,-1,0)
		else:
			return im
	def get_mask_at_scale(self, image, scale, num_scales=10, coarsest_scale=0.1, mask_threshold=0.1):
		im = image.copy()
		if num_scales>1:
			downscale = coarsest_scale**(-1.0/(num_scales-1))
			for s in range(scale):
				im = pyramid_reduce(im, downscale, multichannel=False) > mask_threshold
		return im


	def preprocess(self, scale, num_scales, coarsest_scale, initialization='biharmonic', constant_values=1.0, init_coef=None, mask_threshold=0.1):
		start = time.time()

		# image and mask at current scale
		image = self.get_image_at_scale(self.image, scale, num_scales, coarsest_scale)
		mask  = self.get_mask_at_scale(self.mask, scale, num_scales, coarsest_scale, mask_threshold)

		# dimensions of the image at current scale
		im_ch, im_h, im_w = image.shape

		if scale==num_scales-1:
			# initialize the coarsest scale
			if initialization is not None:
				if isinstance(initialization, np.ndarray):
					image[:,mask] = self.get_image_at_scale(initialization, scale, num_scales, coarsest_scale)[:,mask]
					# image[:,mask] = np.moveaxis(initialization,-1,0)[:,mask]
				elif isinstance(initialization, str):
					if initialization=='harmonic':
						aniso_coef = np.ones_like(mask) if init_coef is None else self.get_image_at_scale(init_coef, scale, num_scales, coarsest_scale)
						self.inpaint_PDE(image, mask, type='harmonic', aniso_coef=aniso_coef)
					elif initialization=='biharmonic':
						self.inpaint_PDE(image, mask, type='biharmonic')
					# elif initialization=='image':
					# 	assert init_image is not None, "init_image must be given"
					# 	image[:,mask] = self.get_image_at_scale(init_image, scale, num_scales, coarsest_scale)[:,mask]
					elif initialization=='constant':
						image[:,mask] = constant_values
					elif initialization=='mean':
						image[:,mask] = np.mean(self.image)
					elif initialization=='random':
						image[:,mask] = np.random.rand(im_ch,im_h,im_w)[:,mask]

			#######################################################################

			# find max kernel size
			self.max_ker_size_0 = self.max_ker_size_1 = 0
			for ker in self.filters:
				self.max_ker_size_0 = max( self.max_ker_size_0, ker.shape[0] )
				self.max_ker_size_1 = max( self.max_ker_size_1, ker.shape[1] )

			# find max patch size
			self.max_patch_size_0 = self.max_patch_size_1 = 0
			for feat in self.features:
				self.max_patch_size_0 = max( self.max_patch_size_0, feat.patch_shape[0] )
				self.max_patch_size_1 = max( self.max_patch_size_1, feat.patch_shape[1] )

			#######################################################################
		else:
			# upscale lowres image from the previous scale
			for ch in range(im_ch):
				image[ch,mask] = resize( self.curr_image[ch,...], mask.shape, mode='reflect', order=1, anti_aliasing=False)[mask]

		# truncate values to the allowed limits
		image[image<0] = 0.0
		image[image>1] = 1.0

		#######################################################################

		# bounding box of the inpainting region at current scale
		inp_top_left_y, inp_top_left_x, inp_bot_rght_y, inp_bot_rght_x = op.masked_bounding_box(mask)

		# boundary extensions of the image and masks to account for nonlocal kernels and patches
		ext_x_lft = abs(min( 0, inp_top_left_x - 2*(self.max_ker_size_0//2 + self.max_patch_size_0//2) ))
		ext_y_top = abs(min( 0, inp_top_left_y - 2*(self.max_ker_size_1//2 + self.max_patch_size_1//2) ))
		ext_x_rgt = abs(min( 0, (im_w-1) - inp_bot_rght_x - 2*(self.max_ker_size_0//2 + self.max_patch_size_0//2) ))
		ext_y_bot = abs(min( 0, (im_h-1) - inp_bot_rght_y - 2*(self.max_ker_size_1//2 + self.max_patch_size_1//2) ))
		print("Image extensions (l,r,t,b)   ... ", ((ext_x_lft,ext_x_rgt),(ext_y_top,ext_y_bot)))


		# pad image to account for nonlocal kernels and patches. Note that 'image' is redefined after this point
		image = np.pad( image, ((0,0),(ext_y_top,ext_y_bot),(ext_x_lft,ext_x_rgt)), 'constant', constant_values=0     )
		mask  = np.pad( mask,  ((ext_y_top,ext_y_bot),(ext_x_lft,ext_x_rgt)),       'constant', constant_values=False )

		# dimensions of the padded image at current scale
		im_ch, im_h, im_w = self.im_shape = image.shape

		self.curr_image = image[:,ext_y_top:im_h-ext_y_bot,ext_x_lft:im_w-ext_x_rgt]


		#######################################################################

		# linear indices of the masked pixels
		self.ind_dof = op.masked_indices(mask)

		# $O_*$ domain - extend mask to account for nonlocal kernels
		conv_target_mask = binary_dilation( mask, selem=morph.rectangle(self.max_ker_size_1,self.max_ker_size_0) )

		# # $\tilde{O}_*$ domain - extend mask to account for patches
		# nonlocal_target_mask = binary_dilation( conv_target_mask, selem=morph.rectangle(self.max_patch_size_1,self.max_patch_size_0) )

		# # $\tilde{O}_*^c$ domain
		# nonlocal_source_mask = nonlocal_target_mask.copy()
		# p_h, p_w = (self.max_patch_size_0//2+self.max_ker_size_0//2, self.max_patch_size_1//2+self.max_ker_size_1//2)
		# nonlocal_source_mask[p_h:-p_h,p_w:-p_w] = np.logical_not( nonlocal_target_mask[p_h:-p_h,p_w:-p_w] )
		# # m_h, m_w = nonlocal_source_mask.shape
		# # nonlocal_source_mask[p_h:m_h-p_h,p_w:m_w-p_w] = np.logical_not( nonlocal_target_mask[p_h:m_h-p_h,p_w:m_w-p_w] )

		# imsave("./dbg/target_mask"+str(num_scales-scale-1)+".png",img_as_ubyte(nonlocal_target_mask),cmap='gray')
		# imsave("./dbg/source_mask"+str(num_scales-scale-1)+".png",img_as_ubyte(nonlocal_source_mask),cmap='gray')

		#######################################################################

		# pad confidence mask to account for nonlocal kernels and patches
		# self.confidence = self.get_image_at_scale(self.calculate_confidence_mask(mask, 0.1, 10.0), scale, num_scales, coarsest_scale).ravel()
		self.confidence = np.pad( self.calculate_confidence_mask(self.get_mask_at_scale(self.mask, scale, num_scales, coarsest_scale, mask_threshold)), ((ext_y_top,ext_y_bot),(ext_x_lft,ext_x_rgt)), 'constant', constant_values=1 ).ravel()
		# self.confidence = np.pad( self.get_image_at_scale(self.conf_mask, scale, num_scales, coarsest_scale), ((ext_y_top,ext_y_bot),(ext_x_lft,ext_x_rgt)), 'constant', constant_values=1 ).ravel()

		#######################################################################

		# relative linear indices of pixels in zero-centered patches
		for feat in self.features:
			feat.patch_h = np.zeros((feat.patch_shape[0]*feat.patch_shape[1],),dtype=np.int32)
			for i in range(feat.patch_shape[0]):
				for j in range(feat.patch_shape[1]):
					feat.patch_h[i*feat.patch_shape[1]+j] = (j-feat.patch_shape[1]//2) + (i-feat.patch_shape[0]//2)*im_w

		# convolution matrices
		conv_mat_start = time.time()
		self.conv_mat     = [ op.conv2mat((im_h,im_w),k,dtype=_im_dtype).tocsr() for k in self.filters ]
		self.adj_conv_mat = [ mat.T[self.ind_dof,:] for mat in self.conv_mat ]
		conv_mat_time = time.time() - conv_mat_start

		# convolutions in the known part of the domain
		# self.convolutions = np.array([ [correlate2d(image[ch],k,mode='same',boundary='symm').ravel() for k in self.filters] for ch in range(im_ch) ])
		# self.convolutions = np.array([ fftconvolve( np.tile(image[ch][np.newaxis,...],(len(self.filters),1,1)), np.flip(np.array(self.filters),axis=(1,2)), mode='same', axes=(1,2)).reshape(len(self.filters),-1) for ch in range(im_ch) ])

		filt_start = time.time()
		self.convolutions = np.array([ fftconvolve( np.tile(image[ch][np.newaxis,...],(len(self.filters),1,1)), np.flip(op.stack_kernels(self.filters),axis=(1,2)), mode='same', axes=(1,2)).reshape(len(self.filters),-1) for ch in range(im_ch) ])
		# self.convolutions = np.array([ oaconvolve( np.tile(image[ch][np.newaxis,...],(len(self.filters),1,1)), np.flip(op.stack_kernels(self.filters),axis=(1,2)), mode='same', axes=(1,2)).reshape(len(self.filters),-1) for ch in range(im_ch) ])
		filt_time = time.time() - filt_start

		# preprocess features at the current scale
		for j,feat in enumerate(self.features):
			feat_max_ker_size_0 = feat_max_ker_size_1 = 0
			for active_ker in feat.active_filters:
				feat_max_ker_size_0 = max( feat_max_ker_size_0, self.filters[active_ker].shape[0] )
				feat_max_ker_size_1 = max( feat_max_ker_size_1, self.filters[active_ker].shape[1] )
			feat_conv_extension_kernel  = np.ones((feat_max_ker_size_0,feat_max_ker_size_1))
			feat_patch_extension_kernel = np.ones(feat.patch_shape)

			#######################################################################
			# nonlocal target and source masks with conv kernels and patch sizes of the given feature

			# $\tilde{O}_*$ domain
			global_feat_target_mask = binary_dilation( binary_dilation( mask, selem=feat_conv_extension_kernel ), selem=feat_patch_extension_kernel )

			# $\tilde{O}_*^c$ domain
			global_feat_source_mask = global_feat_target_mask.copy()
			p_h, p_w = (feat.patch_shape[0]//2+self.max_ker_size_0//2, feat.patch_shape[1]//2+self.max_ker_size_1//2)
			global_feat_source_mask[p_h:-p_h,p_w:-p_w] = np.logical_not( global_feat_target_mask[p_h:-p_h,p_w:-p_w] )

			#######################################################################

			feat_target_mask  = np.pad( feat.target_mask_pyramid[scale], ((ext_y_top,ext_y_bot),(ext_x_lft,ext_x_rgt)), 'constant', constant_values=False )
			feat_target_mask  = binary_dilation( feat_target_mask, selem=feat_conv_extension_kernel  )
			feat.conv_inp_ind = op.masked_indices(np.logical_and(feat_target_mask, conv_target_mask))
			feat_target_mask  = binary_dilation( feat_target_mask, selem=feat_patch_extension_kernel )
			feat_target_mask  = np.logical_and( feat_target_mask, global_feat_target_mask )

			feat_source_mask = np.pad( feat.source_mask_pyramid[scale], ((ext_y_top,ext_y_bot),(ext_x_lft,ext_x_rgt)), 'constant', constant_values=False )
			feat_source_mask = binary_dilation( binary_dilation( feat_source_mask, selem=feat_conv_extension_kernel  ), selem=feat_patch_extension_kernel )
			feat_source_mask = np.logical_and( feat_source_mask, global_feat_source_mask )

			feat.source_ind = op.masked_indices(feat_source_mask)
			feat.target_ind = op.masked_indices(feat_target_mask)
			# feat.source_ind2 = feat.source_ind.copy()

			#######################################################################

			feat.lambdas = np.pad( feat.lambda_pyramid[scale], ((0,0),(ext_y_top,ext_y_bot),(ext_x_lft,ext_x_rgt)), 'constant', constant_values=0.0 ).reshape(feat.lambda_pyramid[scale].shape[0],-1)

			#######################################################################

			feat.build_nnf_index( scale, self.convolutions.reshape(-1,im_h,im_w), feat_target_mask, feat_source_mask )

			#######################################################################

			beta_sum = feat.beta if j==0 else beta_sum+feat.beta

		# normalize betas
		beta_sum[beta_sum<1.e-7] = 1e-7
		for feat in self.features:
			feat.beta /= beta_sum
			# feat.beta[beta_sum<1.e-9] = 1.0/self.num_features

		print("Preprocessing                ... %6.3f sec"%(time.time()-start))
		print("   incl. eval. filters       ... %6.3f sec"%(filt_time))
		print("   incl. eval. matricies     ... %6.3f sec"%(conv_mat_time))
		return image



	def calculate_confidence_mask(self, mask, asympt_val=0.1, decay_time=10.0):
		import scipy.ndimage as ndi
		return (1-asympt_val) * np.exp(-ndi.distance_transform_edt(mask)/decay_time) + asympt_val
		# return np.ones_like(self.mask,dtype=_im_dtype)



	def calculate_operator_matrix(self, scale, image):
		start = time.time()
		im_ch, im_h, im_w = self.im_shape

		# sparse matrix of the operator for the whole image
		Asp = sp.csr_matrix( (self.ind_dof.size,im_h*im_w), dtype=_im_dtype )

		# coefficient
		# self.lambda_k = np.ones((im_h*im_w,),dtype=_im_dtype)
		self.lambda_k = np.zeros((im_h*im_w,),dtype=_im_dtype)

		# contributions from the convolutions
		for i in range(self.num_filters):
			# self.lambda_k[self.conv_inp_ind] = 0.0
			self.lambda_k.fill(0.0)
			for feat in self.features:
				patch_ind = feat.conv_inp_ind[:,np.newaxis] - feat.patch_h[np.newaxis,:]
				self.lambda_k[feat.conv_inp_ind] += feat.lambdas[i][feat.conv_inp_ind] * np.sum( self.confidence[patch_ind] * feat.beta[patch_ind] * feat.patch_weight[np.newaxis,:], axis=-1)
				# self.lambda_k[feat.conv_inp_ind] += self.aniso_coef[feat.conv_inp_ind] * np.sum( self.confidence[patch_ind] * feat.beta[patch_ind] * feat.patch_weight[np.newaxis,:], axis=-1)
				# self.lambda_k[feat.conv_inp_ind] += feat.lambdas[i][feat.conv_inp_ind] * np.sum( self.confidence[patch_ind] * feat.beta[patch_ind] * feat.patch_weight[np.newaxis,:], axis=-1)
			Asp += self.adj_conv_mat[i].dot( self.conv_mat[i].tocsr().multiply(self.lambda_k[:,np.newaxis]) ) # note csr matrix format when using .multiply

		# matrix of the linear system
		self.A = Asp.tocsc()[:,self.ind_dof]

		print("Constructing operator matrix ... %6.3f sec"%(time.time()-start))


		# preconditioner
		start = time.time()
		ILU = sp.linalg.splu(self.A)
		self.precond = sp.linalg.LinearOperator( self.A.shape, lambda v: ILU.solve(v), dtype=_im_dtype )

		print("Constructing preconditioner  ... %6.3f sec"%(time.time()-start))


		# contribution to the rhs from the Dirichlet boundary conditions
		self.F_bc = [ self.A.dot(image[ch][self.ind_dof]) - Asp.dot(image[ch]) for ch in range(im_ch) ]
		self.F = self.F_bc[0].copy()

		self.conv_mat     = [ mat[self.ind_dof,:] for mat in self.conv_mat ]
		# self.adj_conv_mat = [ mat.tocsc()[:,self.conv_inp_ind] for mat in self.adj_conv_mat ]
		self.adj_conv_mat = [ mat.tocsc() for mat in self.adj_conv_mat ]



	def update_weights(self,image):
		start = time.time()

		im_ch, im_h, im_w = self.im_shape

		# update filters for the given updated image
		for ch in range(im_ch):
			for i in range(self.num_filters):
				self.convolutions[ch][i].ravel()[self.ind_dof] = self.conv_mat[i].dot(image[ch].ravel())

		for feat in self.features:
			# feat.initialization works only for single scale at the moment
			if feat.initialization is not None:
				img = image.copy()
				img.reshape(im_ch,-1)[:,feat.target_ind] = feat.initialization.reshape(im_ch,-1)[:,feat.target_ind]
				convolutions = np.array([ fftconvolve( np.tile(img[ch][np.newaxis,...],(len(self.filters),1,1)), np.flip(op.stack_kernels(self.filters),axis=(1,2)), mode='same', axes=(1,2)).reshape(len(self.filters),-1) for ch in range(im_ch) ])
				feat.calculate_nnf(convolutions.reshape(-1,im_h,im_w))
				feat.initialization = None
			else:
				feat.calculate_nnf(self.convolutions.reshape(-1,im_h,im_w))

		print("\t\tNNF ... %6.3f sec"%(time.time()-start))



	def update_image(self,image):
		im_ch, im_h, im_w = self.im_shape

		time_rhs = time_sol = 0.0
		for ch in range(im_ch):
			# calculate RHS
			start = time.time()
			np.copyto(self.F, self.F_bc[ch], casting='same_kind', where=True)
			for feat in self.features:
				patch_ind = feat.conv_inp_ind[:,np.newaxis] - feat.patch_h[np.newaxis,:]
				nnf_ind   = feat.nnf[patch_ind] + feat.patch_h[np.newaxis,:]
				conf_beta_patch = self.confidence[patch_ind] * feat.beta[patch_ind] * feat.patch_weight[np.newaxis,:]
				# for i in range(self.num_filters):
				for i in feat.active_filters:
					lambda_f = feat.lambdas[i][feat.conv_inp_ind] * np.sum( conf_beta_patch * self.convolutions[ch][i][nnf_ind], axis=-1 )
					# lambda_f = self.aniso_coef[feat.conv_inp_ind] * np.sum( conf_beta_patch * self.convolutions[ch][i][nnf_ind], axis=-1 )
					# lambda_f = feat.lambdas[i][feat.conv_inp_ind] * np.sum( conf_beta_patch * self.convolutions[ch][i][nnf_ind], axis=-1 )
					self.F += self.adj_conv_mat[i][:,feat.conv_inp_ind].dot(lambda_f.ravel())

			# for feat in self.features:
			# 	patch_ind = self.conv_inp_ind[:,np.newaxis] - feat.patch_h[np.newaxis,:]
			# 	ind = feat.nnf[patch_ind] + feat.patch_h[np.newaxis,:]
			# 	conf_beta_patch = self.confidence[patch_ind] * feat.beta[patch_ind] * feat.patch_weight[np.newaxis,:]
			# 	for i in range(self.num_filters):
			# 		lambda_f = feat.lambdas[i][self.conv_inp_ind] * np.sum( conf_beta_patch * self.convolutions[ch][i][ind], axis=-1 )
			# 		self.F += self.adj_conv_mat[i].dot(lambda_f.ravel())

			time_rhs += time.time()-start

			# solve linear system for the correction
			start = time.time()
			if self.num_filters==1 and self.filters[0][0,0]==1:
				delta = self.F - self.A.dot(image[ch][self.ind_dof])
				delta /= self.lambda_k.ravel()[self.ind_dof]
			else:
				# delta = sp.linalg.spsolve( self.A, F - self.A.dot(image[self.ind_dof]) )
				# delta = self.precond(F - self.A.dot(image[ch][self.ind_dof]))
				delta, info = sp.linalg.cg( self.A, self.F - self.A.dot(image[ch][self.ind_dof]), x0=np.zeros(self.F.shape,dtype=_im_dtype), tol=1e-6, maxiter=1000, M=self.precond )
				if info!=0:
					# exit("CG not converged, code %d %f"%(info,np.sqrt(np.sum(delta*delta))))
					exit("CG not converged, code %d %f"%(info,np.amax(np.abs(delta))))

			time_sol += time.time()-start

			# update image
			image[ch][self.ind_dof] += delta

		# truncate values to the allowed limits
		image[image<0] = 0
		image[image>1] = 1

		print("\t\tRHS ... %6.3f sec"%(time_rhs))
		print("\t\tSOL ... %6.3f sec"%(time_sol))

		# return np.sqrt(np.sum(delta*delta))
		# return np.amax(np.abs(delta))
		return np.mean(np.abs(delta))


	def add_feature(self, target_mask, source_mask=None, patch_shape=(3,3), patch_weight=None, patch_sigma=None, initialization=None, lambdas=None, beta=None, nn_algorithm='PatchMatch', nnf_field=None):
		lambdas = [1]*self.num_filters if lambdas is None else lambdas
		assert len(lambdas)==self.num_filters, "len(lambdas) for each feature must be equal to self.num_filters"
		patch_weight = op.gauss_weight(patch_shape,patch_sigma=patch_sigma) if patch_sigma is not None else patch_weight
		self.features.append( ft.feature(self.image_shape, target_mask, source_mask, patch_shape=patch_shape, patch_weight=patch_weight, lambdas=lambdas, beta=beta, nn_algorithm=nn_algorithm, nnf_field=nnf_field) )
		self.features[-1].initialization = initialization if initialization is None else np.moveaxis(initialization,-1,0)
		self.num_features += 1


	def process(self, num_scales=10, coarsest_scale=0.1, initialization="biharmonic", init_coef=None, debug=False, TOL=1.e-5, max_iters=100):
		if debug:
			Path(self.caller_path,'debug').mkdir(parents=True, exist_ok=True)
			Path(self.caller_path,'debug/iters').mkdir(parents=True, exist_ok=True)

		num_scales     = 1   if coarsest_scale==1 else num_scales
		coarsest_scale = 1.0 if num_scales==1     else coarsest_scale

		mask_threshold = 0.4
		self.conf_mask = self.calculate_confidence_mask(self.mask)
		self.build_image_pyramids(num_scales, coarsest_scale, mask_threshold=mask_threshold)

		# try:
		# multiscale inpainting
		for scale in range(num_scales-1,-1,-1):
			print("Inpainting at scale %d"%(num_scales-scale-1))

			image = self.preprocess(scale, num_scales, coarsest_scale, initialization, init_coef=init_coef, mask_threshold=mask_threshold)

			self.calculate_operator_matrix( scale, image.reshape(image.shape[0],-1) )
			if scale==num_scales-1 and isinstance(initialization, str) and initialization=='homogeneous':
				temp_conv = self.convolutions.copy()
				self.convolutions.fill(0.0)
				_ = self.update_image(image.reshape(image.shape[0],-1))
				self.convolutions = temp_conv.copy()
				del temp_conv

			# save initialization
			if debug:
				imsave( "./debug/dbg_scale_"+str(num_scales-scale-1)+"_mask.png",       img_as_ubyte(self.get_mask_at_scale(self.mask, scale, num_scales, coarsest_scale, mask_threshold)) )
				imsave( "./debug/dbg_scale_"+str(num_scales-scale-1)+"_conf_mask.png",  img_as_ubyte(self.calculate_confidence_mask(self.get_mask_at_scale(self.mask, scale, num_scales, coarsest_scale, mask_threshold))) )
				# convert to the channels last format
				im_debug = np.moveaxis(image,0,-1).copy()
				# add patch weight
				im_debug[-self.features[0].patch_shape[0]:,:self.features[0].patch_shape[1],:] = (self.features[0].patch_weight.reshape(self.features[0].patch_shape)/np.amax(self.features[0].patch_weight))[:,:,np.newaxis]
				imsave( "./debug/dbg_scale_"+str(num_scales-scale-1)+"_init.png",  img_as_ubyte(im_debug.squeeze()) )

			err      = 1
			rel_err  = 1
			iters    = 0
			err_old1 = 1
			err_old2 = 2
			# for i in range(200):
			while ( (err>TOL) and (rel_err>(TOL/1000000.)) and (iters<=max_iters) ):
				self.update_weights(image)
				err = self.update_image(image.reshape(image.shape[0],-1))
				rel_err = abs(err-err_old1)/err_old1
				# mov_err = abs(err-err_old1)/abs(err_old1-err_old2) if iters==0 else 0.1*abs(err-err_old1)/abs(err_old1-err_old2)+0.9*mov_err
				mov_err = (err-err_old1) if iters==0 else 0.1*(err-err_old1)+0.9*mov_err
				# print("\titer %3d, err %f, rel_err %f, trend %4.2f"%(iters,err,rel_err,mov_err/err_old1))
				print("\titer %3d, err %f"%(iters,err))
				err_old2 = err_old1
				err_old1 = err

				# save iterations
				if debug:
					# convert to the channels last format
					im_debug = np.moveaxis(image,0,-1).copy()
					# im_debug[im_debug<0] = 0
					# im_debug[im_debug>1] = 1
					# add patch weight
					im_debug[-self.features[0].patch_shape[0]:,:self.features[0].patch_shape[1],:] = (self.features[0].patch_weight.reshape(self.features[0].patch_shape)/np.amax(self.features[0].patch_weight))[:,:,np.newaxis]
					imsave( "./debug/iters/scale_"+str(num_scales-scale-1)+"_iter_"+str(iters)+".png", img_as_ubyte(im_debug.squeeze()) )

				iters += 1
				# if iters%10==0 and mov_err>0.99:
				# 	break


			# truncate values to the allowed limits
			image[image<0] = 0
			image[image>1] = 1

			if debug:
				# imsave("./dbg/dbg_scale_"+str(num_scales-scale-1)+".png",img_as_ubyte(np.squeeze(np.moveaxis(image,0,-1))))
				# convert to the channels last format
				im_debug = np.moveaxis(image,0,-1).copy()
				# add patch weight
				im_debug[-self.features[0].patch_shape[0]:,:self.features[0].patch_shape[1],:] = (self.features[0].patch_weight.reshape(self.features[0].patch_shape)/np.amax(self.features[0].patch_weight))[:,:,np.newaxis]
				imsave( "./debug/dbg_scale_"+str(num_scales-scale-1)+"_final.png", img_as_ubyte(im_debug.squeeze()) )
		# except:
		# 		im_save = np.moveaxis(image,0,-1)
		# 		im_save[im_save<0] = 0
		# 		im_save[im_save>1] = 1
		# 		im_save[-self.features[0].patch_shape[0]:,:self.features[0].patch_shape[1],:] = (self.features[0].patch_weight.reshape(self.features[0].patch_shape)/np.amax(self.features[0].patch_weight))[:,:,np.newaxis]
		# 		imsave("./dbg/exception_scale_"+str(num_scales-scale-1)+".png",img_as_ubyte(np.squeeze(im_save)))

		# return np.squeeze(np.moveaxis(self.image_pyramid[0],0,-1))
		return np.squeeze(np.moveaxis(self.curr_image,0,-1))


	def inpaint_PDE(self, image, mask, type='biharmonic', aniso_coef=None):
		if image.ndim==2:
			image = image[np.newaxis,:,:]
		im_ch, im_h, im_w = image.shape
		im = image.reshape(im_ch,-1)

		ind_dof = op.masked_indices(mask)
		# ind_dof = op.masked_indices(np.ones_like(mask))

		aniso_coef = np.ones_like(mask) if aniso_coef is None else aniso_coef

		if type=='harmonic':
			# Asp = op.conv2mat((im_h,im_w),np.array([[0,1,0],[1,-4,1],[0,1,0]]),dtype=_im_dtype).tocsr()[ind_dof,:]
			Asp1 = op.conv2mat((im_h,im_w),np.array([[0,0,0],[0,-1,1],[0,0,0]]),dtype=_im_dtype).tocsr()
			Asp2 = op.conv2mat((im_h,im_w),np.array([[0,0,0],[0,-1,0],[0,1,0]]),dtype=_im_dtype).tocsr()
			Asp  = Asp1.T.dot(Asp1.multiply(aniso_coef.ravel()[:,np.newaxis])) + Asp2.T.dot(Asp2.multiply(aniso_coef.ravel()[:,np.newaxis]))
			Asp  = Asp.tocsr()[ind_dof,:]
		if type=='biharmonic':
			Asp = op.conv2mat((im_h,im_w),np.array([[0,0,1,0,0],[0,2,-8,2,0],[1,-8,20,-8,1],[0,2,-8,2,0],[0,0,1,0,0]]),dtype=_im_dtype).tocsr()[ind_dof,:]
		if type=='homogeneous':
			# set rhs of the nonlocal model to zero
			Asp = sp.csr_matrix( (ind_dof.size,im_h*im_w), dtype=_im_dtype )
			for k in self.filters:
				conv_mat = op.conv2mat((im_h,im_w),k,dtype=_im_dtype).tocsr()
				Asp += conv_mat.T[ind_dof,:].dot(conv_mat.multiply(aniso_coef.ravel()[:,np.newaxis]))
		A = Asp.tocsc()[:,ind_dof]

		ILU     = sp.linalg.splu(A)
		precond = sp.linalg.LinearOperator( A.shape, lambda v: ILU.solve(v), dtype=_im_dtype )

		for ch in range(im_ch):
			F = A.dot(im[ch][ind_dof]) - Asp.dot(im[ch])
			# F = im[ch]
			delta, info = sp.linalg.cg( A, F - A.dot(im[ch][ind_dof]), x0=np.zeros(F.shape,dtype=_im_dtype), tol=1e-6, maxiter=1000, M=precond )
			if info!=0:
				exit("CG in inpaint_PDE not converged, code %d %f"%(info,np.amax(np.abs(delta))))

			im[ch][ind_dof] += delta

		# note that image is modified by reference
		image[image<0] = 0.0
		image[image>1] = 1.0




def cgrad(im):
	du_dx = correlate2d(im,[[0,0,0],[-1,0,1],[0,0,0]],mode='same',boundary='symm')
	du_dy = correlate2d(im,[[0,-1,0],[0,0,0],[0,1,0]],mode='same',boundary='symm')
	return du_dx, du_dy
def fgrad(im):
	du_dx = correlate2d(im,[[0,0,0],[0,-1,1],[0,0,0]],mode='same',boundary='symm')
	du_dy = correlate2d(im,[[0,0,0],[0,-1,0],[0,1,0]],mode='same',boundary='symm')
	return du_dx, du_dy
def bgrad(im):
	du_dx = correlate2d(im,[[0,0,0],[-1,1,0],[0,0,0]],mode='same',boundary='symm')
	du_dy = correlate2d(im,[[0,-1,0],[0,1,0],[0,0,0]],mode='same',boundary='symm')
	return du_dx, du_dy
def bdivergence(im_x,im_y):
	return correlate2d(im_x,[[0,0,0],[-1,1,0],[0,0,0]],mode='same',boundary='symm') + correlate2d(im_y,[[0,-1,0],[0,1,0],[0,0,0]],mode='same',boundary='symm')
def laplacian(im):
	return correlate2d(im,[[0,1,0],[1,-4,1],[0,1,0]],mode='same',boundary='symm')



def inpaint_Navier_Stokes(image, mask):
	# im_ch,im_h,im_w = image.shape
	# vorticity = np.zeros_like(image)
	# for ch in range(im_ch):
	# 	vorticity[ch,...] = correlate2d(image[ch],[[0,1,0],[1,-4,1],[0,1,0]],mode='same',boundary='symm')

	# image[mask] = 0.5
	# mask = binary_dilation( mask, selem=morph.rectangle(1,1) )

	eps = 1.e-6

	dt = 0.1
	aa = np.zeros_like(image)
	for n in range(10000):
		fdu_dx, fdu_dy = fgrad(image)
		bdu_dx, bdu_dy = bgrad(image)
		n_x, n_y = -fdu_dy / np.sqrt(fdu_dx**2+fdu_dy**2+eps), fdu_dx / np.sqrt(fdu_dx**2+fdu_dy**2+eps)
		lapl_dx, lapl_dy = cgrad(laplacian(image))

		beta = lapl_dx*n_x + lapl_dy*n_y

		aa[beta>0] = np.sqrt( np.fmin(bdu_dx,0)**2 + np.fmin(bdu_dy,0)**2 + np.fmax(fdu_dx,0)**2 + np.fmax(fdu_dy,0)**2 )[beta>0]
		aa[beta<0] = np.sqrt( np.fmax(bdu_dx,0)**2 + np.fmax(bdu_dy,0)**2 + np.fmin(fdu_dx,0)**2 + np.fmin(fdu_dy,0)**2 )[beta<0]

		image[mask] = (image + dt * beta * aa)[mask]
	# image = lapl_dy*du_dx - lapl_dx*du_dy

	print([np.amin(image[mask]),np.amax(image[mask])])
	# image[image<0] = 0.0
	# image[image>1] = 1.0
	return image


# def anisotropic_diffusion(image,mask,iters=100):
# 	dt = 0.1
# 	for n in range(iters):
# 		fdu_dx, fdu_dy = fgrad(image)
# 		du_norm = np.sqrt(fdu_dx**2+fdu_dy**2)
# 		fdu_dx, fdu_dy = fdu_dx / du_norm, fdu_dy / du_norm
# 		image[mask] = ( image + dt * du_norm * bdivergence(fdu_dx,fdu_dy) )[mask]
# 	return image

def anisotropic_diffusion(img,mask, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
	# From medpy: https://loli.github.io/medpy/

    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return np.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return np.where(np.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = np.array(img, dtype=np.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [np.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
            slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
            deltas[i][slicer] = np.diff(out, axis=i)

        # update matrices
        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
            matrices[i][slicer] = np.diff(matrices[i], axis=i)

        # update the image
        out[mask] += gamma * (np.sum(matrices, axis=0))[mask]

    return out