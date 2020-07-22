import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from skimage.io        import imread
from sklearn.feature_extraction.image import extract_patches_2d

# from sklearn.neighbors import NearestNeighbors
# import pyflann    as flann
import patchmatch as pm
import utils as op


_im_dtype = np.float64


# TODO: redesign, move everything to main class and make this NN class


# class feature(ABC):
class feature:
	'''
	Feature class
	'''
	def __init__(self, image_shape, target_mask, source_mask, patch_shape=(3,3), patch_weight=None, lambdas=None, beta=None, nn_algorithm='PatchMatch', nnf_field=None):
		# read target mask
		if isinstance(target_mask, str):
			self.target_mask = imread(target_mask, as_gray=True).astype(np.bool)
		elif isinstance(target_mask, np.ndarray):
			self.target_mask = target_mask
		else:
			assert False, "target_mask must be either string or numpy array"

		# source mask from file or as "complement" of target
		if source_mask is None:
			self.source_mask = np.logical_not(self.target_mask)
		elif isinstance(source_mask, str):
			self.source_mask = imread(source_mask, as_gray=True).astype(np.bool)
		elif isinstance(source_mask, np.ndarray):
			self.source_mask = source_mask
		else:
			assert False, "source_mask must be either string, numpy array or None"

		# Gaussian patch
		self.patch_shape = patch_shape
		if patch_weight is None:
			self.patch_weight = np.ones(patch_shape,dtype=_im_dtype).ravel()
			self.patch_weight /= self.patch_weight.sum()
		else:
			self.patch_weight = patch_weight.ravel().astype(_im_dtype)

		##################################
		im_ch, im_h, im_w = image_shape

		self.num_filters = 1 if lambdas is None else len(lambdas)

		# if all lambdas are scalars, one can build the search tree
		if (lambdas is not None) and np.all([ np.isscalar(lmb) for lmb in lambdas ]):
			self.active_channels = np.nonzero(np.tile(lambdas,im_ch))[0]
			self.active_filters  = np.nonzero(lambdas)[0]
		else:
			self.active_channels = np.arange(im_ch*len(lambdas))
			self.active_filters  = np.arange(len(lambdas))

		if lambdas is None:
			self.orig_lambdas = np.ones((self.num_filters*im_ch,im_h,im_w),dtype=_im_dtype)
		else:
			self.orig_lambdas = np.array([ np.array(lmbd,dtype=_im_dtype)*np.ones((im_h,im_w),dtype=_im_dtype) for ch in range(im_ch) for lmbd in lambdas ])

		self.beta_const = 1.0 if beta is None else beta

		# choose algorithm for the nearest neighbors search
		self.nn_algorithm = nn_algorithm
		if self.nn_algorithm!='PatchMatch':
			print('Warning: '+self.nn_algorithm+' will work only with scalar lambdas')

		# explicit nnf field if given
		self.nnf_field = nnf_field


	def build_mask_pyramids(self, num_scales=10, coarsest_scale=0.1, mask_threshold=0.1):
		"""
		Build image and mask pyramids
		"""
		from skimage.transform import pyramid_gaussian

		def pyramid(im, multichannel=False):
			return pyramid_gaussian(im, max_layer=num_scales-1, downscale=downscale, multichannel=multichannel)

		if num_scales>1:
			downscale = coarsest_scale**(-1.0/(num_scales-1))
			self.target_mask_pyramid = [ i > mask_threshold for i in pyramid(self.target_mask) ]
			self.source_mask_pyramid = [ i > mask_threshold for i in pyramid(self.source_mask) ]
			# self.source_mask_pyramid = [ np.logical_and(np.logical_not(i), j>mask_threshold)  for i, j in zip(self.target_mask_pyramid, pyramid(self.source_mask))  ]

			self.lambda_pyramid = [ np.moveaxis(lmd,-1,0) for lmd in pyramid(np.moveaxis(self.orig_lambdas,0,-1),True) ]
		else:
			self.target_mask_pyramid = [ self.target_mask > mask_threshold ]
			self.source_mask_pyramid = [ self.source_mask > mask_threshold ]

			self.lambda_pyramid = [ self.orig_lambdas ]
			# self.beta_pyramid = [ self.orig_betas ]


	# def preprocess(self,scale,image,target_mask,source_mask):
	def build_nnf_index(self, scale, convolutions, target_mask, source_mask):
		active_convs    = convolutions[self.active_channels,...]
		im_ch,im_h,im_w = active_convs.shape

		self.nnf  = self.source_ind[0]*np.ones((target_mask.size,),dtype=np.int32)
		self.beta = np.zeros(target_mask.shape,dtype=_im_dtype).ravel()
		self.beta[self.target_ind] = self.beta_const


		if self.nnf_field is not None:
			# TODO: modify nnf_field for the given scale
			return

		if self.source_ind.size>0:
			if self.nn_algorithm=="FLANN" or self.nn_algorithm=="Sklearn":
				# feature_image = np.moveaxis(active_convs*np.sqrt(self.lambdas.reshape(active_convs.shape)),0,-1)
				# feature_image = np.moveaxis(active_convs*np.sqrt(self.lambdas[self.active_channels,0:1]),0,-1)

				# with FLANN, lambdas are assumed to be constant
				feature_image = np.moveaxis(active_convs,0,-1)*np.sqrt(self.lambdas[self.active_channels,0])

				# im_ch = 1 if (feature_image.ndim==2) else feature_image.shape[2]

				# convert array indices to patch indices
				pad = self.patch_shape[0]//2
				ind_y, ind_x = np.divmod(self.source_ind,im_w)
				self.source_ind = (ind_x - pad) + (ind_y - pad)*(im_w-2*pad)
				# self.source_ind = self.source_ind%im_w - pad + ( self.source_ind//im_w - pad )*(im_w-2*pad)

				source_point_cloud = extract_patches_2d( feature_image, patch_size=self.patch_shape )[self.source_ind].reshape((self.source_ind.size,-1)) \
								   * np.repeat(np.sqrt(self.patch_weight),im_ch)

				# need this because of FLANN bug (?) with memory release
				self.target_point_cloud = np.zeros((self.target_ind.size,source_point_cloud.shape[-1]))

			if self.nn_algorithm=="FLANN":
				start = time.time()
				self.nn = flann.FLANN()
				self.nn.build_index(source_point_cloud, algorithm="kdtree", trees=1) #, log_level = "info")
				# self.nn.build_index(source_point_cloud, algorithm="linear") #, log_level = "info")
				print("Building FLANN index ... ", time.time()-start)
			elif self.nn_algorithm=="Sklearn":
				print("Building Sklearn index ...", end =" ")
				start = time.time()
				self.nn = NearestNeighbors(n_neighbors=1,algorithm='kd_tree',metric='minkowski',n_jobs=-1) #,metric_params={'w':self.patch_weight})
				self.nn.fit(X=source_point_cloud)
				print("Building Sklearn index ... ", time.time()-start)
			elif self.nn_algorithm=="PatchMatch":
				self.max_rand_shots = 30
				lmd = np.moveaxis( self.lambdas[self.active_channels], 0, -1 )
				self.nn = pm.PatchMatch(target_mask, source_mask=source_mask, patch_size=self.patch_shape, weight=self.patch_weight, lambdas=lmd, max_iterations=20, max_rand_shots=self.max_rand_shots)
				# self.nn = pm.PatchMatch(target_mask,source_mask=source_mask,patch_size=self.patch_shape,weight=self.patch_weight,lambdas=self.lambdas,max_iterations=10,max_rand_shots=30)
				if scale==len(self.source_mask_pyramid)-1:
					# random initial guess
					self.nnf[self.target_ind] = np.random.choice(self.source_ind,size=self.target_ind.size)
					# self.nnf[self.target_ind] = self.source_ind[0]
				else:
					# interpolate nnf from coarser scale
					scale_0 = target_mask.shape[0] / self.prev_shape[0]
					scale_1 = target_mask.shape[1] / self.prev_shape[1]

					target_ind_0, target_ind_1 = np.unravel_index( self.target_ind, target_mask.shape )
					prev_target_ind = [ np.rint(target_ind_0/scale_0).astype(np.int32), np.rint(target_ind_1/scale_1).astype(np.int32) ]
					prev_target_ind = np.ravel_multi_index( prev_target_ind, self.prev_shape, mode='clip' )

					nnf_0 = np.rint(self.prev_nnf_0[prev_target_ind]*scale_0).astype(np.int32)
					nnf_1 = np.rint(self.prev_nnf_1[prev_target_ind]*scale_1).astype(np.int32)

					# estimate nnf
					self.nnf[self.target_ind] = np.ravel_multi_index( [nnf_0,nnf_1], target_mask.shape, mode='clip' )
					# self.nnf, _ = self.nn.find_nnf(active_convs,init_guess=self.nnf)

				if len(self.source_mask_pyramid)>1:
					self.prev_nnf_0, self.prev_nnf_1 = np.unravel_index( self.nnf, target_mask.shape )
					self.prev_shape = target_mask.shape


	# def calculate_nnf(self,image):
	def calculate_nnf(self, convolutions, nnf_dist=False):
		# if explicit nnf_field is given
		if self.nnf_field is not None:
			self.nnf = self.nnf_field
			return

		active_convs    = convolutions[self.active_channels,:,:]
		im_ch,im_h,im_w = active_convs.shape

		if self.source_ind.size>0:
			if self.nn_algorithm=="FLANN" or self.nn_algorithm=="Sklearn":
				# feature_image = np.moveaxis(active_convs*np.sqrt(self.lambdas.reshape(active_convs.shape)),0,-1)
				# feature_image = np.moveaxis(active_convs*np.sqrt(self.lambdas[self.active_channels,0:1]),0,-1)
				feature_image = np.moveaxis(active_convs,0,-1)*np.sqrt(self.lambdas[self.active_channels,0])

				# im_ch = 1 if (feature_image.ndim==2) else feature_image.shape[2]

				# convert array indices to patch indices
				pad = self.patch_shape[0]//2
				ind_y, ind_x = np.divmod(self.target_ind,im_w)
				ind = (ind_x - pad) + ( ind_y - pad )*(im_w-2*pad)
				# ind = self.target_ind%im_w - pad + ( self.target_ind//im_w - pad )*(im_w-2*pad)

				# target_point_cloud = extract_patches_2d(feature_image,patch_size=self.patch_shape)[ind].reshape((self.target_ind.size,-1))
				# target_point_cloud *= np.repeat(np.sqrt(self.patch_weight),im_ch)
				# # need this because of FLANN bug with memory release?
				# np.copyto(self.target_point_cloud, target_point_cloud, casting='same_kind', where=True)

				# need this because of FLANN bug (?) with memory release
				np.copyto(self.target_point_cloud, extract_patches_2d(feature_image,patch_size=self.patch_shape)[ind].reshape((self.target_ind.size,-1)), casting='same_kind', where=True)
				self.target_point_cloud *= np.repeat(np.sqrt(self.patch_weight),im_ch)

			if self.nn_algorithm=="FLANN":
				# note that "ind" are patch indices, not array indices
				ind, _ = self.nn.nn_index(self.target_point_cloud,1)
				# ind, dist = self.nn.nn(source_point_cloud, target_point_cloud, num_neighbors=1, algorithm="kdtree", trees=1)

				# convert patch indices to array indices
				ind = self.source_ind[ind.ravel()]
				ind_y, ind_x = np.divmod(ind,im_w-2*pad)
				ind = (ind_x + pad) + (ind_y + pad)*im_w
				# ind = ind%(im_w-2*pad) + pad + (ind//(im_w-2*pad) + pad)*im_w

				self.nnf[self.target_ind] = ind.ravel() #self.source_ind[ind.ravel()]
			elif self.nn_algorithm=="Sklearn":
				_, ind = self.nn.kneighbors(X=self.target_point_cloud,return_distance=True)

				# convert patch indices to array indices
				ind = self.source_ind[ind.ravel()]
				ind = ind%(im_w-2*pad) + pad + (ind//(im_w-2*pad) + pad)*im_w

				self.nnf[self.target_ind] = ind.ravel() #self.source_ind[ind.ravel()]

				# self.nnf[self.target_ind]  = self.source_ind[ind.ravel()]
			elif self.nn_algorithm=="PatchMatch":
				feature_image = np.moveaxis(active_convs,0,-1)
				if nnf_dist:
					self.nnf, self.nnf_dist = self.nn.find_nnf( feature_image, init_guess=self.nnf, max_rand_shots=max(1,self.max_rand_shots) )
				else:
					self.nnf, _ = self.nn.find_nnf( feature_image, init_guess=self.nnf )
				# self.max_rand_shots = self.max_rand_shots - 1