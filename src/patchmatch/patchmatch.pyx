cimport c_patchmatch as c_pm
import numpy as np

# This is Cython .pyx source file

cpdef pm_64( source, source_mask, source_ind, source_y, source_x, source_ch, target, target_mask, target_ind, target_y, target_x, target_ch, target_size, neighbors, distances, patch_ind, patch_size, weight, lambdas, max_rand_shots, max_iterations, max_window_size, TOL ):
	cdef char[::1]	c_source_mask = np.ravel(source_mask)
	cdef char[::1]	c_target_mask = np.ravel(target_mask)
	cdef double[::1] c_source  = np.ravel(source)
	cdef double[::1] c_target  = np.ravel(target)
	cdef double[::1] c_weight  = np.ravel(weight)
	cdef double[::1] c_lambdas = np.ravel(lambdas)
	cdef double[::1] c_distances  = distances
	cdef int[::1]   c_neighbors  = neighbors
	cdef int[::1]   c_source_ind = source_ind
	cdef int[::1]   c_target_ind = target_ind
	cdef int[::1]   c_patch_ind  = patch_ind

	c_pm.pm_64( &c_source[0], &c_source_mask[0], &c_source_ind[0], source_y, source_x, source_ch, &c_target[0], &c_target_mask[0], &c_target_ind[0], target_y, target_x, target_ch, target_size, &c_neighbors[0], &c_distances[0], &c_patch_ind[0], patch_size, &c_weight[0], &c_lambdas[0], max_rand_shots, max_iterations, max_window_size, TOL )

	return neighbors, distances

cpdef pm( source, source_mask, source_ind, source_y, source_x, source_ch, target, target_mask, target_ind, target_y, target_x, target_ch, target_size, neighbors, distances, patch_ind, patch_size, weight, lambdas, max_rand_shots, max_iterations, max_window_size, TOL ):
	cdef char[::1]	c_source_mask = np.ravel(source_mask)
	cdef char[::1]	c_target_mask = np.ravel(target_mask)
	cdef float[::1] c_source  = np.ravel(source)
	cdef float[::1] c_target  = np.ravel(target)
	cdef float[::1] c_weight  = np.ravel(weight)
	cdef float[::1] c_lambdas = np.ravel(lambdas)
	cdef float[::1] c_distances  = distances
	cdef int[::1]   c_neighbors  = neighbors
	cdef int[::1]   c_source_ind = source_ind
	cdef int[::1]   c_target_ind = target_ind
	cdef int[::1]   c_patch_ind  = patch_ind

	c_pm.pm( &c_source[0], &c_source_mask[0], &c_source_ind[0], source_y, source_x, source_ch, &c_target[0], &c_target_mask[0], &c_target_ind[0], target_y, target_x, target_ch, target_size, &c_neighbors[0], &c_distances[0], &c_patch_ind[0], patch_size, &c_weight[0], &c_lambdas[0], max_rand_shots, max_iterations, max_window_size, TOL )

	return neighbors, distances


class PatchMatch:
	"""
	Image inpainting class.

	Parameters
	----------
	source : float32 image
		source image
	target : float32 image
		target image
	source_ind : int32 array
		linear indices in the source image

	Attributes
	----------
	im_w : int
		width of the image
	"""

	def __init__(self, target_mask, source_mask=None, patch_size=(3,3), weight=None, lambdas=None, max_iterations=10, max_rand_shots=10, max_window_size=-1, TOL=1.e-5):
		assert (patch_size[0]==patch_size[1]), "patch is assumed to be square but rectangular is given"
		self.patch_size = int(patch_size[0])

		self.dtype = lambdas.dtype

		# source and target masks
		self.target_mask = target_mask.astype(np.byte)
		if source_mask is None:
			self.source_mask = np.logical_not(target_mask).astype(np.byte)
		else:
			self.source_mask = np.array(source_mask,dtype=np.byte)

		# patch weighting kernel
		if weight is None:
			self.weight = np.ones(patch_size,dtype=self.dtype).ravel() / (patch_size[0]*patch_size[1])
		else:
			self.weight = weight.ravel()

		# relative linear indices of pixels in zero-centered patch
		self.patch_h = np.zeros((self.patch_size*self.patch_size,),dtype=np.int32)
		for i in range(self.patch_size):
			for j in range(self.patch_size):
				self.patch_h[i*self.patch_size+j] = j - self.patch_size//2 + (i-self.patch_size//2)*self.target_mask.shape[1]

		self.source_ind = np.nonzero(np.ravel(self.source_mask,order='C'))[0].astype(np.int32)
		self.target_ind = np.nonzero(np.ravel(self.target_mask,order='C'))[0].astype(np.int32)
		# self.source_ind = op.masked_indices(self.source_mask).astype(np.int32)
		# self.target_ind = op.masked_indices(self.target_mask).astype(np.int32)

		self.nn_dist = np.zeros((self.target_mask.size,),dtype=self.dtype)

		self.lambdas = lambdas.ravel()

		self.max_iterations  = max_iterations
		self.max_rand_shots  = max_rand_shots
		self.max_window_size = max_window_size
		self.TOL = np.array(TOL,dtype=self.dtype)


	# def find_nn(self,image,target_ind,init_guess=None):
	# 	if init_guess is None:
	# 		# nn_ind = np.zeros(target_ind.shape,dtype=np.int32)
	# 		nn_ind = -1*np.ones((self.target_mask.size,),dtype=np.int32)
	# 		nn_ind[target_ind] = np.random.choice(self.source_ind,size=target_ind.size)
	# 	else:
	# 		nn_ind = np.array(init_guess,dtype=np.int32)
	# 	# nn_dist = np.zeros(target_ind.shape,dtype=np.float32).ravel()
	# 	nn_dist = np.zeros((self.target_mask.size,),dtype=np.float32)

	# 	img = image.astype(np.float32)
	# 	im_ch = 1 if img.ndim==2 else img.shape[2]

	# 	# pm( img, self.source_mask, self.source_ind, img, self.target_mask, target_ind, nn_ind, nn_dist, img.shape[0], img.shape[1], im_ch, self.source_ind.size, target_ind.size, self.patch_size, self.weight, self.max_rand_shots, self.max_iterations, self.max_window_size )
	# 	pm( img, self.source_mask, self.source_ind, img.shape[0], img.shape[1], im_ch,
	# 		img, self.target_mask, target_ind, img.shape[0], img.shape[1], im_ch, target_ind.size,
	# 		nn_ind, nn_dist, self.patch_h, self.patch_h.size, self.weight, self.lambdas, self.max_rand_shots, self.max_iterations, self.max_window_size, self.TOL )
	# 	return nn_ind[target_ind], nn_dist[target_ind]


	def find_nnf(self, image, init_guess=None, max_rand_shots=None):
		if init_guess is None:
			# nn_ind = -1*np.ones((self.target_mask.size,),dtype=np.int32)
			# nn_ind[self.target_ind] = self.source_ind[0]
			nn_ind = np.zeros((self.target_mask.size,), dtype=np.int32)
			nn_ind[self.target_ind] = np.random.choice(self.source_ind, size=self.target_ind.size)
		else:
			nn_ind = init_guess.ravel() if init_guess.dtype==np.int32 else init_guess.astype(np.int32).ravel()
			# nn_ind[self.target_ind] = np.random.choice(self.source_ind,size=self.target_ind.size)

		max_rand_shots = self.max_rand_shots if max_rand_shots is None else max_rand_shots

		img = image.ravel() #if image.dtype==np.float32 else image.astype(np.float32).ravel()
		# im_ch = 1 if image.ndim==2 else image.shape[0]
		# im_ch,im_h,im_w = (1,image.shape[1],image.shape[2]) if image.ndim==2 else image.shape
		im_h,im_w,im_ch = (image.shape[1],image.shape[2],1) if image.ndim==2 else image.shape
		# im_ch,im_h,im_w = (1,image.shape[0],image.shape[1]) if image.ndim==2 else (image.shape[2],image.shape[0],image.shape[1])

		if self.dtype==np.float32:
			pm( img, self.source_mask, self.source_ind, im_h, im_w, im_ch,
				img, self.target_mask, self.target_ind, im_h, im_w, im_ch, self.target_ind.size,
				nn_ind, self.nn_dist, self.patch_h, self.patch_h.size, self.weight, self.lambdas, max_rand_shots, self.max_iterations, self.max_window_size, self.TOL )
		elif self.dtype==np.float64:
			pm_64( img, self.source_mask, self.source_ind, im_h, im_w, im_ch,
				   img, self.target_mask, self.target_ind, im_h, im_w, im_ch, self.target_ind.size,
				   nn_ind, self.nn_dist, self.patch_h, self.patch_h.size, self.weight, self.lambdas, max_rand_shots, self.max_iterations, self.max_window_size, self.TOL )

		return nn_ind, self.nn_dist