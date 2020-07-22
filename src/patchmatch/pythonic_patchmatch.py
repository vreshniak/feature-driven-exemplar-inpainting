from numba import njit
import numpy as np
import src.operators as op
import matplotlib.pyplot as plt


Wclass PatchMatch:
	def __init__(self,target_mask,source_mask=None,patch_size=(3,3),weight=None,lambdas=None,max_iterations=10,max_rand_shots=10,max_window_size=-1,TOL=1.e-5):
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

		self.source_ind = op.masked_indices(self.source_mask).astype(np.int32)
		self.target_ind = op.masked_indices(self.target_mask).astype(np.int32)

		self.target_mask = self.target_mask.ravel()
		self.source_mask = self.source_mask.ravel()


		self.nn_dist = np.zeros((self.target_mask.size,),dtype=self.dtype)

		self.lambdas = lambdas.ravel()

		self.max_iterations  = max_iterations
		self.max_rand_shots  = max_rand_shots
		self.max_window_size = max_window_size
		self.TOL = np.array(TOL,dtype=self.dtype)

	
	@njit
	def find_nnf(self,image,init_guess=None):
		# if init_guess is None:
		# 	nn_ind = np.zeros((self.target_mask.size,),dtype=np.int32)
		# 	nn_ind[self.target_ind] = np.random.choice(self.source_ind,size=self.target_ind.size)
		# else:
		# nn_ind = init_guess.ravel() if init_guess.dtype==np.int32 else init_guess.astype(np.int32).ravel()
		nn_ind = init_guess.ravel()

		img = image.ravel()
		im_ch,im_h,im_w = (1,image.shape[1],image.shape[2]) if image.ndim==2 else image.shape


		def L2d2(ps, pt):
			dist = 0.0
			for ch in range(im_ch):
				offset_s = ch*im_h*im_w + ps
				offset_t = ch*im_h*im_w + pt
				for i in range(self.patch_h.size):
					s_id = offset_s + self.patch_h[i]
					t_id = offset_t + self.patch_h[i]
					dist += self.lambdas[t_id] * (img[s_id] - img[t_id])**2 * self.weight[i]
			return dist


		no_improve_iters = 0
		max_dist = 0.0
		# Improve NNF
		for it in range(self.max_iterations):
			if it%2==0:
				ind_begin = 0
				ind_end   = self.target_ind.size
				shift     = -1
			else:
				ind_begin = self.target_ind.size - 1
				ind_end   = -1
				shift     = 1

			# loop through the target image
			for count in range(self.target_ind.size):
				ind  = ind_begin - count * shift
				dist = self.nn_dist[self.target_ind[ind]]
				nn   = nn_ind[self.target_ind[ind]]

				# Propagation step
				# Left\Right neighbor
				shifted_ind = self.target_ind[ind] + shift
				if shifted_ind>=0 and shifted_ind<self.target_mask.size and self.target_mask[shifted_ind]>0:
					candidate_nn = nn_ind[shifted_ind] - shift

					if candidate_nn>=0 and candidate_nn<self.source_mask.size and self.source_mask[candidate_nn]>0:
						candidate_dist = L2d2(candidate_nn, self.target_ind[ind])
						if candidate_dist < dist:
							dist = candidate_dist
							nn   = candidate_nn
				# Above\Below neighbor
				shifted_ind = self.target_ind[ind] + shift * im_w
				if shifted_ind>=0 and shifted_ind<self.target_mask.size and self.target_mask[shifted_ind]>0:
					candidate_nn = nn_ind[shifted_ind] - shift * im_w;

					if candidate_nn>=0 and candidate_nn<self.source_mask.size and self.source_mask[candidate_nn]>0:
						candidate_dist = L2d2(candidate_nn, self.target_ind[ind])
						if candidate_dist < dist:
							dist = candidate_dist
							nn   = candidate_nn


				# Random search step
				nn_x = nn%im_w
				nn_y = nn//im_w
				w_size = self.max_window_size
				while w_size>=1:
					# truncate window to account for the source image size
					x_min = nn_x - w_size if (nn_x-w_size)>0     else 0
					y_min = nn_y - w_size if (nn_y-w_size)>0     else 0
					x_max = nn_x + w_size if (nn_x+w_size)<im_w  else im_w-1
					y_max = nn_y + w_size if (nn_y+w_size)<im_h  else im_h-1

					# sample max_rand_shots pixels from the window around current nn
					for k in range(self.max_rand_shots):
						candidate_nn = im_w * ( y_min + np.random.rand()%(y_max-y_min) ) + x_min + np.random.rand()%(x_max-x_min)

						if self.source_mask[candidate_nn]>0:
							candidate_dist = L2d2(candidate_nn, self.target_ind[ind])
							if candidate_dist<dist:
								dist = candidate_dist
								nn   = candidate_nn

					w_size /= 2

				if dist < self.nn_dist[self.target_ind[ind]]:
					self.nn_dist[self.target_ind[ind]] = dist
					nn_ind[self.target_ind[ind]] = nn

			# Max distance
			max_dist_old = max_dist
			max_dist = self.nn_dist[self.target_ind[ind_begin]];
			for ind in range(ind_begin-shift,ind_end-np.sign(shift),-shift):
				if self.nn_dist[self.target_ind[ind]] > max_dist:
					max_dist = self.nn_dist[self.target_ind[ind]]
			no_improve_iters = no_improve_iters+1 if np.abs(max_dist-max_dist_old)<TOL else 0

			# Early break if desired tolerance is achieved or there is no improvement
			if it>=max_iterations or no_improve_iters>5 or max_dist < TOL:
				break

		return nn_ind, self.nn_dist
