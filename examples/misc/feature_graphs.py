import time
from pathlib import Path
import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
from skimage.util import montage
from skimage.exposure import rescale_intensity
from skimage.io   import imread, imsave
from skimage.transform import rescale, resize
from skimage.draw import line_aa
from skimage.color import rgb2grey
from sklearn.feature_extraction.image import extract_patches_2d

import inpainting.utils as op

_im_dtype     = np.float32
_NN_algorithm = "FLANN"

if _NN_algorithm=="Sklearn":
	from sklearn.neighbors import NearestNeighbors
if _NN_algorithm=="PatchMatch":
	import patchmatch as pm
if _NN_algorithm=="FLANN":
	import pyflann as flann
if _NN_algorithm=="FAISS":
	import faiss

_counter = 1


class NNF():
	def __init__(self, image, target_mask, source_mask, patch_size=(11,11), patch_weight=None, num_neighbors=1):
		im_h, im_w, im_ch = image.shape

		if patch_weight is None:
			self.patch_weight = np.ones(patch_size, dtype=_im_dtype)

		self.patch_size = patch_size
		self.num_neighb = num_neighbors

		print("Build NNF index: ", end=" ")
		start = time.time()

		if _NN_algorithm!="PatchMatch":
			self.source_ind = op.masked_indices(source_mask)
			self.target_ind = op.masked_indices(target_mask)

			# convert array indices to patch indices
			pad = patch_size[0]//2
			ind_y, ind_x = np.divmod(self.source_ind, im_w)
			self.source_ind = (ind_x - pad) + (ind_y - pad)*(im_w-2*pad)

			source_point_cloud = extract_patches_2d( image, patch_size=patch_size )[self.source_ind].reshape((self.source_ind.size,-1)) \
							   * np.repeat(np.sqrt(self.patch_weight),im_ch)

			# need this because of FLANN bug (?) with memory release
			self.target_point_cloud = np.zeros((self.target_ind.size,source_point_cloud.shape[-1]), dtype=_im_dtype)

		if _NN_algorithm=="FLANN":
			self.nn = flann.FLANN()
			self.nn.build_index(source_point_cloud, algorithm="kdtree", trees=1) #, log_level = "info")
		elif _NN_algorithm=="Sklearn":
			self.nn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='kd_tree', metric='minkowski', n_jobs=-1) #,metric_params={'w':self.patch_weight})
			self.nn.fit(X=source_point_cloud)
		elif _NN_algorithm=="FAISS":
			self.nn = faiss.IndexHNSWFlat(source_point_cloud.shape[1], 50)
			self.nn.add(source_point_cloud)

		if _NN_algorithm=="PatchMatch":
			self.nn = pm.PatchMatch(target_mask, source_mask, patch_size=patch_size, lambdas=np.ones_like(image, dtype=_im_dtype) )

		print('%f sec' % (time.time()-start))


	def calculate_nnf(self, image, init_guess=None):
		im_h, im_w, im_ch = image.shape

		print("Query NNF index: ", end=" ")
		start = time.time()

		if _NN_algorithm!="PatchMatch":
			ind_nnf  = np.zeros((im_h*im_w,self.num_neighb), dtype='int32')
			dist_nnf = np.zeros((im_h*im_w,self.num_neighb))

			# convert array indices to patch indices
			pad = self.patch_size[0]//2
			ind_y, ind_x = np.divmod(self.target_ind,im_w)
			ind = (ind_x - pad) + ( ind_y - pad )*(im_w-2*pad)

			# need this because of FLANN bug (?) with memory release
			np.copyto(self.target_point_cloud, extract_patches_2d(image,patch_size=self.patch_size)[ind].reshape((self.target_ind.size,-1)), casting='same_kind', where=True)
			self.target_point_cloud *= np.repeat(np.sqrt(self.patch_weight),im_ch)

		# note that "ind" are patch indices, not array indices
		if _NN_algorithm=="FLANN":
			ind, dist = self.nn.nn_index(self.target_point_cloud, self.num_neighb)
		elif _NN_algorithm=="Sklearn":
			dist, ind = self.nn.kneighbors(X=self.target_point_cloud, return_distance=True)
		elif _NN_algorithm=="FAISS":
			dist, ind = self.nn.search(self.target_point_cloud, self.num_neighb)

		if _NN_algorithm!="PatchMatch":
			ind  = ind.reshape((ind.shape[0],self.num_neighb))
			dist = dist.reshape((dist.shape[0],self.num_neighb))

			# convert patch indices to array indices
			ind = self.source_ind[ind.ravel()]
			ind_y, ind_x = np.divmod(ind,im_w-2*pad)
			ind = (ind_x + pad) + (ind_y + pad)*im_w

			ind = np.reshape(ind, (-1,self.num_neighb))
			for n in range(self.num_neighb):
				ind_nnf[self.target_ind,:]  = ind #[:,n]
				dist_nnf[self.target_ind,:] = dist #[:,n]
		elif _NN_algorithm=="PatchMatch":
			ind_nnf, dist_nnf = self.nn.find_nnf( image, init_guess=init_guess )

		print('%f sec' % (time.time()-start))


		ind_nnf  = ind_nnf.reshape((-1,self.num_neighb))
		dist_nnf = dist_nnf.reshape((-1,self.num_neighb))

		return ind_nnf, dist_nnf


class feature_graph:
	def __init__(self, image, mask, feature_mask=None, kernels=[[1]], patch_size=11, num_neighbors=1):
		im_h, im_w, im_ch = image.shape
		half_patch_size = patch_size//2

		self.patch_size = patch_size
		self.half_patch_size = half_patch_size

		self.feature_mask = np.ones_like(mask) if feature_mask is None else feature_mask

		self.image = image
		self.mask  = mask
		self.num_neighb = num_neighbors

		# image features
		self.features = []
		for ker in kernels:
			feature = np.zeros_like(image)
			for ch in range(im_ch):
				feature[:,:,ch] = correlate2d(image[:,:,ch], ker, mode='same')
			self.features.append(feature)
		features_im = np.concatenate(self.features, axis=2)

		# make source mask
		source_mask = op.extend_mask_nonlocal(mask.astype(bool), kernel=np.ones((patch_size,patch_size))).copy()
		source_mask = np.logical_not(source_mask)
		# remove border from the source mask to account for patch size
		source_mask[:half_patch_size,:] = False
		source_mask[:,:half_patch_size] = False
		source_mask[mask.shape[0]-half_patch_size:,:] = False
		source_mask[:,mask.shape[1]-half_patch_size:] = False
		# plt.imshow(source_mask,'gray')
		# plt.show()
		# exit()
		source_mask = np.logical_and(source_mask, self.feature_mask)

		# make target mask
		target_mask = np.zeros(mask.shape);
		target_mask[half_patch_size:mask.shape[0]-half_patch_size,half_patch_size:mask.shape[1]-half_patch_size] = True
		target_mask = np.logical_and(target_mask, self.feature_mask)

		# find nearest neighbor field
		nnf = NNF(features_im, target_mask, source_mask, patch_size=(patch_size,patch_size), num_neighbors=num_neighbors)
		self.ind_nnf, self.dist_nnf = nnf.calculate_nnf(features_im)

		# relative linear indices of pixels in zero-centered patches
		self.patch_h = np.zeros((patch_size*patch_size,),dtype=np.int32)
		for i in range(patch_size):
			for j in range(patch_size):
				self.patch_h[i*patch_size+j] = (j-half_patch_size) + (i-half_patch_size)*image.shape[1]




if __name__ == '__main__':

	Path("output/").mkdir(parents=True, exist_ok=True)

	# read image and mask
	mask  = imread("./data/mask.png", as_gray=True)
	image = imread("./data/image.png", as_gray=False)[:,:,:3] / 255.
	image = image.astype('float32')

	table_mask   = imread("./data/table_area.png", as_gray=True)
	window_mask  = imread("./data/window_area.png", as_gray=True)
	printer_mask = imread("./data/printer_area.png", as_gray=True)

	# image = rescale(image,0.3,multichannel=True)
	# mask  = rescale(mask,0.3).astype('bool')
	# table_mask   = rescale(table_mask,0.3).astype('bool')
	# window_mask  = rescale(window_mask,0.3).astype('bool')
	# printer_mask = rescale(printer_mask,0.3).astype('bool')

	num_neighbors = 1 if _NN_algorithm=="PatchMatch" else 50


	# image features
	ker_dx, ker_dy = op.grad_kernels()
	ker_dx = np.array(ker_dx)
	ker_dy = np.array(ker_dy)
	lapl   = np.array(op.laplacian_kernel()[0])


	feature_graphs = []
	feature_graphs.append( feature_graph(image, mask, feature_mask=printer_mask, kernels=[[[1]]], patch_size=17, num_neighbors=num_neighbors) )
	feature_graphs.append( feature_graph(image, mask, feature_mask=table_mask,   kernels=[20*ker_dy, 20*ker_dx, [[1]]], patch_size=21, num_neighbors=num_neighbors) )
	feature_graphs.append( feature_graph(image, mask, feature_mask=window_mask,  kernels=[10*lapl, [[1]]], patch_size=25, num_neighbors=num_neighbors) )


	# feature masks
	im = np.ones_like(image)
	for graph_id, graph in enumerate(feature_graphs):
		im[graph.feature_mask.astype('bool'), graph_id%3] -= 0.3
		# im[graph.feature_mask.astype('bool'), :] -= 0.2
	im[mask.astype('bool'), :] -= 0.2
	lmd = 0.3
	im = lmd*image + (1-lmd)*im
	# feature images
	features_im = [im]
	im_h, im_w, im_ch = image.shape
	shift = int(0.4*im_w)
	for i, graph in enumerate(feature_graphs):
		shift_x = (len(graph.features)-1) * shift
		shift_y = (len(graph.features)-1) * shift
		im = np.ones((im_h+shift_y,im_w+shift_x,im_ch))
		for j in range(len(graph.features)-1,-1,-1):
			feat = rescale_intensity(graph.features[j], in_range=(0,1))
			shift_x = j * shift
			shift_y = (len(graph.features)-1-j) * shift
			im[shift_y:shift_y+im_h,shift_x:shift_x+im_w,:] = feat
		features_im.append(resize(im, image.shape))
	features_im = montage( features_im, multichannel=True, grid_shape=(1,len(feature_graphs)+1) )
	imsave("./output/features.png", rescale(features_im,0.5))



	def plot_graph(x,y):
		im_width = image.shape[1]

		# linear index of the chosen pixel
		target_ind = np.array([y * im_width + x]).astype('int32')

		plt.cla()
		img = image.copy()
		feat_graphs = [img]
		for graph_id, graph in enumerate(feature_graphs):
			# index and distance to the nearest neighbor
			nns, dists = graph.ind_nnf[target_ind], graph.dist_nnf[target_ind]
			nns   = nns[0,:]
			dists = dists[0,:]

			# feature graph image
			feat_graph = np.ones_like(image)
			# feat_graph[graph.feature_mask.astype('bool'),:] -= 0.2
			feat_graph[graph.feature_mask.astype('bool'),graph_id%3] -= 0.2
			# feat_graph[graph.feature_mask.astype('bool'),:] -= 0.2
			feat_graph[mask.astype('bool'),:] -= 0.2
			for nn_id in range(graph.num_neighb-1,-1,-1):
				lmd = nn_id / graph.num_neighb
				nn, dist = nns[nn_id], dists[nn_id]

				# coordinates of the nearest neighbor
				# nn_x = np.array( nn%im_width, dtype=np.int32 )
				# nn_y = np.array( nn//im_width, dtype=np.int32 )
				nn_x = nn%im_width
				nn_y = nn//im_width
				print('nn_id=%d, nn_x=%d, nn_y=%d nn_dist=%f'%(nn, nn_x, nn_y, dist))

				# show patches in the image
				for k in range(-graph.half_patch_size,graph.half_patch_size+1):
					img[y-graph.half_patch_size,x+k] = 0
					img[y+graph.half_patch_size,x+k] = 0
					img[y+k,x-graph.half_patch_size] = 0
					img[y+k,x+graph.half_patch_size] = 0

					img[nn_y-graph.half_patch_size,nn_x+k] = 0
					img[nn_y+graph.half_patch_size,nn_x+k] = 0
					img[nn_y+k,nn_x-graph.half_patch_size] = 0
					img[nn_y+k,nn_x+graph.half_patch_size] = 0


				# show graph edges
				if graph.feature_mask[y,x]:
					rr, cc, _ = line_aa(y,x,nn_y,nn_x)
					feat_graph[rr,cc,:]     = lmd * feat_graph[rr,cc,:]
					feat_graphs[0][rr,cc,:] = 0.0 #lmd * feat_graphs[0][rr,cc,:] #np.reshape( 1 - val, (-1,1))

				# show patches in the graph
				feat_graph = feat_graph.reshape((-1,3))
				img1 = image.reshape((-1,3))
				target_patch = graph.patch_h + target_ind
				source_patch = graph.patch_h + nn
				# for i, feat in enumerate(graph.features):
				for i in range(len(graph.features)-1,-1,-1):
					feat  = rescale_intensity(graph.features[i], in_range=(0,1))
					shift = i * int(0.3*graph.patch_size) * (1-im_width)
					# target patch
					feat_graph[target_patch+shift,:] = feat.reshape((-1,3))[target_patch,:]
					# source patch
					if graph.feature_mask[y,x]:
						feat_graph[source_patch+shift,:] = lmd * feat_graph[source_patch+shift,:] + (1-lmd) * feat.reshape((-1,3))[source_patch,:]
				feat_graph = feat_graph.reshape(image.shape)

				# if graph.feature_mask[y,x]:
				# 	plt.gca().add_line(mlines.Line2D([x,nn_x], [y,nn_y], color='black', linewidth=(graph.num_neighb-nn_id)/graph.num_neighb))
				# 	plt.gca().add_line(mlines.Line2D([x+(graph_id+1)*im_width,nn_x+(graph_id+1)*im_width], [y,nn_y], color='black', linewidth=(graph.num_neighb-nn_id)/graph.num_neighb))
			feat_graphs.append(feat_graph)

		feat_graphs = montage( feat_graphs, multichannel=True, grid_shape=(1,len(feat_graphs)) )
		plt.imshow(feat_graphs)
		imsave("./output/feature_graph_"+str(x)+"_"+str(y)+".png", rescale(feat_graphs,0.5))
		plt.draw()



	fig = plt.figure(1)
	def onclick(event):
		im_width = image.shape[1]

		# coordinates of the chosen pixel
		x = np.floor(event.xdata).astype('int32') % im_width
		y = np.floor(event.ydata).astype('int32')
		print('%s click: button=%d, x=%d, y=%d, px_x=%d, px_y=%d' % ('double' if event.dblclick else 'single', event.button, event.x, event.y, x, y))

		plot_graph(x,y)

	cid = fig.canvas.mpl_connect('button_press_event', onclick)
	plot_graph(205,365)
	plot_graph(227,270)
	# plot_graph(205//3,365//3)
	plt.show()


