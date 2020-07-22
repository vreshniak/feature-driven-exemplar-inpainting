import numpy as np
import scipy.sparse as sp



_dtype = np.float64


def gauss_weight(patch_shape, patch_sigma=3):
	"""
	Gaussian patch weight
	"""
	patch_x, patch_y = np.meshgrid( np.arange(-(patch_shape[1]//2),patch_shape[1]//2+1), np.arange(-(patch_shape[0]//2),patch_shape[0]//2+1), sparse=False, indexing='xy')
	patch_weight = np.exp(-(patch_x**2+patch_y**2)/(patch_sigma**2),dtype=_dtype).ravel()
	return patch_weight / patch_weight.sum()


def masked_indices(mask):
	"""
	Find linear indices of the masked pixels
	"""
	return np.nonzero(np.ravel(mask,order='C'))[0]


def non_masked_indices(mask):
	"""
	Find linear indices of the non masked pixels
	"""
	return np.nonzero(np.ravel(mask-1,order='C'))[0]


def masked_bounding_box(mask):
	'''
	Bounding box of the masked region
	'''
	inp_ind_y, inp_ind_x = np.nonzero(mask)
	inp_top_left_x = np.amin(inp_ind_x);	inp_bot_rght_x = np.amax(inp_ind_x)
	inp_top_left_y = np.amin(inp_ind_y);	inp_bot_rght_y = np.amax(inp_ind_y)
	return [inp_top_left_y, inp_top_left_x, inp_bot_rght_y, inp_bot_rght_x]

# def mask_boundary_indices(mask):
# 	num_y, num_x = mask.shape
# 	indices = np.empty((num_x*num_y,),dtype='int32')
# 	ind = 0
# 	for i in range(mask.shape[0]):
# 		for j in range(mask.shape[1]):
# 			for k in range(-1,2):
# 				for l in range(-1,2):
# 					if not mask[i+k,j+l]:
# 						indices[ind] = j + i*num_x
# 			if mask[i,j]:
# 				indices[ind] = j + i*num_x
# 				ind += 1
# 	return indices[:ind].copy()


def extend_mask_nonlocal(mask,kernel=np.ones((3,3))):
	"""
	Extend inpainting mask to contain pixels in the support of the nonlocal kernel
	"""
	assert (mask.dtype is np.dtype(np.bool)), "input mask must be of bool type"

	ext_mask = mask.copy()
	inp_ind  = masked_indices(mask)
	im_h, im_w = mask.shape

	ker_y, ker_x = kernel.shape
	assert(ker_x%2>0), "kernel must have odd dimensions"
	assert(ker_y%2>0), "kernel must have odd dimensions"

	# indices of the nonzero kernel elements
	ker_ind_y, ker_ind_x = np.nonzero(kernel)
	ker_ind_x -= ker_x//2
	ker_ind_y -= ker_y//2

	for ind_x, ind_y in zip(inp_ind%im_w, inp_ind//im_w):
		for i, j in zip(ker_ind_x, ker_ind_y):
			ext_mask[min(max(0,ind_y+j),im_h-1),min(max(0,ind_x+i),im_w-1)] = True

	return ext_mask


def adjoint_conv_kernel(kernel):
	"""
	Adjoint convolution kernel
	"""
	return np.flip(kernel.ravel(),axis=0).reshape((kernel.shape[1],kernel.shape[0]))


def convmat(signal_size,kernel,dtype=_dtype):
	"""
	1D convolution (correlation) matrix with zero boundary conditions
	"""
	assert (kernel.size%2==1), "kernel is assumed to have odd number of elements"

	mat = sp.dia_matrix( (signal_size,signal_size), dtype=dtype )

	half_ker_size = kernel.size//2
	# correlation
	for i in range(-half_ker_size,half_ker_size+1):
		if ( kernel[half_ker_size+i]!=0 ):
			mat.setdiag(kernel[half_ker_size+i],i)
	# # convolution
	# for i in range(-half_ker_size,half_ker_size+1):
	# 	if ( kernel[half_ker_size-i]!=0 ):
	# 		mat.setdiag(kernel[half_ker_size-i],i)

	return mat


def conv2mat(im_shape,kernel,format="channels_first",dtype=_dtype):
	"""
	2D convolution (correlation) matrix with zero boundary conditions
	"""
	if len(im_shape)==2:
		im_size_y, im_size_x = im_shape
		num_channels = 1
	else:
		if format=="channels_last":
			im_size_y, im_size_x, num_channels = im_shape
		else:
			num_channels, im_size_y, im_size_x = im_shape

	ker_size_y, ker_size_x = kernel.shape
	ker_size = kernel.size
	kernel = kernel.ravel()

	mat_size_x = im_size_x * im_size_y
	mat = sp.dia_matrix( (mat_size_x,mat_size_x), dtype=dtype )

	# diagonal blocks corresponding to the rows of the kernel
	for j in range(-(ker_size_y//2),ker_size_y//2+1):
		# diagonal of the block corresponding to the given row of the kernel
		# correlation
		diag = sp.eye(im_size_y,im_size_y,-j,dtype=dtype)
		# # convolution
		# diag = sp.eye(im_size_y,im_size_y,-j,dtype=dtype)

		# contribution from convolution matrix corresponding to the given row of the kernel
		mat += sp.kron( diag, convmat(im_size_x,kernel[ker_size//2-ker_size_x//2-j*ker_size_x:ker_size//2-ker_size_x//2-(j-1)*ker_size_x],dtype=dtype) ) #.astype(dtype=dtype)

	if num_channels>1:
		if format=="channels_last":
			diag_ch = sp.eye(num_channels,num_channels,dtype=dtype)
			return sp.kron(mat,diag_ch) #.tocsr()
		else:
			diag_ch = sp.eye(num_channels,num_channels,dtype=dtype)
			return sp.kron(diag_ch,mat) #.tocsr()

	return mat #.tocsr()


def rgb2greymat(im_shape,format="channels_first",dtype=_dtype):
	"""
	Matrix converting rgb image to greyscale
	"""
	if len(im_shape)==2:
		im_size_y, im_size_x = im_shape
		num_channels = 1
	else:
		if format=="channels_last":
			im_size_y, im_size_x, num_channels = im_shape
		else:
			num_channels, im_size_y, im_size_x = im_shape
	im_size = im_size_x*im_size_y*num_channels

	mat_size_x = im_size_x * im_size_y
	if num_channels>1:
		if format=="channels_last":
			return sp.kron( sp.eye(mat_size_x,dtype=dtype), np.array([[0.2125,0.7154,0.0721]],dtype=dtype) )
		else:
			return sp.kron( np.array([[0.2125,0.7154,0.0721]],dtype=dtype), sp.eye(mat_size_x,dtype=dtype) )
	else:
		return sp.eye(mat_size_x,dtype=dtype)



def fill_region(image,mask,value=1):
	"""
	Fill masked region of the image with given value
	"""
	im = image.copy().ravel()
	if image.ndim > 2:
		im_h, im_w, im_ch = image.shape
	else:
		im_ch = 1
		im_h, im_w = self.image.shape
	# linear indices of masked pixels
	ind = masked_indices(mask)
	for i in ind:
		for ch in range(im_ch):
			im.data[i*im_ch+ch] = value
	return im.reshape(image.shape)



def apply_kernel(image,kernel,format="channels_first"):
	return conv2mat(image.shape,np.array(kernel),format=format).dot(image.ravel()).reshape(image.shape)


def stack_kernels(kernels):
	from skimage.util import pad
	max_ker_size_0 = max_ker_size_1 = 0
	for ker in kernels:
		npker = np.array(ker)
		max_ker_size_0 = max( max_ker_size_0, npker.shape[0] ) if isinstance(ker, list) else max( max_ker_size_0, ker.shape[0] )
		max_ker_size_1 = max( max_ker_size_1, npker.shape[1] ) if isinstance(ker, list) else max( max_ker_size_1, ker.shape[1] )
	result = np.zeros((len(kernels),max_ker_size_0,max_ker_size_1))
	for i,ker in enumerate(kernels):
		npker = np.array(ker)
		pad0  = (max_ker_size_0 - npker.shape[0]) // 2
		pad1  = (max_ker_size_1 - npker.shape[1]) // 2
		result[i,...] = pad(npker,((pad0,pad0),(pad1,pad1)))
	return result


def add_patch(img, patch):
	from skimage import img_as_float
	image = img_as_float(img.copy())
	if image.ndim==2:
		image[-patch.shape[0]:,:patch.shape[1]] = patch/np.amax(patch)
	elif image.ndim==3:
		image[-patch.shape[0]:,:patch.shape[1],:] = (patch/np.amax(patch))[:,:,np.newaxis]
	return image


###############################################################################
###############################################################################



def laplacian_kernel_2():
	return [ [[0.25,0.5,0.25],[0.25,-3,0.25],[0.25,0.5,0.25]] ]

def laplacian_kernel():
	return [ [[0,1,0],[1,-4,1],[0,1,0]] ]
	# return [ [[0.25,0.5,0.25],[0.25,-3,0.25],[0.25,0.5,0.25]] ]
	# gamma = 0.33
	# return [ (1-gamma)*np.array([[0,1,0],[1,-4,1],[0,1,0]]) + gamma*np.array([[0.5,0,0.5],[0,-2,0],[0.5,0,0.5]]) ]
	# return [ [[1,1,1],[1,-8,1],[1,1,1]] ]


def biharmonic_kernel():
	# return [ [[0,0,1,0,0],[0,0,-4,0,0],[1,-4,12,-4,1],[0,0,-4,0,0],[0,0,1,0,0]] ]
	return [ [[0,0,1,0,0],[0,2,-8,2,0],[1,-8,20,-8,1],[0,2,-8,2,0],[0,0,1,0,0]] ]



def second_order_central():
	return [ [[0,0,0],[1,-2,1],[0,0,0]], [[0,1,0],[0,-2,0],[0,1,0]] ]

# def third_order_forward():
# 	return [ [[0,0,0],[1,-2,1],[0,0,0]], [[0,1,0],[0,-2,0],[0,1,0]] ]

def fourth_order_central():
	return [ [[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]], [[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]] ]


def grad_kernels(mode="forward"):
	if mode=="forward":
		return [ [[0,0,0],[0,-1,1],[0,0,0]], [[0,0,0],[0,-1,0],[0,1,0]] ]
	elif mode=="backward":
		return [ [[0,0,0],[-1,1,0],[0,0,0]], [[0,-1,0],[0,1,0],[0,0,0]] ]
	else:
		raise NameError("wrong 'mode' option in grad_kernel")

def derivative_kernels(mode="forward",order=1,mixed=False):
	if mode=="forward":
		if order==2:
			if mixed:
				return [ [[0,0,0],[1,-2,1],[0,0,0]], [[0,1,0],[0,-2,0],[0,1,0]], [[0.25,0,-25],[0,0,0],[-25,0,25]] ]
			else:
				return [ [[0,0,0],[1,-2,1],[0,0,0]], [[0,1,0],[0,-2,0],[0,1,0]] ]
		if order==3:
			return [ [[0,0,0,0,0,0,0],[0,0,0,-1,3,-3,1],[0,0,0,0,0,0,0]], [[0,0,0,0,0,0,0]] ]
	elif mode=="backward":
		return None
	else:
		raise NameError("wrong 'mode' option in grad_kernel")


def nonlocal_laplacian(size=3,s=0.0):
	half_size = size//2
	X, Y = np.meshgrid( np.arange(-half_size,half_size+1), np.arange(-half_size,half_size+1), sparse=False, indexing='xy' )
	dist = X**2 + Y**2
	dist[size//2,size//2] = 1
	dist = dist**(1+s)
	# max_dist = np.amax(dist).astype(_dtype)
	max_dist = 1.0
	kernel = np.zeros((size,size))
	kernel.ravel()[...] = max_dist / dist.ravel()
	kernel[size//2,size//2] = 0.0
	kernel[size//2,size//2] = -np.sum(kernel.ravel())
	return [kernel]


def nonlocal_grad_kernels(size=3,s=0.0,sigma=None):
	half_size = size//2
	X, Y = np.meshgrid( np.arange(-half_size,half_size+1), np.arange(-half_size,half_size+1), sparse=False, indexing='xy' )
	dist = np.sqrt( X**2 + Y**2 )**(1+s)
	# max_dist = np.amax(dist).astype(_dtype)
	max_dist = 1.0
	num_kernels = size**2-1
	kernels = []

	weight = max_dist/dist.ravel() if sigma is None else np.sqrt(gauss_weight((size,size),patch_sigma=sigma).ravel())

	# for i in range(num_kernels+1):
	# 	if (2*i)!=num_kernels:
	# 		kernel = sp.coo_matrix( ([-weight[i],weight[i]],([half_size,i//size],[half_size,i%size])), shape=(size,size), dtype=_dtype )
	# 		kernels.append( kernel.tocsr() )
	# return kernels
	# mask1 = np.ones((11,11)).astype(np.bool)
	# mask1[1:-1,1:-1] = False
	# mask2 = np.ones((3,3)).astype(np.bool)
	# mask2[1:-1,1:-1] = False
	# ind1 = masked_indices(mask1)
	# ind2 = masked_indices(mask2)
	# for i in list(ind1)+list(ind2):
	for i in range(num_kernels+1):
		if (2*i)!=num_kernels:
			kernel = np.zeros((size,size))
			kernel.ravel()[num_kernels//2] = -weight[i]
			kernel.ravel()[i] = weight[i]
			# kernel.ravel()[num_kernels//2] = -max_dist/dist.ravel()[i]
			# kernel.ravel()[i] = max_dist/dist.ravel()[i]
			kernels.append( kernel )
	return kernels
# def nonlocal_grad_kernels(size=3,s=0.0):
# 	half_size = size//2
# 	X, Y = np.meshgrid( np.arange(-half_size,half_size+1), np.arange(-half_size,half_size+1), sparse=False, indexing='xy' )
# 	dist = np.sqrt( X**2 + Y**2 )**(1+s)
# 	num_kernels = size**2-1
# 	kernels = []
# 	for i in range(num_kernels+1):
# 		if (2*i)!=num_kernels:
# 			kernel = np.zeros((size,size))
# 			kernel.ravel()[num_kernels//2] = -1. / dist.ravel()[i]
# 			kernel.ravel()[i] = 1. / dist.ravel()[i]
# 			kernels.append( kernel )
# 	return kernels


def nonlocal_grad_x_kernels(size=3,s=0.0):
	half_size = size//2
	X = np.arange(-half_size,half_size+1)
	dist = np.sqrt( X**2 )**(0.5+s)
	num_kernels = 2*half_size
	kernels = []
	for i in range(num_kernels+1):
		if (2*i)!=num_kernels:
			kernel = np.zeros((size,size))
			kernel[int(num_kernels/2),int(num_kernels/2)] = -1. / dist[i]
			kernel[int(num_kernels/2),i] = 1. / dist[i]
			kernels.append( kernel )
	return kernels





def rotate(x,y,angle):
	c = np.cos(angle)
	s = np.sin(angle)
	x_theta = c * x - s * y
	y_theta = s * x + c * y
	return x_theta, y_theta


def generate_filter_support(sigma=(1,1),angle=0,nstd=3):
	x_max = max( abs(nstd * sigma[0] * np.cos(angle)), abs(nstd * sigma[1] * np.sin(angle)) )
	y_max = max( abs(nstd * sigma[0] * np.sin(angle)), abs(nstd * sigma[1] * np.cos(angle)) )
	x_max = y_max = np.ceil(max(x_max,y_max))
	return np.meshgrid( np.arange(-x_max,x_max+1), np.arange(y_max,-y_max-1,-1) )



def gauss1d(sigma=1,order=0,nstd=3,x=np.empty((0,)),normalize=True):
	"""
	Derivative of the 1d Gaussian filter

	"""
	assert sigma>0, "sigma cannot be equal to zero"

	x_max = nstd * sigma
	if x.size==0:
		x = np.arange(-x_max,x_max+1)
	var = sigma**2
	num = x * x
	den = 2 * var
	g   = np.exp(-num/den) / (np.sqrt(2*np.pi)*sigma)
	if order==1:
		g *= -x/var
	elif order==2:
		g *= (num-var)/var**2
	if normalize:
		# return g / np.linalg.norm(g,1)
		return g / g.sum()
	else:
		return g


def gauss2d(sigma=(1,1), order=(0,0), angle=0, nstd=3, normalize=True):
	"""
	Derivative of the rotated 2d Gaussian filter
	"""
	assert (sigma[0]>0)&(sigma[1]>0), "sigma cannot be equal to zero"
	# if angle==None:
	# 	g = np.outer( gauss1d(size[1],sigma[1],order[1]), gauss1d(size[0],sigma[0],order[0]) )
	# else:
	x,y = generate_filter_support(sigma,angle,nstd)
	x_theta,y_theta = rotate(x,y,-angle)
	g = gauss1d(x=x_theta,sigma=sigma[0],order=order[0],normalize=False) * gauss1d(x=y_theta,sigma=sigma[1],order=order[1],normalize=False)
	if normalize:
		# return g / np.linalg.norm(g,1)
		return g / g.sum()
	else:
		return g


def LoG(sigma=(1,1), angle=0, nstd=3, normalize=True):
	"""
	Laplacian of Gaussian filter
	"""
	assert (sigma[0]>0)&(sigma[1]>0), "sigma cannot be equal to zero"
	x,y = generate_filter_support(sigma,angle,nstd)
	x_theta,y_theta = rotate(x,y,-angle)
	Lambda = sigma[0] / sigma[1]
	g = ( x_theta*x_theta + Lambda**2 * y_theta*y_theta - sigma[0]**2*(1+Lambda**2) ) / sigma[0]**4
	g *= gauss2d(sigma,order=(0,0),angle=angle,nstd=nstd)
	g[x.shape[0]//2,x.shape[1]//2] -= np.sum(g)
	# return [ [[0,0,1,0,0],[0,1,2,1,0],[1,2,-16,2,1],[0,1,2,1,0],[0,0,1,0,0]] ]
	if normalize:
		return [g / np.linalg.norm(g,1)]
	else:
		return [g]


def test(arr):
	arr[0] = 101