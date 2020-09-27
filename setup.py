# setup.py script to build and install patchmatch extension module written in C

import platform
plt = platform.system()

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Build        import cythonize


# instantiate Extension class for the Cython project
# https://iscinumpy.gitlab.io/post/omp-on-high-sierra/
ext_modules = Extension(name="patchmatch",
						sources=["src/patchmatch/patchmatch.pyx", "src/patchmatch/c_patchmatch.c", "src/patchmatch/c_patchmatch_64.c"],	# Cython source file
						extra_compile_args=['-Xpreprocessor -fopenmp -lomp'] if plt=='Darwin' else ['-fopenmp'],
						extra_link_args=['-Xpreprocessor -fopenmp -lomp']    if plt=='Darwin' else ['-fopenmp'],
						)

# Note: setup() has access to cmd arguments of the setup.py script via sys.argv
# https://packaging.python.org/tutorials/packaging-projects/
setup(name='inpainting',
	ext_modules=cythonize( module_list=[ext_modules], force=True, annotate=False, compiler_directives={'language_level' : "3"} ),
	version='0.1.0',
	packages=['inpainting'],
	package_dir={'inpainting': 'src'},
	author='Viktor Reshniak',
	author_email='reshniakv@ornl.gov',
	url='https://github.com/vreshniak/feature-driven-exemplar-inpainting'
	)