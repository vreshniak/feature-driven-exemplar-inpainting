# setup.py script to build and install patchmatch extension module written in C

from distutils.core      import setup
from distutils.extension import Extension
from Cython.Build        import cythonize


# instantiate Extension class for the Cython project
ext_modules = Extension(name="patchmatch",
						sources=["src/patchmatch/patchmatch.pyx"],	# Cython source file
						libraries=["patchmatch"]					# pre-compiled libpatchmatch.so to link with patchmatch.pyx
						)

# Note: setup() has access to cmd arguments of the setup.py script via sys.argv
setup(name="pm",
	ext_modules=cythonize( module_list=[ext_modules], force=True, annotate=False, compiler_directives={'language_level' : "3"} ),
	# py_modules=["inpainting", "features", "utils"],
	version='0.1.0',
	packages=["imgproc"],
	package_dir={'imgproc': 'src'},
	author='Viktor Reshniak',
	author_email='reshniakv@ornl.gov'
	)