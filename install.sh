#!/bin/bash

python -m pip install git+https://github.com/vreshniak/feature-driven-exemplar-inpainting.git
exit


# compilers
# gcc=gcc-8
gcc=gcc
python=python

# directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$DIR/src
if [ $# -eq 0 ]; then
	INSTALL_DIR=$DIR/lib
else
	INSTALL_DIR=$1
fi


# compile shared "libpatchmatch.so" library from c_patchmatch.c source file
cd "$SRC_DIR"/patchmatch
$gcc -shared -fPIC -fopenmp -O3 -o ../libpatchmatch.so c_patchmatch.c c_patchmatch_64.c
cd "$DIR"

# note implicit rule LDFLAGS for library location
LDFLAGS="-L'$SRC_DIR'" $python setup.py build_ext #--inplace
LDFLAGS="-L'$SRC_DIR'" $python setup.py install --install-lib="$INSTALL_DIR" #--prefix="$SRC_DIR" --user
# LDFLAGS="-L'$SRC_DIR'" $python "$SRC_DIR"/setup.py install --install-lib="$SRC_DIR/../.." --user
$python setup.py clean
rm "$SRC_DIR"/patchmatch/patchmatch.c
rm -r "$DIR"/build #>/dev/null 2>&1

mv "$SRC_DIR"/libpatchmatch.so "$INSTALL_DIR"/libpatchmatch.so

# add library paths to .bashrc
if grep -Fxq "# >>> patchmatch init >>>" ~/.bashrc; then
	# replace old paths to the library
	start='# >>> patchmatch init >>>'
	end='# <<< patchmatch init <<<'
	sed -i -n "/$start/{p;:a;N;/$end/!ba; s|.*\n|export PYTHONPATH='$INSTALL_DIR':$PYTHONPATH\nexport LD_LIBRARY_PATH='$INSTALL_DIR':$LD_LIBRARY_PATH\n|}; p" ~/.bashrc
	export PYTHONPATH="$INSTALL_DIR":$PYTHONPATH
	export LD_LIBRARY_PATH="$INSTALL_DIR":$LD_LIBRARY_PATH
else
/bin/cat <<EOM >>~/.bashrc
# >>> patchmatch init >>>
export PYTHONPATH='$INSTALL_DIR':$PYTHONPATH
export LD_LIBRARY_PATH='$INSTALL_DIR':$LD_LIBRARY_PATH
# <<< patchmatch init <<<
EOM
fi