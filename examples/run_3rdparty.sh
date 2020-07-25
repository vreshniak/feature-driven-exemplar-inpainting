#!/bin/bash

# directory of the script
DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"


if [ $# -eq 0 ]; then
	venv_DIR="$HOME"/3rdparty/venvs
	code_DIR="$HOME"/3rdparty/src
	# cd ..
	# venv_DIR="$(pwd)/3rdparty/venvs"
	# code_DIR="$(pwd)/3rdparty/src"
	# cd "$DIR"
else
	venv_DIR="$1"/venvs
	code_DIR="$1"/src
fi

mkdir -p "$venv_DIR"
mkdir -p "$code_DIR"


for ex in 1 2 3 4; do
	dataDIR="$DIR"/example_"$ex"/data
	outDIR="$DIR"/example_"$ex"/output/3rdparty


	#########################################################
	# "Variational Framework for Non-Local Inpainting" by Vadim Fedorov, Gabriele Facciolo, Pablo Arias

	if [ ! -d "$code_DIR/inpaint_9" ]; then
		cd "$code_DIR"
		wget https://www.ipol.im/pub/art/2015/136/inpaint_8.tgz
		tar -xzvf inpaint_8.tgz
		cd inpaint_9
		# sed -i 's/#SET( EXTRA_COMPILER_FLAGS "${EXTRA_COMPILER_FLAGS} -DDBG_OUTPUT" )/SET( EXTRA_COMPILER_FLAGS "${EXTRA_COMPILER_FLAGS} -DDBG_OUTPUT" )/' ./CMakeLists.txt
		cmake .
		make
	fi

	cd "$code_DIR"/inpaint_9
	mkdir -p "$outDIR"
	for patch in 9 15; do
		./Inpainting "$dataDIR"/image.png "$dataDIR"/mask.png ./fedorov_means_p15.png -method nlmeans   -patch "$patch" -psigma 5 -conft 10 -scales 5 -coarse 0.3
		mv *.png "$outDIR"/fedorov_means_p"$patch".png
		./Inpainting "$dataDIR"/image.png "$dataDIR"/mask.png ./fedorov_means_p15.png -method nlpoisson -patch "$patch" -psigma 5 -conft 10 -scales 5 -coarse 0.3 -lambda 0.1
		mv *.png "$outDIR"/fedorov_poisson_p"$patch"_lmd01.png
	done

	cd "$DIR"


	#########################################################
	# "Non-Local Patch-Based Image Inpainting" by Alasdair Newson, Andrés Almansa, Yann Gousseau, Patrick Pérez

	if [ ! -d "$venv_DIR/newson_ipol" ]; then
		cd "$code_DIR"
		wget https://www.ipol.im/pub/art/2017/189/Inpainting_ipol_code.tar.gz
		tar -xzvf Inpainting_ipol_code.tar.gz
		cd Inpainting_ipol_code
		make
	fi

	cd "$code_DIR"/Inpainting_ipol_code
	mkdir -p "$outDIR"
	bin/inpaint_image "$dataDIR"/image.png "$dataDIR"/mask.png "$outDIR"/newson.png -patchSizeX 9 -patchSizeY 9 -nLevels -1 -seFeatures 1

	cd "$DIR"


	#########################################################
	# "Region Filling and Object Removal by Exemplar-Based Image Inpainting" by A. Criminisi et al.

	if [ ! -d "$venv_DIR/inpaint-object-remover" ]; then
		python3 -m venv "$venv_DIR"/inpaint-object-remover
		source "$venv_DIR"/inpaint-object-remover/bin/activate

		cd "$code_DIR"
		git clone https://github.com/igorcmoura/inpaint-object-remover.git
		cd inpaint-object-remover
		sed -i "s/mask = imread(args.mask, as_grey=True)/mask = imread(args.mask, as_gray=True).astype('bool')/" ./inpainter/__main__.py
		python3 -m pip install -r requirements.txt
	else
		source "$venv_DIR"/inpaint-object-remover/bin/activate
	fi

	cd "$code_DIR"/inpaint-object-remover
	mkdir -p "$outDIR"
	# greyscale images are not accepted
	convert "$dataDIR"/image.png PNG24:"$dataDIR"/image1.png
	python3 -W ignore inpainter "$dataDIR"/image1.png "$dataDIR"/mask.png --output "$outDIR"/crim_p9 --patch-size 9
	rm "$dataDIR"/image1.png

	deactivate
	cd "$DIR"


	#########################################################
	# Edge-Connect

	if [ ! -d "$venv_DIR/edge-connect" ]; then
		python3 -m venv "$venv_DIR"/edge-connect
		source "$venv_DIR"/edge-connect/bin/activate

		cd "$code_DIR"
		git clone https://github.com/knazeri/edge-connect.git
		cd edge-connect
		# to ensure correct behavior for given edges
		# cp "$DIR"/edge_connect_replacement.py ./src/edge_connect.py
		sed -i "s/edges = self.edge_model(images_gray, edges, masks).detach()/# edges = self.edge_model(images_gray, edges, masks).detach()/" ./src/edge_connect.py
		python3 -m pip install -r requirements.txt
		python3 -m pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

		# dowload pretrained models
		bash scripts/download_model.sh
	else
		source "$venv_DIR"/edge-connect/bin/activate
	fi

	cd "$code_DIR"/edge-connect
	# outDIR="$DIR"/example_"$ex"/output/3rdparty/edge-connect
	for dataset in places2; do #psv celeba
		sed 's/NMS: 1/NMS: 0/' ./config.yml.example > ./checkpoints/"$dataset"/config.yml
		python3 -W ignore test.py --model 3 --checkpoints ./checkpoints/"$dataset" --input "$dataDIR"/image.png --mask "$dataDIR"/mask.png --output "$outDIR"
		mv "$outDIR"/image.png "$outDIR"/ec_"$dataset".png

		sed 's/EDGE: 1/EDGE: 2/; s/NMS: 1/NMS: 0/' ./config.yml.example > ./checkpoints/"$dataset"/config.yml
		for edge_file in "$dataDIR"/*edges*; do
			# if edge file(s) exist
			if [ -f "$edge_file" ]; then
				python3 -W ignore test.py --model 3 --checkpoints ./checkpoints/"$dataset" --input "$dataDIR"/image.png --mask "$dataDIR"/mask.png --edge "$edge_file" --output "$outDIR"
				mv "$outDIR"/image.png "$outDIR"/ec_"$dataset"_"$(basename "$edge_file")"
			fi
		done
	done
	deactivate
	cd "$DIR"


	#########################################################
	# Globally and Locally Consistent Image Completion

	if [ ! -d "$venv_DIR/siggraph2017" ]; then
		python3 -m venv "$venv_DIR"/siggraph2017
		source "$venv_DIR"/siggraph2017/bin/activate

		cd $code_DIR
		git clone https://github.com/akmtn/pytorch-siggraph2017-inpainting.git
		cd pytorch-siggraph2017-inpainting
		python3 -m pip install numpy===1.15.0 scipy opencv-python pyamg torch===0.4.1.post2 torchvision===0.2.1

		# dowload pretrained models
		wget --continue -O completionnet_places2.t7 http://hi.cs.waseda.ac.jp/~iizuka/data/completionnet_places2.t7
	else
		source "$venv_DIR"/siggraph2017/bin/activate
	fi

	cd "$code_DIR"/pytorch-siggraph2017-inpainting
	mkdir -p "$outDIR"

	python3 -W ignore inpaint.py --input "$dataDIR"/image.png --mask "$dataDIR"/mask.png
	mv out.png "$outDIR"/siggraph_places2.png
	deactivate
	cd "$DIR"


	#########################################################
	# Generative Image Inpainting with contextual attention

	if [ ! -d "$venv_DIR/contextual_attention" ]; then
		python3 -m venv "$venv_DIR"/contextual_attention
		source "$venv_DIR"/contextual_attention/bin/activate

		cd $code_DIR
		git clone https://github.com/JiahuiYu/generative_inpainting.git
		cd generative_inpainting
		pip install gdown opencv-python tensorflow===1.13.0rc2 git+https://github.com/JiahuiYu/neuralgym PyYAML pillow

		# dowload pretrained models
		mkdir -p model_logs/places2/
		cd model_logs/places2/
		gdown --id 1dyPD2hx0JTmMuHYa32j-pu--MXqgLgMy
		gdown --id 1z9dbEAzr5lmlCewixevFMTVBmNuSNAgK
		gdown --id 1ExY4hlx0DjVElqJlki57la3Qxu40uhgd
		gdown --id 1C7kPxqrpNpQF7B2UAKd_GvhUMk0prRdV

		mkdir -p ../../model_logs/celeba/
		cd ../../model_logs/celeba/
		gdown --id 1bW1qyxl6KX0M84ZFso4WbS71r3IzAfQN
		gdown --id 1OmEBFDTWsZ5eSt0bexOtXBq4utXtUTQN
		gdown --id 1EMil_P4eVS1YAVoohHXVljvqMKDfP-XI
		gdown --id 1kK9IiJkTxYmtL1w1rjQ_ZnILAZP7gH46
	else
		source "$venv_DIR"/contextual_attention/bin/activate
	fi

	cd "$code_DIR"/generative_inpainting
	mkdir -p "$outDIR"

	python3 -W ignore test.py --image "$dataDIR"/image.png --mask "$dataDIR"/mask.png --output "$outDIR"/cntx_att_places2.png --checkpoint_dir "$code_DIR"/generative_inpainting/model_logs/places2
	# python3 -W ignore test.py --image "$dataDIR"/image.png --mask "$dataDIR"/mask.png --output "$outDIR"/cntx_att_celeba.png  --checkpoint_dir "$code_DIR"/generative_inpainting/model_logs/celeba
	deactivate
	cd "$DIR"


	#########################################################
	# Generative Multi-column Convolutional Neural Networks inpainting

	if [ ! -d "$venv_DIR/inpainting_gmcnn" ]; then
		python3 -m venv "$venv_DIR"/inpainting_gmcnn
		source "$venv_DIR"/inpainting_gmcnn/bin/activate

		cd "$code_DIR"
		git clone https://github.com/shepnerd/inpainting_gmcnn.git
		cd inpainting_gmcnn/tensorflow
		# comment lines
		sed -i "54,58 s/^/#/" test.py
		sed -i "59 i \        mask  = cv2.imread(os.path.join(config.dataset_path, 'mask', 'mask.png'), 0)[:,:,np.newaxis].astype(np.float) / 255." test.py
		sed -i "75 s/^/#/" test.py
		sed -i "88 s/^/#/" test.py
		sed -i "89 i \        cv2.imwrite(os.path.join(config.saving_path, '..', 'image.png'), result[0][:, :, ::-1])" test.py
		pip install gdown numpy scipy easydict opencv-python setuptools===41.0.0 tensorflow===1.15.0

		# dowload pretrained models
		mkdir -p ./checkpoints
		cd ./checkpoints
		rm *.zip
		gdown --id 1wgesxSUfKGyPwGQMw6IXZ9GLeZ7YNQxu && unzip *.zip && rm *.zip && mv paris-streetview_256x256_rect psv
		gdown --id 1aakVS0CPML_Qg-PuXGE1Xaql96hNEKOU && unzip *.zip && rm *.zip && mv places2_512x680_freeform places2
	else
		source "$venv_DIR"/inpainting_gmcnn/bin/activate
	fi

	cd "$code_DIR"/inpainting_gmcnn/tensorflow
	mkdir -p "$outDIR"

	mkdir -p ./imgs/data
	mkdir -p ./imgs/data/mask

	for dataset in places2; do # psv
		cp "$dataDIR"/image.png ./imgs/data/image.png
		cp "$dataDIR"/mask.png  ./imgs/data/mask/mask.png
		python3 -W ignore test.py --dataset paris_streetview --data_file ./imgs/data/ --load_model_dir ./checkpoints/"$dataset" --img_shapes "$( identify -format %h,%w,3 ./imgs/data/image.png )"
		mv ./test_results/image.png "$outDIR"/gmcnn_"$dataset".png
		rm -r ./test_results/*
	done
	rm -r ./imgs/data
	deactivate
	cd "$DIR"
done