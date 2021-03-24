
## Install

The project is based on [Maskrcnn Benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [Gliding Vertex](https://github.com/MingtaoFu/gliding_vertex).

### Requirements
```
Python: 3.6.8
PyTorch: 1.2.0.
CUDA: 10.1
CUDNN: 7.6
```

### MaskRCNN benchmark and coco api dependencies
```
pip install ninja yacs cython matplotlib tqdm opencv-python
export INSTALL_DIR=$PWD
```

### Install pycocotools
```
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

### Install apex
```
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

### Install PyTorch Detection
```
python setup.py build develop
```

### Install DOTA_devkit
```shell
REPO_ROOT/maskrcnn_benchmark/DOTA_devkit$ sudo apt-get install swig
REPO_ROOT/maskrcnn_benchmark/DOTA_devkit$ swig -c++ -python polyiou.i
REPO_ROOT/maskrcnn_benchmark/DOTA_devkit$ python setup.py build_ext --inplace
```

### Compile the `poly_nms`:
```shell
REPO_ROOT/maskrcnn_benchmark/utils/poly_nms$ python setup.py build_ext --inplace
```

Don't forget to add `maskrcnn_benchmark` into `$PYTHONPATH`:
```shell
REPO_ROOT/maskrcnn_benchmark$ export PYTHONPATH=$PYTHONPATH:`pwd`
```


## Run

Please edit the file `maskrcnn_benchmark/config/paths_catalog.py` to set the datasets.

Train:
```shell
REPO_ROOT$ python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py --config-file $PATH_TO_CONFIG
```

Test:
```shell
REPO_ROOT$ python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/test_net.py --config-file $PATH_TO_CONFIG --ckpt=$PATH_TO_CKPT
```

### If you want to train with your own data
This project use the json annotation file with COCO format.
Make your directory layout like this:
```
.
└── trainset
    ├── images
    │   ├── 1.png
    │   └── 2.png
    └── labelTxt
        ├── 1.txt
        └── 2.txt
```
A example of the \*.txt files ('1' means the object is difficult):
```
x1 y1 x2 y2 x3 y3 x4 y4 plane 0
x1 y1 x2 y2 x3 y3 x4 y4 harbor 1
```
Run the following Python snippet, and it will generate the json annotation file:
```python
from txt2json import collect_unaug_dataset, convert
img_dic = collect_unaug_dataset( os.path.join( "trainset", "labelTxt" ) )
convert( img_dic, "trainset",  os.path.join( "trainset", "train.json" ) )
```

### If you want to reproduce the results on DOTA

Config: `configs/glide/dota_keypoint.yaml`

#### 1. Prepare the data

Edit the `config.json` and run:
```shell
REPO_ROOT$ python prepare_keypoint.py
```

#### 2. Train
```shell
REPO_ROOT$ python -m torch.distributed.launch --nproc_per_node=3 tools/train_net.py --config-file configs/glide/dota.yaml
```

#### 3. Test
```shell
REPO_ROOT$ python -m torch.distributed.launch --nproc_per_node=3 tools/test_net.py --config-file configs/glide/dota.yaml
# Edit ResultMerge.py and run it.
# srcpath = "REPO_ROOT/exp_dota/dota/inference/dota_test_cut/results"
REPO_ROOT/maskrcnn_benchmark/DOTA_devkit$ python ResultMerge.py
```


