# tensorflow_user_ops
tensorflow-wrapped layers from Caffe <br />

available layers (GPU implementation only):

coordinate_roiwarp - roi warping with backprop on bbox coords only <br />
feature_maskpool	- mask pooling with backprop on features only <br />
feature_roiwarp	- roi warping with backprop on features only <br />
girshick_roipool	- roi pooling from base faster rcnn <br />
mask_resize	- mnc mask resizing <br />
ps_roipool	- position sensitive roi pooling for rfcn <br />
roiwarp - mnc roi warping with backprop on both bbox coords and features <br />

original caffe layers for girshick_roipool are from [py-faster-rcnn] (https://github.com/rbgirshick/py-faster-rcnn), by rbgirshick <br />
for roiwarp, and mask pooling / resizing, original caffe layers are from [mnc] (https://github.com/daijifeng001/MNC), by daijifeng001 <br />
ps_roipool is from [py-rfcn] (https://github.com/Orpine/py-R-FCN), by Orpine <br />

created by A. Labao and P. Naval, CVMIG lab, Univ of the Philippines <br />

# installation
to install, copy desired folder (i.e. roiwarp) to /tensorflow/tensorflow/core/user_ops/, and build .so file through bazel <br />
```
  bazel build -c opt --config=cuda //tensorflow/core/user_ops/<folder_name>:<so_name>
```
# gradient tests
the folder user_ops_tests contains python files for testing gradients and registering gradients in tensorflow

# requirements
the layers are tested with the ff specs: <br />
tensorflow v0.10 and up <br />
python 2.7 <br />
python numpy <br />
CuDNN v5 <br />
CUDA v8.0 <br />
