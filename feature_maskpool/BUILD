load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "mask_pooling_feature.so",
    srcs = ["mask_pooling_op.cc"],
    gpu_srcs = ["mask_pooling_op_gpu.cu.cc", "mask_pooling_op_gpu.h"],
)
