load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "mask_resize.so",
    srcs = ["mask_resize_op.cc"],
    gpu_srcs = ["mask_resize_op_gpu.cu.cc", "mask_resize_op_gpu.h"],
)
