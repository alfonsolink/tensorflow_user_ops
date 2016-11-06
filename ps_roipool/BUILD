load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "psroi_pooling.so",
    srcs = ["psroi_pooling_op.cc"],
    gpu_srcs = ["psroi_pooling_op_gpu.cu.cc", "psroi_pooling_op_gpu.h"],
)
