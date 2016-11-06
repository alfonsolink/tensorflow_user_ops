load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "roi_warping_coordinate.so",
    srcs = ["roi_warping_op.cc"],
    gpu_srcs = ["roi_warping_op_gpu.cu.cc", "roi_warping_op_gpu.h"],
)
