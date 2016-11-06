load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "roi_pooling_girshick.so",
    srcs = ["roi_pooling_op.cc"],
    gpu_srcs = ["roi_pooling_op_gpu.cu.cc", "roi_pooling_op_gpu.h"],
)
