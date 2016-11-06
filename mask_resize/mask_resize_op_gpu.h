#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_MASKRESIZE_OP_GPU_H_
#define TENSORFLOW_USER_OPS_MASKRESIZE_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool MaskResizeForwardLauncher(
    const float* bottom_data,  const int output_width, const int output_height, const int output_channels, const int num_masks,
    const int input_width, const int input_height, const int input_channels, float* top_data,
    const Eigen::GpuDevice& d);

bool MaskResizeBackwardLauncher(const float* top_diff, const int output_width, const int output_height, const int output_channels, const int num_masks,
                               const int input_width, const int input_height, const int input_channels, float* bottom_diff, const Eigen::GpuDevice& d) ;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_