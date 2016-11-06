#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_FEATUREMASKPOOLING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_FEATUREMASKPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool MaskPoolForwardLauncher(const float* bottom_data, const float* bottom_masks, const int num_masks, const int channels, 
                             const int height, const int width, float* top_data, const Eigen::GpuDevice& d) ;

bool MaskPoolBackwardLauncher(const float* bottom_data, const float* bottom_masks, float* bottom_diff, float* bottom_mask_diff,
                              const float* top_diff, const int num_masks, const int channels, const int height, 
                              const int width, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_