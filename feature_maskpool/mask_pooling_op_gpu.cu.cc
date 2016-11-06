#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "mask_pooling_op_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;


using namespace tensorflow;

template <typename Dtype>
__global__ void MaskPoolingForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_masks,
			       Dtype* top_data, const int channels, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the top output
    int pw = index % width;
    int ph = (index / width) % height;
    // int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int mask_index = n * height * width + ph * width + pw;

    // top feature map has identical shape with bottom feature map, so we reuse index here
    top_data[index] = bottom_data[index] * bottom_masks[mask_index];
  }
}

bool MaskPoolForwardLauncher(
    const float* bottom_data, const float* bottom_masks, const int num_masks, const int channels,
    const int height, const int width, float* top_data, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_masks * channels * height * width;
  cudaError_t err;

  MaskPoolingForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, bottom_masks, top_data, channels, height, width);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

template <typename Dtype>
__global__ void MaskPoolingBackwardFeature(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_masks,
        Dtype* bottom_diff, const Dtype* top_diff, const int channels, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    // int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    // output w,h coordinate has the same size with input's w,h coordinate
    int mask_index = n * height * width + h * width + w;
    Dtype float_mask = bottom_masks[mask_index];
    bottom_diff[index] = top_diff[index] * float_mask;
  }
}

template <typename Dtype>
__global__ void MaskPoolingBackwardMask(const int nthreads, const Dtype* bottom_data, Dtype* bottom_diff, 
  const Dtype* top_diff, const int channels, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, w, h) are index of mask element, with channel dim = 1
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height / 1;
    Dtype gradient = 0.0;
    for (int i = 0; i < channels; ++i) {
      int data_index = ((n * channels + i) * height + h) * width + w;
      gradient += top_diff[data_index] * bottom_data[data_index];
    }
    int mask_index = ((n * height) + h) * width + w;
    bottom_diff[mask_index] = gradient;
  }
}


bool MaskPoolBackwardLauncher(const float* bottom_data, const float* bottom_masks, float* bottom_diff, 
                              const float* top_diff, const int num_masks, const int channels, const int height, 
                              const int width, const Eigen::GpuDevice& d) 
{
  
      const int kThreadsPerBlock = 1024;
      const int output_size = num_masks * channels * height * width;
      SetZero<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>(output_size, bottom_diff);

      cudaError_t err;
      MaskPoolingBackwardFeature<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
      (output_size, bottom_data, bottom_masks, bottom_diff, top_diff, channels, height, width);
  
  //else if (propagate_down == 1) {
  //    const int kThreadsPerBlock = 1024;
  //    const int output_size = num_masks * 1 * height * width
  //    cudaError_t err;
  //    MaskPoolingBackwardMask<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>
  //    (output_size, bottom_data, bottom_diff, top_diff, channels, height, width);
  //}

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

#endif  // GOOGLE_CUDA
