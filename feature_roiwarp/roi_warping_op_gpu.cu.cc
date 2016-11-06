#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "roi_warping_op_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;

using namespace tensorflow;

template <typename Dtype>
__device__ void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, Dtype w, Dtype & maxval, Dtype & maxidx_h, Dtype & maxidx_w) {
  
  // deal with cases that inverse elements are out of feature map boundary
  if (h < -0.5 || h > height - 0.5 || w < -0.5 || w > width - 0.5) {
    //empty
    return;
  }
  
  if (h <= 0) h = 0;
  if (w <= 0) w = 0;
  
  int h_low = (int) h;
  int w_low = (int) w;
  int h_high;
  int w_high;
  
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }
  
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }
  
  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;
  // do bilinear interpolation
  Dtype v1 = bottom_data[h_low * width + w_low];
  Dtype v2 = bottom_data[h_low * width + w_high];
  Dtype v3 = bottom_data[h_high * width + w_low];
  Dtype v4 = bottom_data[h_high * width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  
  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  if (val > maxval) {
    maxval = val;
    maxidx_h = h;
    maxidx_w = w;
  }
}

template <typename Dtype>
__global__ void ROIWarpForward(const int nthreads, const Dtype* bottom_data,
			       const Dtype spatial_scale, const int channels, const int height, const int width,
			       const int pooled_height, const int pooled_width, const Dtype* bottom_rois,
			       Dtype* top_data, int* argmax_data_h, int* argmax_data_w) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    
    bottom_rois += n * 5;
    int roi_level = bottom_rois[0];
    Dtype roi_start_w = round(bottom_rois[1] * spatial_scale);
    Dtype roi_start_h = round(bottom_rois[2] * spatial_scale);
    Dtype roi_end_w = round(bottom_rois[3] * spatial_scale);
    Dtype roi_end_h = round(bottom_rois[4] * spatial_scale);
    
    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, (Dtype)0.);
    Dtype roi_height = max(roi_end_h - roi_start_h, (Dtype)0.);
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);
    
    // Define an empty pooling region to be zero
    Dtype maxval = -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backpropgated
    Dtype maxidx_h = -1;
    Dtype maxidx_w = -1;
    bottom_data += (roi_level * channels + c) * height * width;
    
    // in the forward part, since typically ROI size is larger than pooled output
    // so we still do take the max value over a set of bilinear interpolated values
    int roi_pool_ratio_h = ceil(roi_height / pooled_height);
    int roi_pool_ratio_w = ceil(roi_width / pooled_width);
    Dtype step_size_h = 1.0 / static_cast<Dtype>(roi_pool_ratio_h);
    Dtype step_size_w = 1.0 / static_cast<Dtype>(roi_pool_ratio_w);

    for (Dtype fph = static_cast<Dtype>(ph); fph <= ph + 1; fph += step_size_h)
    {
      for (Dtype fpw = static_cast<Dtype>(pw); fpw <= pw + 1; fpw += step_size_w)
      {
        // inverse index in the feature map
        Dtype ih = roi_start_h + static_cast<Dtype>(fph) * bin_size_h;
        Dtype iw = roi_start_w + static_cast<Dtype>(fpw) * bin_size_w;
        bilinear_interpolate(bottom_data, height, width, ih, iw, maxval, maxidx_h, maxidx_w);
      }
    }
    
    if (maxidx_h == -1 && maxidx_w == -1) maxval = 0;
    top_data[index] = maxval;
    argmax_data_h[index] = maxidx_h;
    argmax_data_w[index] = maxidx_w;
  }
}

bool ROIWarpForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data_h, int* argmax_data_w, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  cudaError_t err;

  ROIWarpForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, spatial_scale, channels, height, width, pooled_height,
      pooled_width, bottom_rois, top_data, argmax_data_h, argmax_data_w);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

template <typename Dtype>
__device__ Dtype get_feature_gradient(Dtype argmax_h, Dtype argmax_w, const int h, const int w, const int height, const int width)
{
  if (argmax_h < -0.5 || argmax_h >(height - 0.5) || argmax_w < -0.5 || argmax_w >(width - 0.5))
    {
      //empty
      return 0;
    }
  
  if (argmax_h < 0) argmax_h = 0;
  if (argmax_w < 0) argmax_w = 0;
  
  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (Dtype)argmax_h_low;
  }
  else
    argmax_h_high = argmax_h_low + 1;
  
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (Dtype)argmax_w_low;
  }
  else
    argmax_w_high = argmax_w_low + 1;
  
  Dtype weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  }
  else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}
  
template <typename Dtype>
__global__ void ROIWarpBackwardFeature(const int nthreads, const Dtype* top_diff,
				const int* argmax_data_h, const int* argmax_data_w, const int num_rois, const Dtype spatial_scale, const int channels,
				const int height, const int width, const int pooled_height,
				const int pooled_width, Dtype* bottom_diff, const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_level = offset_bottom_rois[0];
      // Skip if ROI's level doesn't match n
      if (n != roi_level) {
	continue;
      }
      
      Dtype roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      Dtype roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      Dtype roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      Dtype roi_end_h = round(offset_bottom_rois[4] * spatial_scale);
      
      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= floor(roi_start_w) && w <= ceil(roi_end_w) &&
			   h >= floor(roi_start_h) && h <= ceil(roi_end_h));
      if (!in_roi) {
	continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data_h = argmax_data_h + offset;
      const int* offset_argmax_data_w = argmax_data_w + offset;
      
      // Compute feasible set of pooled units that could have pooled
      // this bottom unit
      
      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w+(Dtype)1.0, (Dtype)1.0);
      Dtype roi_height = max(roi_end_h - roi_start_h+(Dtype)1.0, (Dtype)1.0);
      Dtype bin_size_h = static_cast<Dtype>(roi_height)
	/ static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
	/ static_cast<Dtype>(pooled_width);
      
      int phstart = floor(static_cast<Dtype>(h - roi_start_h - 1) / bin_size_h - 1);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w - 1) / bin_size_w - 1);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);
      
      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);
      
      for (int ph = phstart; ph < phend; ++ph) {
	for (int pw = pwstart; pw < pwend; ++pw) {
	  Dtype weight = get_feature_gradient(offset_argmax_data_h[ph * pooled_width + pw],
					   offset_argmax_data_w[ph * pooled_width + pw], h, w, height, width);
	  gradient += weight * offset_top_diff[ph * pooled_width + pw];
	}
      }
    }
    bottom_diff[index] = gradient;
  }
}

bool ROIWarpBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int channels, const int height, const int width, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const int* argmax_data_h, const int * argmax_data_w, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  const int bottom_size = batch_size * height * width * channels;
  cudaError_t err;
  
  SetZero<<<(bottom_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, d.stream()>>>(bottom_size, bottom_diff);

  ROIWarpBackwardFeature<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, argmax_data_h, argmax_data_w, num_rois, spatial_scale, channels, height, width, pooled_height,
      pooled_width, bottom_diff, bottom_rois);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

#endif  // GOOGLE_CUDA
