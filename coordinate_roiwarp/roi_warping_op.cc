/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("RoiWarp")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .Output("argmax_h: int32")
    .Output("argmax_w: int32");

REGISTER_OP("RoiWarpGrad")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("argmax_h: int32")
    .Input("argmax_w: int32")
    .Input("grad: T")
    .Output("output: T")
    .Output("buffer_data: T");
    
template <typename Device, typename T>
class RoiWarpOp : public OpKernel {
 public:
  explicit RoiWarpOp(OpKernelConstruction* context) : OpKernel(context) {}
   void Compute(OpKernelContext* context) override {}
  private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

bool ROIWarpForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data_h, int* argmax_data_w, const Eigen::GpuDevice& d) ;

static void RoiWarpingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois,
    const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  Tensor* argmax_h = nullptr;
  Tensor* argmax_w = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &argmax_h));
  OP_REQUIRES_OK(context, context->allocate_output(2, tensor_output_shape, &argmax_w));

  if (!context->status().ok()) {
    return;
  }

  ROIWarpForwardLauncher(
    bottom_data->flat<float>().data(), spatial_scale, num_rois, channels,
    height, width, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax_h->flat<int>().data(), argmax_w->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class RoiWarpOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit RoiWarpOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(2);
    // data width
    int data_width = bottom_data.dim_size(3);
    // Number of channels
    int num_channels = bottom_data.dim_size(1);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = num_channels;
    dims[2] = pooled_height_;
    dims[3] = pooled_width_;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    RoiWarpingKernel(context, &bottom_data, &bottom_rois, spatial_scale_, num_rois, num_channels, 
                     data_height, data_width, pooled_height_, pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiWarp").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiWarpOp<Eigen::GpuDevice, float>);

bool ROIWarpBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const float* bottom_rois, const float* bottom_data,
    float* bottom_rois_diff, float* buffer_data, const int* argmax_data_h, const int * argmax_data_w, const Eigen::GpuDevice& d);

static void RoiWarpingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois, const Tensor* argmax_data_h, const Tensor* argmax_data_w, const Tensor* out_backprop,
    const float spatial_scale, const int batch_size, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape, const TensorShape& buffer_output_shape)  
{
  Tensor* output = nullptr;
  Tensor* buffer_data = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, buffer_output_shape, &buffer_data));

  if (!context->status().ok()) {
    return;
  }
  
  ROIWarpBackwardLauncher(
    out_backprop->flat<float>().data(), spatial_scale, batch_size, num_rois, channels,
    height, width, pooled_height, pooled_width, bottom_rois->flat<float>().data(), bottom_data->flat<float>().data(),
    output->flat<float>().data(), buffer_data->flat<float>().data(), argmax_data_h->flat<int>().data(), 
    argmax_data_w->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class Device, class T>
class RoiWarpGradOp : public OpKernel {
 public:
  explicit RoiWarpGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& argmax_data_h = context->input(2);
    const Tensor& argmax_data_w = context->input(3);
    const Tensor& out_backprop = context->input(4);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data_h.dims() == 4,
                errors::InvalidArgument("argmax_data must be 4-dimensional"));
    
    OP_REQUIRES(context, argmax_data_w.dims() == 4,
                errors::InvalidArgument("argmax_data must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int height = bottom_data.dim_size(2);
    // data width
    int width = bottom_data.dim_size(3);
    // Number of channels
    int channels = bottom_data.dim_size(1);

    // construct the output shape
    TensorShape output_shape = bottom_rois.shape();
    TensorShape buffer_shape;
    buffer_shape.AddDim(num_rois*5);
    buffer_shape.AddDim(channels);
    buffer_shape.AddDim(pooled_width_);
    buffer_shape.AddDim(pooled_height_);

    RoiWarpingGradKernel(
      context, &bottom_data, &bottom_rois, &argmax_data_h, &argmax_data_w, &out_backprop,
      spatial_scale_, batch_size, num_rois, channels, height, width, pooled_height_,
      pooled_width_, output_shape, buffer_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiWarpGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiWarpGradOp<Eigen::GpuDevice, float>);
