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

REGISTER_OP("RoiPool")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .Output("argmax: int32");

REGISTER_OP("RoiPoolGrad")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("argmax: int32")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class RoiPoolOp : public OpKernel {
 public:
  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {}
   void Compute(OpKernelContext* context) override {}
  private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

//REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolOp<CPUDevice, float>);
//#REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_CPU).TypeConstraint<double>("T"), RoiPoolOp<CPUDevice, double>);

bool ROIPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data, const Eigen::GpuDevice& d);

static void RoiPoolingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois,
    const float spatial_scale, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  Tensor* argmax = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &argmax));

  if (!context->status().ok()) {
    return;
  }

  ROIPoolForwardLauncher(
    bottom_data->flat<float>().data(), spatial_scale, num_rois, channels,
    height, width, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class RoiPoolOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {

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

    RoiPoolingKernel(context, &bottom_data, &bottom_rois, spatial_scale_, num_rois, num_channels, 
                     data_height, data_width, pooled_height_, pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiPoolOp<Eigen::GpuDevice, float>);


bool ROIPoolBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int channels, const int height, const int width, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const int* argmax_data, const Eigen::GpuDevice& d);

static void RoiPoolingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois, const Tensor* argmax_data, const Tensor* out_backprop,
    const float spatial_scale, const int batch_size, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  ROIPoolBackwardLauncher(
    out_backprop->flat<float>().data(), spatial_scale, batch_size, num_rois, channels,
    height, width, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax_data->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class RoiPoolGradOp : public OpKernel {
 public:
  explicit RoiPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {

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
    const Tensor& argmax_data = context->input(2);
    const Tensor& out_backprop = context->input(3);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data.dims() == 4,
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
    TensorShape output_shape = bottom_data.shape();

    RoiPoolingGradKernel(
      context, &bottom_data, &bottom_rois, &argmax_data, &out_backprop,
      spatial_scale_, batch_size, num_rois, channels, height, width, pooled_height_,
      pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiPoolGradOp<Eigen::GpuDevice, float>);