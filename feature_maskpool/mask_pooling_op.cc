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

REGISTER_OP("MaskPool")
    .Attr("T: {float, double}")
    .Attr("height: int")
    .Attr("width: int")
    .Input("bottom_data: T")
    .Input("bottom_masks: T")
    .Output("top_data: T");

REGISTER_OP("MaskPoolGrad")
    .Attr("T: {float, double}")
    .Attr("height: int")
    .Attr("width: int")
    .Input("bottom_data: T")
    .Input("bottom_masks: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class MaskPoolOp : public OpKernel {
 public:
  explicit MaskPoolOp(OpKernelConstruction* context) : OpKernel(context) {}
   void Compute(OpKernelContext* context) override {}
  private:
  int height_;
  int width_;
};

//REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolOp<CPUDevice, float>);
//#REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_CPU).TypeConstraint<double>("T"), RoiPoolOp<CPUDevice, double>);

bool MaskPoolForwardLauncher(const float* bottom_data, const float* bottom_masks, const int num_masks, const int channels, 
                             const int height, const int width, float* top_data, const Eigen::GpuDevice& d) ;

static void MaskPoolingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_masks,
    const int num_masks, const int channels, const int height, const int width, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  
  if (!context->status().ok()) {
    return;
  }

  MaskPoolForwardLauncher(
    bottom_data->flat<float>().data(), bottom_masks->flat<float>().data(), num_masks, channels,
    height, width, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class MaskPoolOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit MaskPoolOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("height", &height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, height_ >= 0,
                errors::InvalidArgument("Need height >= 0, got ",
                                        height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("width", &width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, width_ >= 0,
                errors::InvalidArgument("Need width >= 0, got ",
                                        width_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_masks = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_masks.dims() == 4,
                errors::InvalidArgument("masks must be 4-dimensional"));

    // Number of masks
    int num_masks = bottom_masks.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    //int data_height = bottom_data.dim_size(2);
    // data width
    //int data_width = bottom_data.dim_size(3);
    // Number of channels
    int num_channels = bottom_data.dim_size(1);

    // construct the output shape
    int dims[4];
    dims[0] = num_masks;
    dims[1] = num_channels;
    dims[2] = height_;
    dims[3] = width_;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    MaskPoolingKernel(context, &bottom_data, &bottom_masks, num_masks, num_channels, 
                     height_, width_, output_shape);

  }
 private:
  int height_;
  int width_;
};

REGISTER_KERNEL_BUILDER(Name("MaskPool").Device(DEVICE_GPU).TypeConstraint<float>("T"), MaskPoolOp<Eigen::GpuDevice, float>);

bool MaskPoolBackwardLauncher(const float* bottom_data, const float* bottom_masks, float* bottom_diff, 
                              const float* top_diff, const int num_masks, const int channels, const int height, 
                              const int width, const Eigen::GpuDevice& d) ;

static void MaskPoolingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_masks, const Tensor *out_backprop,
    const int num_masks, const int channels, const int height, const int width, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  MaskPoolBackwardLauncher(
    bottom_data->flat<float>().data(), bottom_masks->flat<float>().data(), output->flat<float>().data(), 
    out_backprop->flat<float>().data(), num_masks, channels, height, width, context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class MaskPoolGradOp : public OpKernel {
 public:
  explicit MaskPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("height", &height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, height_ >= 0,
                errors::InvalidArgument("Need height >= 0, got ",
                                        height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("width", &width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, width_ >= 0,
                errors::InvalidArgument("Need width >= 0, got ",
                                        width_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_masks = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_masks.dims() == 4,
                errors::InvalidArgument("masks must be 2-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    int num_masks = bottom_masks.dim_size(0);
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

    MaskPoolingGradKernel(
      context, &bottom_data, &bottom_masks, &out_backprop,
      num_masks, channels, height, width, output_shape);

  }
 private:
  int height_;
  int width_;
};

REGISTER_KERNEL_BUILDER(Name("MaskPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), MaskPoolGradOp<Eigen::GpuDevice, float>);