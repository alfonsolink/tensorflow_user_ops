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

REGISTER_OP("MaskResize")
    .Attr("T: {float, double}")
    .Attr("output_height: int")
    .Attr("output_width: int")
    .Attr("output_channels: int")
    .Input("bottom_data: T")
    .Output("top_data: T");

REGISTER_OP("MaskResizeGrad")
    .Attr("T: {float, double}")
    .Attr("output_height: int")
    .Attr("output_width: int")
    .Attr("output_channels: int")
    .Input("bottom_data: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class MaskResizeOp : public OpKernel {
 public:
  explicit MaskResizeOp(OpKernelConstruction* context) : OpKernel(context) {}
   void Compute(OpKernelContext* context) override {}
  private:
  int output_height_;
  int output_width_;
  int output_channels_;
};


bool MaskResizeForwardLauncher(
    const float* bottom_data,  const int output_width, const int output_height, const int output_channels, const int num_masks,
    const int input_width, const int input_height, const int input_channels, float* top_data,
    const Eigen::GpuDevice& d);

static void MaskResizeKernel(
    OpKernelContext* context, const Tensor* bottom_data, const int output_width, const int output_height, const int output_channels, const int num_masks,
    const int input_width, const int input_height, const int input_channels, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  MaskResizeForwardLauncher(
    bottom_data->flat<float>().data(), output_width, output_height, output_channels, num_masks, 
                           input_width, input_height, input_channels, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class MaskResizeOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit MaskResizeOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the output dim
      
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_width", &output_width_));
    
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_height", &output_height_));
    
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_channels", &output_channels_));
    
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));


    // batch size
    int num_masks = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(2);
    // data width
    int data_width = bottom_data.dim_size(3);
    // Number of channels
    int num_channels = bottom_data.dim_size(1);

    
    // construct the output shape
    int dims[4];
    dims[0] = num_masks;
    dims[1] = output_channels_;
    dims[2] = output_height_;
    dims[3] = output_width_;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    MaskResizeKernel(context, &bottom_data, output_width_, output_height_, output_channels_, num_masks,
                     data_width, data_height, num_channels, output_shape);

  }
 private:

  int output_height_;
  int output_width_;
  int output_channels_;
};

REGISTER_KERNEL_BUILDER(Name("MaskResize").Device(DEVICE_GPU).TypeConstraint<float>("T"), MaskResizeOp<Eigen::GpuDevice, float>);


bool MaskResizeBackwardLauncher(const float* top_diff, const int output_width, const int output_height, const int output_channels, const int num_masks,
                               const int input_width, const int input_height, const int input_channels, float* bottom_diff, const Eigen::GpuDevice& d) ;

static void MaskResizeGradKernel(
    OpKernelContext* context, const Tensor* out_backprop, const int output_width, const int output_height,
    const int output_channels, const int num_masks, const int input_width, const int input_height, const int input_channels, const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  MaskResizeBackwardLauncher(
    out_backprop->flat<float>().data(), output_width, output_height, output_channels, num_masks, input_width, input_height,
                             input_channels, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class MaskResizeGradOp : public OpKernel {
 public:
  explicit MaskResizeGradOp(OpKernelConstruction* context) : OpKernel(context) {

    OP_REQUIRES_OK(context,
                   context->GetAttr("output_width", &output_width_));
    
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_height", &output_height_));
    
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_channels", &output_channels_));
  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& out_backprop = context->input(1);

    // rois should have 2 dimensions.
    
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("bottom data must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));
    
    // batch size
    int num_masks = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(2);
    // data width
    int data_width = bottom_data.dim_size(3);
    // Number of channels
    int num_channels = bottom_data.dim_size(1);

    
    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    MaskResizeGradKernel(
      context, &out_backprop, output_width_, output_height_, output_channels_, num_masks,
      data_width, data_height, num_channels, output_shape);

  }
 private:
  int output_height_;
  int output_width_;
  int output_channels_;
};

REGISTER_KERNEL_BUILDER(Name("MaskResizeGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), MaskResizeGradOp<Eigen::GpuDevice, float>);