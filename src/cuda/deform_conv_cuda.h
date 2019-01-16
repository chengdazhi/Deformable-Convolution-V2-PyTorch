#pragma once
#include <torch/extension.h>

at::Tensor
deform_conv_cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int group,
                    const int deformable_group, 
                    const int im2col_step);

std::vector<at::Tensor>
deform_conv_cuda_backward(const at::Tensor &input,
                     const at::Tensor &weight,
                     const at::Tensor &bias,
                     const at::Tensor &offset,
                     const at::Tensor &grad_output,
                     const int kernel_h, 
                     const int kernel_w,
                     const int stride_h, 
                     const int stride_w,
                     const int pad_h, 
                     const int pad_w,
                     const int dilation_h, 
                     const int dilation_w,
                     const int group,
                     const int deformable_group, 
                     const int im2col_step);

