#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


at::Tensor
deform_conv_cpu_forward(const at::Tensor &input,
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
                   const int deformable_group)
{
    AT_ERROR("Not implement on cpu");
}

std::vector<at::Tensor>
deform_conv_cpu_backward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const at::Tensor &grad_output,
                    int kernel_h, int kernel_w,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dilation_h, int dilation_w,
                    int deformable_group)
{
    AT_ERROR("Not implement on cpu");
}

