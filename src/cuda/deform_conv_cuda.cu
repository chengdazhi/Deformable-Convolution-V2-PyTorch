#include <vector>
#include "cuda/deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

// extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu


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
                    const int deformable_group)
{
    // THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, bias, offset, mask));

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    // printf("Kernels: %d %d %d %d\n", kernel_h_, kernel_w_, kernel_w, kernel_h);
    // printf("Channels: %d %d\n", channels, channels_kernel);
    // printf("Channels: %d %d\n", channels_out, channels_kernel);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto columns = at::empty({channels * kernel_h * kernel_w, batch * height_out * width_out}, input.options());
    auto output = at::empty({channels_out, batch, height_out, width_out}, input.options());


    AT_DISPATCH_FLOATING_TYPES(input.type(), "deform_conv_forward_cuda", ([&] {
        deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                         input.data<scalar_t>(),
                                         offset.data<scalar_t>(),
                                         batch, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                         deformable_group,
                                         columns.data<scalar_t>());

    }));

    auto columns_m = columns.t();
    auto weight_m = weight.view({channels_out, channels_kernel * kernel_h * kernel_w}).t();
    output = at::addmm(bias, columns_m, weight_m);
    output = output.view({batch, height_out, width_out, channels_out}).permute({0, 3, 1, 2}).contiguous();

    return output;
}

std::vector<at::Tensor> deform_conv_cuda_backward(const at::Tensor &input,
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

    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(weight.is_contiguous(), "weight tensor has to be contiguous");

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    const int batch_ = grad_output.size(0);
    const int channels_out_ = grad_output.size(1);
    const int height_out_ = grad_output.size(2);
    const int width_out_ = grad_output.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
               "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
               "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    AT_ASSERTM(batch == batch_,
               "Input shape and grad_out batch wont match: (%d vs %d).", batch, batch_);

    AT_ASSERTM(channels_out == channels_out_,
               "Input shape and grad_out channels_out wont match: (%d vs %d).", channels_out, channels_out_);

    AT_ASSERTM(height_out == height_out_ && width_out == width_out_,
               "Input shape and grad_out shape wont match: (%d x %d vs %d x %d).", height_out, height_out_, width_out, width_out_);

    auto ones = at::ones({batch * height_out * width_out}, input.options());
    auto columns = at::empty({channels * kernel_h * kernel_w, batch * 1 * height_out * width_out}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);
    auto grad_offset = at::zeros_like(offset);

    auto grad_output_m = grad_output.permute({1, 0, 2, 3}).contiguous().view({channels_out, batch * height_out * width_out});
    auto weight_m = weight.view({channels_out, channels_kernel * kernel_h * kernel_w}).t();
    columns = at::mm(weight_m, grad_output_m);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "deform_conv_backward_cuda", ([&] {
        deformable_col2im_coord_cuda(at::cuda::getCurrentCUDAStream(),
                                               columns.data<scalar_t>(),
                                               input.data<scalar_t>(),
                                               offset.data<scalar_t>(),
                                               batch, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset.data<scalar_t>());
        // gradient w.r.t. input data
        deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                         columns.data<scalar_t>(),
                                         offset.data<scalar_t>(),
                                         batch, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input.data<scalar_t>());

        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                                         input.data<scalar_t>(),
                                         offset.data<scalar_t>(),
                                         batch, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns.data<scalar_t>());

    }));

    grad_weight = at::mm(grad_output_m, columns.t()).view_as(weight);
    grad_bias = at::mv(grad_output_m, ones);

    return {
        grad_input, grad_offset, grad_weight, grad_bias
    };
}