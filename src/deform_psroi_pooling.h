#pragma once

#include "cpu/deform_psroi_pooling_cpu.h"

#ifdef WITH_CUDA
#include "cuda/deform_psroi_pooling_cuda.h"
#endif


std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_forward(const at::Tensor &input,
                             const at::Tensor &bbox,
                             const at::Tensor &trans,
                             const int no_trans,
                             const float spatial_scale,
                             const int output_dim,
                             const int group_size,
                             const int pooled_size,
                             const int part_size,
                             const int sample_per_part,
                             const float trans_std)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_psroi_pooling_cuda_forward(input,
                                                 bbox,
                                                 trans,
                                                 no_trans,
                                                 spatial_scale,
                                                 output_dim,
                                                 group_size,
                                                 pooled_size,
                                                 part_size,
                                                 sample_per_part,
                                                 trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_backward(const at::Tensor &out_grad,
                              const at::Tensor &input,
                              const at::Tensor &bbox,
                              const at::Tensor &trans,
                              const at::Tensor &top_count,
                              const int no_trans,
                              const float spatial_scale,
                              const int output_dim,
                              const int group_size,
                              const int pooled_size,
                              const int part_size,
                              const int sample_per_part,
                              const float trans_std)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_psroi_pooling_cuda_backward(out_grad,
                                                  input,
                                                  bbox,
                                                  trans,
                                                  top_count,
                                                  no_trans,
                                                  spatial_scale,
                                                  output_dim,
                                                  group_size,
                                                  pooled_size,
                                                  part_size,
                                                  sample_per_part,
                                                  trans_std);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}