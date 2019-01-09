#pragma once
#include <torch/extension.h>


std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_cpu_forward(const at::Tensor &input,
                                  const at::Tensor &bbox,
                                  const at::Tensor &trans,
                                  const int no_trans,
                                  const float spatial_scale,
                                  const int output_dim,
                                  const int group_size,
                                  const int pooled_size,
                                  const int part_size,
                                  const int sample_per_part,
                                  const float trans_std);

std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_cpu_backward(const at::Tensor &out_grad,
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
                                   const float trans_std);