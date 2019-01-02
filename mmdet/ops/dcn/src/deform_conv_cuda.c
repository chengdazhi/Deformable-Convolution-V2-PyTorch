#include <THC/THC.h>

#include "deform_conv_cuda_kernel.h"

extern THCState *state;

void shape_check(THCState *state, THCudaTensor *input, THCudaTensor *offset,
                 THCudaTensor *gradOutput, THCudaTensor *weight, int kH, int kW,
                 int dH, int dW, int padH, int padW, int dilationH,
                 int dilationW, int deformable_group) {

//  THArgCheck(weight->nDimension == 4, 5,
//             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
//             "but got: %s",
//             weight->nDimension);
  THArgCheck(THCudaTensor_nDimension(state, weight) == 4, 5,
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             THCudaTensor_nDimension(state, weight));

  THArgCheck(THCudaTensor_isContiguous(state, weight), 5,
             "weight tensor has to be contiguous");

  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d",
             kH, kW);

//  THArgCheck((weight->size[2] == kH && weight->size[3] == kW), 9,
//             "kernel size should be consistent with weight, ",
//             "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
//             kW, weight->size[2], weight->size[3]);
  THArgCheck((THCudaTensor_size(state, weight, 2) == kH &&
             THCudaTensor_size(state, weight, 3) == kW), 9,
             "kernel size should be consistent with weight, ",
             "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, THCudaTensor_size(state, weight, 2), THCudaTensor_size(state, weight, 3));


  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  THArgCheck(dilationW > 0 && dilationH > 0, 14,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

//  int ndim = input->nDimension;
  int ndim = THCudaTensor_nDimension(state, input);
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THArgCheck(ndim == 3 || ndim == 4, 2,
             "3D or 4D input tensor expected but got: %s", ndim);

//  long nInputPlane = weight->size[1];
//  long inputHeight = input->size[dimh];
//  long inputWidth = input->size[dimw];
//  long nOutputPlane = weight->size[0];
  long nInputPlane = THCudaTensor_size(state, weight, 1);
  long inputHeight = THCudaTensor_size(state, input, dimh);
  long inputWidth = THCudaTensor_size(state, input, dimw);
  long nOutputPlane = THCudaTensor_size(state, weight, 0);
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  THArgCheck(nInputPlane % deformable_group == 0, 2,
             "input channels must divide deformable group size");

  if (outputWidth < 1 || outputHeight < 1)
    THError(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  THArgCheck(THCudaTensor_size(state, input, 1) == nInputPlane, 2,
             "invalid number of input planes, expected: %d, but got: %d",
             nInputPlane, THCudaTensor_size(state, input, 1));

  THArgCheck((inputHeight >= kH && inputWidth >= kW), 2,
             "input image is smaller than kernel");

//  THArgCheck(
//      (offset->size[2] == outputHeight && offset->size[3] == outputWidth), 3,
//      "invalid spatial size of offset, expected height: %d width: %d, but got height: %d width: %d", outputHeight, outputWidth,
//      offset->size[2], offset->size[3]);
  THArgCheck(
      (THCudaTensor_size(state, offset, 2) == outputHeight &&
      THCudaTensor_size(state, offset, 3) == outputWidth), 3,
      "invalid spatial size of offset, expected height: %d width: %d, but got height: %d width: %d",
      outputHeight, outputWidth, THCudaTensor_size(state, offset, 2),
      THCudaTensor_size(state, offset, 3));

  THArgCheck((THCudaTensor_size(state, offset, 1) == deformable_group * 2 * kH * kW), 3,
             "invalid number of channels of offset");

  if (gradOutput != NULL) {
    THArgCheck(THCudaTensor_size(state, gradOutput, dimf) == nOutputPlane, 4,
               "invalid number of gradOutput planes, expected: %d, but got: %d",
               nOutputPlane, THCudaTensor_size(state, gradOutput, dimf));

    THArgCheck((THCudaTensor_size(state, gradOutput, dimh) == outputHeight &&
                THCudaTensor_size(state, gradOutput, dimw) == outputWidth),
               4, "invalid size of gradOutput, expected height: %d width: %d , but got height: %d width: %d",
               outputHeight, outputWidth, THCudaTensor_size(state, gradOutput, dimh),
               THCudaTensor_size(state, gradOutput, dimw));
  }
}

int deform_conv_forward_cuda(THCudaTensor *input, THCudaTensor *weight,
                             THCudaTensor *offset, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH,
                             int deformable_group, int im2col_step) {

  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly transpose output)
  // todo: possibly change data indexing because of parallel_imgs

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, weight, offset,
                                         output, columns, ones));

  shape_check(state, input, offset, NULL, weight, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW, deformable_group);

  input = THCudaTensor_newContiguous(state, input);
  offset = THCudaTensor_newContiguous(state, offset);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;
  if (THCudaTensor_nDimension(state, input) == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, THCudaTensor_size(state, input, 0), THCudaTensor_size(state, input, 1),
                          THCudaTensor_size(state, input, 2));
    THCudaTensor_resize4d(state, offset, 1, THCudaTensor_size(state, offset, 0), THCudaTensor_size(state, offset, 1),
                          THCudaTensor_size(state, offset, 2));
  }

  // todo: assert batchsize dividable by im2col_step

  long batchSize = THCudaTensor_size(state, input, 0);
  long nInputPlane = THCudaTensor_size(state, input, 1);
  long inputHeight = THCudaTensor_size(state, input, 2);
  long inputWidth = THCudaTensor_size(state, input, 3);

  long nOutputPlane = THCudaTensor_size(state, weight, 0);

  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((THCudaTensor_size(state, offset, 0) == batchSize), 3, "invalid batch size of offset");

  // bias = bias ? THCudaTensor_newContiguous(state, bias) : bias;

  THCudaTensor_resize5d(state, output, batchSize / im2col_step, im2col_step, nOutputPlane, outputHeight, outputWidth);
  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth);

  if (THCudaTensor_nDimension(state, ones) != 2 || THCudaTensor_size(state, ones, 0) *
      THCudaTensor_size(state, ones, 1) < outputHeight * outputWidth) {
    THCudaTensor_resize2d(state, ones, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *offset_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  THCudaTensor *output_buffer = THCudaTensor_new(state);
  THCudaTensor_resize4d(state, output_buffer, batchSize / im2col_step, nOutputPlane, im2col_step * outputHeight, outputWidth);

  THCudaTensor_resize5d(state, input, batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize5d(state, offset, batchSize / im2col_step, im2col_step,
      deformable_group * 2 * kH * kW, outputHeight, outputWidth);

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {

    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, offset_n, offset, 0, elt);
    THCudaTensor_select(state, output_n, output_buffer, 0, elt);

    // long m_ = nOutputPlane;
    // long n_ = outputHeight * outputWidth;
    // long k_ = 1;

    // TODO(BZ) add bias term
    // if (bias) {
    //   THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
    //                    THCudaTensor_data(state, ones), k_,
    //                    THCudaTensor_data(state, bias), k_, 0.0f,
    //                    THCudaTensor_data(state, output_n), n_);
    // } else {
    //   THCudaTensor_zero(state, output_n);
    // }

    THCudaTensor_zero(state, output_n);

    deformable_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n),
        THCudaTensor_data(state, offset_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        im2col_step, deformable_group, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = THCudaTensor_size(state, columns, 1); // todo: see if we need to change this
    long k = nInputPlane * kH * kW;

    // cublas use column major indexing
    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
                     THCudaTensor_data(state, columns), n,
                     THCudaTensor_data(state, weight), k, 1.0f,
                     THCudaTensor_data(state, output_n), n);
  }

  // the reason I use seemingly redundant output_buffer is that THCudaTensor API handles successive transpose and resize poorly
  THCudaTensor_resize5d(state, output_buffer, batchSize / im2col_step, nOutputPlane, im2col_step, outputHeight, outputWidth);
  THCudaTensor_transpose(state, output_buffer, NULL, 1, 2);
  THCudaTensor_copy(state, output, output_buffer);
  THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);

  THCudaTensor_resize4d(state, input, batchSize, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize4d(state, offset, batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth);

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, offset_n);
  THCudaTensor_free(state, output_n);
  THCudaTensor_free(state, output_buffer);

  if (batch == 0) {
    THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, offset, THCudaTensor_size(state, offset, 1),
        THCudaTensor_size(state, offset, 2), THCudaTensor_size(state, offset, 3));
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, offset);
  THCudaTensor_free(state, weight);
  // if (bias) THCudaTensor_free(state, bias);

  return 1;
}

int deform_conv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *gradOffset, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int deformable_group, int im2col_step) {

  THCAssertSameGPU(THCudaTensor_checkGPU(state, 6, input, gradOutput, weight,
                                         offset, columns, gradInput));

  shape_check(state, input, offset, gradOutput, weight, kH, kW, dH, dW, padH,
              padW, dilationH, dilationW, deformable_group);

  input = THCudaTensor_newContiguous(state, input);
  offset = THCudaTensor_newContiguous(state, offset);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  weight = THCudaTensor_newContiguous(state, weight);

  int batch = 1;

  if (THCudaTensor_nDimension(state, input) == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, THCudaTensor_size(state, input, 0), THCudaTensor_size(state, input, 1),
                          THCudaTensor_size(state, input, 2));
    THCudaTensor_resize4d(state, offset, 1, THCudaTensor_size(state, offset, 0), THCudaTensor_size(state, offset, 1),
                          THCudaTensor_size(state, offset, 2));
    THCudaTensor_resize4d(state, gradOutput, 1, THCudaTensor_size(state, gradOutput, 0),
                          THCudaTensor_size(state, gradOutput, 1), THCudaTensor_size(state, gradOutput, 2));
  }

  long batchSize = THCudaTensor_size(state, input, 0);
  long nInputPlane = THCudaTensor_size(state, input, 1);
  long inputHeight = THCudaTensor_size(state, input, 2);
  long inputWidth = THCudaTensor_size(state, input, 3);

  long nOutputPlane = THCudaTensor_size(state, weight, 0);

  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((THCudaTensor_size(state, offset, 0) == batchSize), 3, "invalid batch size of offset");
  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth);


  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOffset_n = THCudaTensor_new(state);
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *offset_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // change order of grad output
  THCudaTensor_resize5d(state, gradOutput, batchSize / im2col_step, im2col_step, nOutputPlane, outputHeight, outputWidth);
  THCudaTensor_transpose(state, gradOutput, NULL, 1, 2);

  THCudaTensor *gradOutputBuffer = THCudaTensor_new(state);
  THCudaTensor_resize5d(state, gradOutputBuffer, batchSize / im2col_step, nOutputPlane, im2col_step, outputHeight, outputWidth);
  THCudaTensor_copy(state, gradOutputBuffer, gradOutput);
  THCudaTensor_resize4d(state, gradOutputBuffer, batchSize / im2col_step, nOutputPlane, im2col_step * outputHeight, outputWidth);

  THCudaTensor_transpose(state, gradOutput, NULL, 1, 2);
  THCudaTensor_resize4d(state, gradOutput, batchSize, nOutputPlane, outputHeight, outputWidth);

  THCudaTensor_resize5d(state, gradInput, batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize5d(state, input, batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize5d(state, gradOffset, batchSize / im2col_step, im2col_step,
      deformable_group * 2 * kH * kW, outputHeight, outputWidth);
  THCudaTensor_resize5d(state, offset, batchSize / im2col_step, im2col_step,
      deformable_group * 2 * kH * kW, outputHeight, outputWidth);


  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, gradOffset_n, gradOffset, 0, elt);
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, offset_n, offset, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutputBuffer, 0, elt);

    long m = nInputPlane * kW * kH;
    long n = THCudaTensor_size(state, columns, 1);
    long k = nOutputPlane;

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k, 1.0f,
                     THCudaTensor_data(state, gradOutput_n), n,
                     THCudaTensor_data(state, weight), m, 0.0f,
                     THCudaTensor_data(state, columns), n);


    deformable_col2im_coord(
        THCState_getCurrentStream(state), THCudaTensor_data(state, columns),
        THCudaTensor_data(state, input_n), THCudaTensor_data(state, offset_n),
        nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
        dilationH, dilationW, im2col_step, deformable_group,
        THCudaTensor_data(state, gradOffset_n));

    deformable_col2im(
        THCState_getCurrentStream(state), THCudaTensor_data(state, columns),
        THCudaTensor_data(state, offset_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW, im2col_step,
        deformable_group, THCudaTensor_data(state, gradInput_n));
  }

  THCudaTensor_resize4d(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize4d(state, input, batchSize, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize4d(state, gradOffset, batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth);
  THCudaTensor_resize4d(state, offset, batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth);

  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, gradOffset_n);
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, offset_n);
  THCudaTensor_free(state, gradOutput_n);
  THCudaTensor_free(state, gradOutputBuffer);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(state, gradInput, nInputPlane, inputHeight,
                          inputWidth);
    THCudaTensor_resize3d(state, offset, THCudaTensor_size(state, offset, 1), THCudaTensor_size(state, offset, 2),
                          THCudaTensor_size(state, offset, 3));
    THCudaTensor_resize3d(state, gradOffset, THCudaTensor_size(state, offset, 1), THCudaTensor_size(state, offset, 2),
                          THCudaTensor_size(state, offset, 3));
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, offset);
  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, weight);

  return 1;
}

int deform_conv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *offset, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, /*THCudaTensor *gradBias, */
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int deformable_group,
    float scale, int im2col_step) {

  // todo: transpose and reshape outGrad
  // todo: reshape columns
  // todo: add im2col_step as input
  THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, offset, gradOutput,
                                         gradWeight, columns));

  shape_check(state, input, offset, gradOutput, gradWeight, kH, kW, dH, dW,
             padH, padW, dilationH, dilationW, deformable_group);

  input = THCudaTensor_newContiguous(state, input);
  offset = THCudaTensor_newContiguous(state, offset);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  int batch = 1;

  if (THCudaTensor_nDimension(state, input) == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(state, input, 1, THCudaTensor_size(state, input, 0), THCudaTensor_size(state, input, 1),
                          THCudaTensor_size(state, input, 2));
    THCudaTensor_resize4d(state, gradOutput, 1, THCudaTensor_size(state, gradOutput, 0),
                          THCudaTensor_size(state, gradOutput, 1), THCudaTensor_size(state, gradOutput, 2));
  }

  long batchSize = THCudaTensor_size(state, input, 0);
  long nInputPlane = THCudaTensor_size(state, input, 1);
  long inputHeight = THCudaTensor_size(state, input, 2);
  long inputWidth = THCudaTensor_size(state, input, 3);

  long nOutputPlane = THCudaTensor_size(state, gradWeight, 0);

  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  THArgCheck((THCudaTensor_size(state, offset, 0) == batchSize), 3, "invalid batch size of offset");

  THCudaTensor_resize2d(state, columns, nInputPlane * kW * kH,
                        im2col_step * outputHeight * outputWidth);

  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *offset_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  THCudaTensor_resize5d(state, gradOutput, batchSize / im2col_step, im2col_step, nOutputPlane, outputHeight, outputWidth);
  THCudaTensor_transpose(state, gradOutput, NULL, 1, 2);

  THCudaTensor *gradOutputBuffer = THCudaTensor_new(state);
  THCudaTensor_resize5d(state, gradOutputBuffer, batchSize / im2col_step, nOutputPlane, im2col_step, outputHeight, outputWidth);
  THCudaTensor_copy(state, gradOutputBuffer, gradOutput);
  THCudaTensor_resize4d(state, gradOutputBuffer, batchSize / im2col_step, nOutputPlane, im2col_step * outputHeight, outputWidth);

  THCudaTensor_transpose(state, gradOutput, NULL, 1, 2);
  THCudaTensor_resize4d(state, gradOutput, batchSize, nOutputPlane, outputHeight, outputWidth);


  THCudaTensor_resize5d(state, input, batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize5d(state, offset, batchSize / im2col_step, im2col_step,
      deformable_group * 2 * kH * kW, outputHeight, outputWidth);

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, offset_n, offset, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutputBuffer, 0, elt);

    deformable_im2col(
        THCState_getCurrentStream(state), THCudaTensor_data(state, input_n),
        THCudaTensor_data(state, offset_n), nInputPlane, inputHeight,
        inputWidth, kH, kW, padH, padW, dH, dW, dilationH, dilationW,
        im2col_step, deformable_group, THCudaTensor_data(state, columns));

    long m = nOutputPlane;
    long n = nInputPlane * kW * kH;
    long k = THCudaTensor_size(state, columns, 1);

    THCudaBlas_Sgemm(state, 't', 'n', n, m, k, scale,
                     THCudaTensor_data(state, columns), k,
                     THCudaTensor_data(state, gradOutput_n), k, 1.0f,
                     THCudaTensor_data(state, gradWeight), n);
  }

  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, offset_n);
  THCudaTensor_free(state, gradOutput_n);
  THCudaTensor_free(state, gradOutputBuffer);

  THCudaTensor_resize4d(state, input, batchSize, nInputPlane, inputHeight, inputWidth);
  THCudaTensor_resize4d(state, offset, batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth);

  if (batch == 0) {
    THCudaTensor_resize3d(state, gradOutput, nOutputPlane, outputHeight,
                          outputWidth);
    THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
  }

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, offset);
  THCudaTensor_free(state, gradOutput);
  return 1;
}
