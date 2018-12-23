cd src
nvcc -c -o deform_conv_cuda_kernel.cu.o deform_conv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11
nvcc -c -o roioffset_pooling.cu.o roioffset_pooling_kernel.cu -x cu -Xcompiler -fPIC -std=c++11

#nvcc -c -o roioffset_pooling.cu.o roioffset_pooling_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
