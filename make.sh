cd src
nvcc -c -o deform_conv_cuda_kernel.cu.o deform_conv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11

cd cuda

# compile dcn
nvcc -c -o dcn_v2_im2col_cuda.cu.o dcn_v2_im2col_cuda.cu -x cu -Xcompiler -fPIC

# compile dcn-roi-pooling
nvcc -c -o dcn_v2_psroi_pooling_cuda.cu.o dcn_v2_psroi_pooling_cuda.cu -x cu -Xcompiler -fPIC

cd ../..
CC=g++ python build.py
python build_modulated.py
