#ifndef SCREEN_CONFIG_CUH
#define SCREEN_CONFIG_CUH

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__global__ void ConvertFloatToRGBA(float* w, float4* rgba_out,uchar4 *img, int N);

#endif