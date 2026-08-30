#ifndef SOLVE_EDOS_CUH
#define SOLVE_EDOS_CUH

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "types.h"

__global__ void SolveNSWithoutPressure(float* u,float* v, float* new_u,float* new_v,bool* a_img,float* fx,float* fy,SIM_ARGS s_args,GPU_ARGS g_args);
__global__ void SolvePoisson(float*u, float* v, float* p,float* new_p, SIM_ARGS s_args,GPU_ARGS g_args);
__global__ void CorrectVelocity(float* u, float* v, float* p,bool* a_img, SIM_ARGS s_args, GPU_ARGS g_args);
__global__ void ComputeVorticity(float* u, float* v, float* w, SIM_ARGS s_args);
__global__ void eval2DFuntionU(FParams fp,int N, float* ptr);
__global__ void eval2DFuntionV(FParams fp,int N, float* ptr);
__global__ void ConvertAlpha2Bool(uchar4 *img, bool* ptr, int N);
__global__ void getForcesFromImage(uchar4 *img, float* ptr_x,float* ptr_y, int N,float max_force);
#endif