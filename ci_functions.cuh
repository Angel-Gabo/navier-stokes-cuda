#ifndef CI_FUNCTIONS_CUH
#define CI_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

 
__device__ float u_inicial(float x,float y){
    return 50.0f;
}
__device__ float v_inicial(float x,float y){
    return __cosf(0.6*y*x);
}

//Condiciones iniciales extra sugeridas por gemini
/*
__device__ float u_inicial(float x, float y) {
    // Aumentamos la amplitud a 40.0f
    float x1 = x - 6.0f, y1 = y - 10.0f;
    float denom1 = __powf(x1 * x1 + y1 * y1 + 0.5f, 3.0f);
    float u1 = 40.0f * x1 / denom1;

    float x2 = -(x - 14.0f), y2 = y - 10.0f;
    float denom2 = __powf(x2 * x2 + y2 * y2 + 0.5f, 3.0f);
    float u2 = -40.0f * x2 / denom2;

    return u1 + u2;
}

__device__ float v_inicial(float x, float y) {
    float x1 = x - 6.0f, y1 = y - 10.0f;
    float denom1 = __powf(x1 * x1 + y1 * y1 + 0.5f, 3.0f);
    float v1 = -40.0f * y1 / denom1;

    float x2 = -(x - 14.0f), y2 = y - 10.0f;
    float denom2 = __powf(x2 * x2 + y2 * y2 + 0.5f, 3.0f);
    float v2 = -40.0f * y2 / denom2;

    return v1 + v2;
}

/*
__device__ static inline float u_inicial(float x, float y){
    float kx = 2.0f*3.14159265f / 5.0f; 
    float ky = 2.0f*3.14159265f / 5.0f;
    return 10.0f*__sinf(kx*x)*__cosf(ky * y);
}

__device__ static inline float v_inicial(float x, float y){
    float kx = 2.0f * 3.14159265f / 5.0f; 
    float ky = 2.0f * 3.14159265f / 5.0f;
    return -10.0f*__cosf(kx * x)*__sinf(ky * y);
}
*/
/*
__device__ static inline float u_inicial(float x, float y){
    float des_x = x-10;
    float des_y = y-10;

    float exp_term = __expf(-des_x-des_y);
    float div_term = 1 + exp_term;
    return -10*des_y*exp_term/(div_term*div_term);

}
__device__ static inline float v_inicial(float x, float y){
    float des_x = x-10;
    float des_y = y-10;

    float exp_term = __expf(-des_x-des_y);
    float div_term = 1 + exp_term;
    return -10*des_x*exp_term/(div_term*div_term);

}
*/
#endif