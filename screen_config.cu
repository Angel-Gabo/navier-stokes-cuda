#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <stdlib.h>


__global__ void ConvertFloatToRGBA(float* w, float4* rgba_out,uchar4 *img, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {
        int idx = y * N + x;
        float val = w[idx];
        float norm = __tanhf(val * 0.5f);
        
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (fabsf(norm) > 0.01f) {
            if (norm > 0.0f) {
                r = norm;
            } else {
                b = -norm;
            }
        }
        
        uchar4 px = img[idx];
        float img_r = px.x/255.0f;
        float img_g = px.y/255.0f;
        float img_b = px.z/255.0f;
        float alpha = px.w/255.0f;

        float f_r = img_r*alpha + r*(1.0f-alpha);
        float f_g = img_g*alpha + g*(1.0f-alpha);
        float f_b = img_b*alpha + b*(1.0f-alpha);
        
        
        rgba_out[idx] = make_float4(f_r, f_g, f_b, 1.0f); 


    }
}
