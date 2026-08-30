#include <types.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "ci_functions.cuh"

//Codigo ultra comentado para no perder hilo
__global__ void SolveNSWithoutPressure(float* u,float* v, float* new_u,float* new_v,bool* a_img,float* fx,float* fy,SIM_ARGS s_args,GPU_ARGS g_args){
    __shared__ float cacheU[18][18]; //Aqui consideramos las fronteras del stride
    __shared__ float cacheV[18][18];
    //__shared__ float cachelPhi[16][16]; //Aqui no vamos a necesitar las fronteras
    int tX = threadIdx.x;
    //int idxGlobal = blockDim.x*blockIdx.x + threadIdx.x;
    
    int bloques_por_lado = g_args.bloques_por_lado;
    int bloque_fila = blockIdx.x / bloques_por_lado;
    int bloque_columna = blockIdx.x % bloques_por_lado;

    int n_fila = bloque_fila * 16;
    int n_columna = bloque_columna * 16;

    int thn_fila = tX/16;
    int thn_columna = tX%16;
    
    int fila_global = n_fila + thn_fila;
    int columna_global = n_columna + thn_columna;


    bool idx_valido;
    int idx = -1;
    bool alpha_bul;//JAJAJAJJA BUUUUUL

    float fx_v=0.0f;
    float fy_v=0.0f;


    if(fila_global>=0 && fila_global<s_args.N && columna_global>=0 && columna_global<s_args.N){
        idx = fila_global*s_args.N+columna_global;
        fx_v = fx[idx];
        fy_v = fy[idx];
        alpha_bul = a_img[idx];
        
        float factor_fluido = 1-(float)alpha_bul;
        cacheU[thn_fila+1][thn_columna+1] = factor_fluido*u[idx];
        cacheV[thn_fila+1][thn_columna+1] = factor_fluido*v[idx];
        //cachelPhi[thn_fila][thn_columna] = lphi[idx];
        idx_valido = true;
    }
    else{
        //Haremos que en las fronteras el fluido no se deslize
        cacheU[thn_fila+1][thn_columna+1]=0;
        cacheV[thn_fila+1][thn_columna+1]=0;
        //cachelPhi[thn_fila][thn_columna] = 0.0f;
        idx_valido = false;
    }

    if(tX<68){
        int fila;
        int columna;
        if(tX<18){
            fila = 0;
            columna = tX;
        }
        else if(tX<36){
            fila = 17;
            columna = tX-18;
        }
        else if(tX<52){
            fila = (tX-36)+1;
            columna = 0;
        }
        else{
            fila = (tX-52)+1;
            columna = 17;
        }
        
        int fg = fila + n_fila-1;
        int cg = columna + n_columna-1;
        if(fg>=0 && fg<s_args.N && cg>=0 && cg<s_args.N){
            cacheU[fila][columna] = u[fg*s_args.N+cg];
            cacheV[fila][columna] = v[fg*s_args.N+cg];
        }
        else{
            cacheU[fila][columna] = 0;
            cacheV[fila][columna] = 0;
        }
    }
    __syncthreads();
    if(idx_valido && !alpha_bul){
        int c_fila = thn_fila+1;
        int c_columna = thn_columna+1;
        
        //Cargamos u
        float center_u = cacheU[c_fila][c_columna];
        float up_u = cacheU[c_fila-1][c_columna];
        float down_u = cacheU[c_fila+1][c_columna];
        float left_u = cacheU[c_fila][c_columna-1];
        float right_u = cacheU[c_fila][c_columna+1];

        //Cargamos v
        float center_v = cacheV[c_fila][c_columna];
        float up_v = cacheV[c_fila-1][c_columna];
        float down_v = cacheV[c_fila+1][c_columna];
        float left_v = cacheV[c_fila][c_columna-1];
        float right_v = cacheV[c_fila][c_columna+1];

        //Calculamos los laplacianos escalados(es decir, les falta dividir sobre dx^2 o dy^2) y luego calculamos el laplaciano completo
        float laplacian_x_scaled_u = right_u-2*center_u+left_u;
        float laplacian_y_scaled_u = up_u-2*center_u+down_u;
        float laplacian_u = laplacian_x_scaled_u/(s_args.dx*s_args.dx)+laplacian_y_scaled_u/(s_args.dy*s_args.dy);

        float laplacian_x_scaled_v = right_v-2*center_v+left_v;
        float laplacian_y_scaled_v = up_v-2*center_v+down_v;
        float laplacian_v = laplacian_x_scaled_v/(s_args.dx*s_args.dx)+laplacian_y_scaled_v/(s_args.dy*s_args.dy);


        float adv_u_x = (center_u > 0.0f) ? center_u * (center_u - left_u) / s_args.dx 
                                          : center_u * (right_u - center_u) / s_args.dx;
                                          
        float adv_u_y = (center_v > 0.0f) ? center_v * (center_u - up_u) / s_args.dy 
                                          : center_v * (down_u - center_u) / s_args.dy;

        float adv_v_x = (center_u > 0.0f) ? center_u * (center_v - left_v) / s_args.dx 
                                          : center_u * (right_v - center_v) / s_args.dx;
                                          
        float adv_v_y = (center_v > 0.0f) ? center_v * (center_v - up_v) / s_args.dy 
                                          : center_v * (down_v - center_v) / s_args.dy;

        float nu = center_u+s_args.dt*(s_args.v*laplacian_u-adv_u_x-adv_u_y+s_args.fl+fx_v);
        float nv = center_v+s_args.dt*(s_args.v*laplacian_v-adv_v_x-adv_v_y-s_args.g+fy_v);
        if (!isfinite(nu) || fabsf(nu) > 1e5f){nu=center_u;}
        if (!isfinite(nv) || fabsf(nv) > 1e5f){nv=center_v;}

        //float center_lphi = cachelPhi[thn_fila][thn_columna];
   
        //Y POR FIN ACABAMOS... EL CALCULO AQUI SE TERMINA   
        new_u[idx] = nu; //Asignamos a la cache
        new_v[idx] = nv;
        //w[idx] = du_dx+dv_dy;
    }
}


__global__ void SolvePoisson(float*u, float* v, float* p,float* new_p, SIM_ARGS s_args,GPU_ARGS g_args){
    __shared__ float cacheU[18][18];
    __shared__ float cacheV[18][18];
    __shared__ float cacheP[18][18];

    int tX = threadIdx.x;
    //int idxGlobal = blockDim.x*blockIdx.x + threadIdx.x;
    
    int bloques_por_lado = g_args.bloques_por_lado;
    int bloque_fila = blockIdx.x / bloques_por_lado;
    int bloque_columna = blockIdx.x % bloques_por_lado;

    int n_fila = bloque_fila * 16;
    int n_columna = bloque_columna * 16;

    int thn_fila = tX/16;
    int thn_columna = tX%16;
    
    int fila_global = n_fila + thn_fila;
    int columna_global = n_columna + thn_columna;


    bool idx_valido;
    int idx = -1;
    if(fila_global>0 && fila_global<s_args.N && columna_global>0 && columna_global<s_args.N){
        idx = fila_global*s_args.N+columna_global;
        cacheU[thn_fila+1][thn_columna+1] = u[idx];
        cacheV[thn_fila+1][thn_columna+1] = v[idx];
        cacheP[thn_fila+1][thn_columna+1] = p[idx];
        
        idx_valido = true;
    }
    else{
        
        cacheU[thn_fila+1][thn_columna+1]=0.0f;
        cacheV[thn_fila+1][thn_columna+1]=0.0f;
        if (fila_global == 0) {
            cacheP[thn_fila+1][thn_columna+1] = p[s_args.N + columna_global];
        } 
        else if (fila_global == s_args.N - 1) {
            cacheP[thn_fila+1][thn_columna+1] = p[(s_args.N - 2)*s_args.N + columna_global];
        } 
        if (columna_global == 0) {
            cacheP[thn_fila+1][thn_columna+1] = p[fila_global*s_args.N + 1];
        } 
        else if (columna_global == s_args.N - 1) {
            cacheP[thn_fila+1][thn_columna+1] = p[fila_global*s_args.N + (s_args.N - 2)];
        }
        idx_valido = false;
    }

    if(tX<68){
        int fila;
        int columna;
        //IFS SUPER COMPACTOS
        if(tX<18){fila = 0;columna = tX;}
        else if(tX<36){fila = 17;columna = tX-18;}
        else if(tX<52){fila = (tX-36)+1;columna = 0;}
        else{fila = (tX-52)+1;columna = 17;}
        
        int fg = fila + n_fila-1;
        int cg = columna + n_columna-1;
        if(fg>=0 && fg<s_args.N && cg>=0 && cg<s_args.N){
            int idx_fc = fg*s_args.N+cg;
            cacheU[fila][columna] = u[idx_fc];
            cacheV[fila][columna] = v[idx_fc];
            cacheP[fila][columna] = p[idx_fc];
        }
        else{
            cacheU[fila][columna] = 0;
            cacheV[fila][columna] = 0;
           int target_fg = fg;
            if(fg < 0){
                target_fg = 1;
            }
            else if(fg >= s_args.N){
                target_fg = s_args.N - 2;
            }
            int target_cg = cg;
            if (cg < 0){
                target_cg = 1;
            }
            else if (cg >= s_args.N){
                target_cg = s_args.N - 2;
            };
            cacheP[fila][columna] = p[target_fg * s_args.N + target_cg];
        }
    }
    __syncthreads();


    if(idx_valido){
        int c_fila = thn_fila+1;
        int c_columna = thn_columna+1;

        float v_up = cacheV[c_fila-1][c_columna];
        float v_down = cacheV[c_fila+1][c_columna];

        float u_left = cacheU[c_fila][c_columna-1];
        float u_right = cacheU[c_fila][c_columna+1];

        float div = (u_right-u_left)/(2*s_args.dx)+(v_down-v_up)/(2*s_args.dy);
        float poisson_nht = s_args.dx*s_args.dx*s_args.rho*div/s_args.dt; //OJo que esta parte solo funciona si dx=dy
        
        float p_left = cacheP[c_fila-1][c_columna];
        float p_right = cacheP[c_fila+1][c_columna];
        float p_up = cacheP[c_fila][c_columna-1];
        float p_down = cacheP[c_fila][c_columna+1];

        float newP = 0.25*(p_up+p_down+p_left+p_right-poisson_nht);

        new_p[idx] = newP;
    }
}

//Aqui vamos a realizar la correccion a partir de la solucion de la ecuacion de Poisson
__global__ void CorrectVelocity(float* u, float* v, float* p,bool* a_img, SIM_ARGS s_args, GPU_ARGS g_args){
    int tX = threadIdx.x;
    
    int bloques_por_lado = g_args.bloques_por_lado;
    int bloque_fila = blockIdx.x / bloques_por_lado;
    int bloque_columna = blockIdx.x % bloques_por_lado;

    int n_fila = bloque_fila * 16;
    int n_columna = bloque_columna * 16;

    int thn_fila = tX / 16;
    int thn_columna = tX % 16;
    
    int fila_global = n_fila + thn_fila;
    int columna_global = n_columna + thn_columna;


    if (fila_global > 0 && fila_global < s_args.N - 1 && columna_global > 0 && columna_global < s_args.N - 1) {
        //Hacemos la lectura 1 a 1 de cada punto
        int idx = fila_global * s_args.N + columna_global;
        bool a_arg = a_img[idx];

        float dp_dx = (p[idx + 1] - p[idx - 1]) / (2.0f * s_args.dx);
        float dp_dy = (p[(fila_global + 1) * s_args.N + columna_global] - p[(fila_global - 1) * s_args.N + columna_global]) / (2.0f * s_args.dy);
        float scale = (1-(float)a_arg)*s_args.dt/s_args.rho;

        //Le calculamos la correcion
        u[idx] -= scale * dp_dx;
        v[idx] -= scale * dp_dy;
    }
    else if (fila_global >= 0 && fila_global < s_args.N && columna_global >= 0 && columna_global < s_args.N) {
        int idx = fila_global * s_args.N + columna_global;
        u[idx] = 0.0f;
        v[idx] = 0.0f;
    }
}
//Calculamos la vorticidad con la corrección de U,V 
__global__ void ComputeVorticity(float* u, float* v, float* w, SIM_ARGS s_args){
    //Obtenemos la coordenada en que estamos
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Hacemos la clasica verificacion de que (x,y) no se sale de la malla
    if (x>0 && x<s_args.N-1 && y>0 && y<s_args.N-1) {
        int idx = y * s_args.N + x;
        
        float dv_dx = (v[idx+1] - v[idx-1]) / (2.0f*s_args.dx);
        float du_dy = (u[(y+1)*s_args.N+x]-u[(y-1)*s_args.N+x])/(2.0f*s_args.dy);
        w[idx] = dv_dx - du_dy;
    }
}

//Por simplicidad y como solo será ejecutada al inicio, la haremos muy sencilla pero veloz
__global__ void eval2DFuntionU(FParams fp,int N, float* ptr){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<N){
        float yn = fp.yi + (float)idx*fp.dy;
        for(int i=0;i<N;i++){
            float xn = fp.xi + (float)i*fp.dx;
            ptr[idx*N+i] = u_inicial(xn,yn);
        }
    }
}

__global__ void eval2DFuntionV(FParams fp,int N, float* ptr){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<N){
        float yn = fp.yi + (float)idx*fp.dy;
        for(int i=0;i<N;i++){
            float xn = fp.xi + (float)i*fp.dx;
            ptr[idx*N+i] = v_inicial(xn,yn);
        }
    }
}

__global__ void ConvertAlpha2Bool(uchar4 *img, bool* ptr, int N){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<N && y<N){
        int idx = y*N + x;
        uchar4 px = img[idx];
        float alpha = px.w/255.0f;
        
        ptr[idx] = (alpha>=0.5) ? 1 : 0;
    }
}


__global__ void getForcesFromImage(uchar4 *img, float* ptr_x,float* ptr_y, int N,float max_force){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x<N && y<N){
        int idx = y*N + x;
        uchar4 px = img[idx];
        float red = px.x/255.0f;
        float green = px.y/255.0f;

        //Normalizamos entre -1 y 1
        float x_force = 2*red-1;
        float y_force = 2*green-1;

        //Para permitir fuerzas variadas vamos a asumir un maximo de fuerza permitido
        float alpha = max_force*px.w/255.0f;
        
        //FACILISIMO
        ptr_x[idx] = alpha*x_force;
        ptr_y[idx] = alpha*y_force;
    }
}