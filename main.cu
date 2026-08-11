//OJO AL LECTOR INTELIGENTE
//NO VOY A HACER ENFASIS EN LA DISTRIBUCION DE LA MEMORIA COMPARTIDA EN CUDA
//Leer main.cu de pde2D

//Este programa solo va a profundizar en la fisica
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#include "glad/glad.h" 
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "types.h"
#include "solve_edos.cuh"
#include "screen_config.cuh"


int main(){

    if (!glfwInit()) return -1;
    
    GLFWwindow* window = glfwCreateWindow(800, 800, "PDE 2D", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        printf("Error al inicializar GLAD\n");
        return -1;
    }

    cudaFree(0);
    //Tamaño por lado de la malla
    int N = 500;
    int N_N = N*N;
    //Dimensiones de la caja - dejar cuadrada porque la poisson no la voy a modificar hasta nuevo aviso
    float xi = 0.0f;
    float yi = 0.0f;
    float xf = 20.0f;
    float yf = xf; //DEJAR ASI PORQUE SI NO TODOS MORIMOS

    //Ajustar basandose en la evaluacion que arroja el programa
    float dt = 1e-4;
    int steps_in_frame = 50; //Pasos que calcula la gpu antes de mostrar en pantalla
    int steps_for_pressure = 30; //Pasos que se usan para la ecuacion de Poisson, entre 20 y 50 esta bien


    float dx = (xf-xi)/(float)N;
    float dy = (yf-yi)/(float)N;

    float viscocidad = 0.001;
    float rho = 1; //Densidad del fluido, dejar en 1
    float r = viscocidad*dt/(dx*dx);
    float g = 100.0f; //Gravedad

    printf("Evaluacion de la estabilidad de la solucion: \n");
    printf("%0.5f \n",r);
    printf("Al comparar con 0.25...");
    if(r<=0.25f){
        printf("La solucion es estable");
    }
    else{
        printf("La solucion no sera estable");
    }
    printf("\n");

    //Ahora definimos los punteros con los que vamos a trabajar
    //float* cpu_phi = (float*)calloc(N_N,sizeof(float)); //Solo ocupamos este puntero en cpu, todos los demas estaran en gpu
    
    //Cargamos los punteros en gpu
    float *u,*v,*nu,*nv,*w,*p,*np;
    bool *a_img;
    cudaMalloc((void**)&u,N_N*sizeof(float));
    cudaMalloc((void**)&nu,N_N*sizeof(float));

    cudaMalloc((void**)&v,N_N*sizeof(float));
    cudaMalloc((void**)&nv,N_N*sizeof(float));
    cudaMalloc((void**)&w,N_N*sizeof(float));
    cudaMalloc((void**)&p,N_N*sizeof(float));
    cudaMalloc((void**)&np,N_N*sizeof(float));
    cudaMalloc((void**)&a_img,N_N*sizeof(bool));

    //Los punteros asociados a la presion van a tener ceros por defecto siempre para evitar problemas al resolver poisson
    cudaMemset(p, 0, N_N*sizeof(float));
    cudaMemset(np, 0, N_N*sizeof(float));
    cudaMemset(a_img,0,N_N*sizeof(bool)); //Lo lleno de ceros para evitar basuras

    float4* d_rgba_buffer;
    cudaMalloc((void**)&d_rgba_buffer, N_N * sizeof(float4));

    //Ojo, blocks esta hecho a medida unica y exclusivamente para las ED
    int threads = 256; //NO CAMBIAR
    int bloques_por_lado = (N + 15) / 16;
    int blocks = bloques_por_lado*bloques_por_lado;

    FParams fp = {
        .xi = xi, .yi = yi, .dx = dx, .dy = dy
    };

    //Para las CI y en general cualquier calculo donde podamos darnos el lujo de no ultra optimizar vamos a usar init_blocks
    int init_blocks = (N+threads-1)/threads;
    //Ahora evaluamos la condicion inicial
    eval2DFuntionU<<<init_blocks,threads>>>(fp,N,u);
    eval2DFuntionV<<<init_blocks,threads>>>(fp,N,v);


    //Cargamos la imagen con objetos dentro
    int width,height, channels;
    int d_channels = 4;
    stbi_set_flip_vertically_on_load(1);
    unsigned char *img = stbi_load("../figura.png",&width,&height,&channels,d_channels);

    uchar4 *img_in_cuda;

    //Necesitamos asegurar que las dimensiones coincidan con la simulacion
    if(!img || width != N || height != N){
        printf("OOOPS, ocurrio un error... \n");
        printf("Asegurate que la imagen cargada exista y ademas tenga las dimensiones de la malla \n");
        //Un easter egg para los curiosos
        if(!img){
            //Tenia mucha cafeina mientras escribia esto... pero no lo borrare
            printf("BUSCA TENER VIDA SOCIAL PORQUE ESTO NO ES LO TUYO \n");
            printf("AMIGO/A, UNA TAREA TENIAS Y NO LA HICISTE BIEN...");
            printf("Solo pon el nombre de la imagen correctamente \n");
        }
        if(width != N || height != N){
            printf("Las dimensiones de la imagen no coinciden, las dimensiones esperadas son ");
            printf("%dx%d",N,N);
            printf("pero se obtuvo %dx%d \n",width,height);
        }
        //return 0;
    }
    cudaMalloc((void**)&img_in_cuda,N*N*sizeof(uchar4));
    cudaMemcpy(img_in_cuda,img,N*N*sizeof(uchar4),cudaMemcpyHostToDevice);
    
    
    dim3 blockDimColors(16, 16);
    dim3 gridDimColors((N + 15)/16, (N + 15)/16);
    
    ConvertAlpha2Bool<<<blockDimColors,gridDimColors>>>(img_in_cuda,a_img,N);

    stbi_image_free(img); //Ya no ocupamos la imagen

    GLuint textureID;
    glGenTextures(1,&textureID);
    glBindTexture(GL_TEXTURE_2D,textureID);


    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, N, N, 0, GL_RGBA, GL_FLOAT, NULL);
    glEnable(GL_TEXTURE_2D);

    //Para evitar mandar y regresar informacion entre cpu y gpu dedicada, vamos a graficar usando cuda y no las graficas integradas
    //En caso de querer liberar uso de la gpu dedicada y usar la integrada, esto se tiene que modificar, pero habrá latencia agregada
    struct cudaGraphicsResource* cuda_tex_res;
    cudaError_t err = cudaGraphicsGLRegisterImage(
        &cuda_tex_res, 
        textureID, 
        GL_TEXTURE_2D, 
        cudaGraphicsRegisterFlagsWriteDiscard
    );

    if (err != cudaSuccess) {
        printf("Error al registrar la textura en CUDA: %s\n", cudaGetErrorString(err));
    }

    GLuint fboID;
    glGenFramebuffers(1, &fboID);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fboID);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);

    //Verificación de seguridad
    if(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("Error: El Framebuffer no está completo.\n");
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    //cudaMemcpy(cpu_phi,phi,N_N*sizeof(float),cudaMemcpyDeviceToHost);
    //show2Dptr(cpu_phi,N);

    SIM_ARGS sm_args = {
        .N=N,
        .dx = dx,
        .dy = dy,
        .dt = dt,
        .xi = xi,
        .yi = yi,
        .v = viscocidad,
        .rho = rho,
        .g = g
    };
    GPU_ARGS g_args = {
        .bloques_por_lado=(N + 15) / 16,
    };
    int k=0;
    while(!glfwWindowShouldClose(window)){
        sm_args.t = (float)k*dt;  
        //calculamos el siguiente paso
        for(int i=0;i<steps_in_frame;i++){
            //Resolvemos para U,V sin presion
            SolveNSWithoutPressure<<<blocks,threads>>>(u,v,nu,nv,a_img,sm_args,g_args);
            cudaDeviceSynchronize();
            float* temp_u = u;
            u = nu;
            nu = temp_u;

            float* temp_v = v;
            v = nv;
            nv = temp_v;
            //Resolvemos la ecuacion de Poisson para la presion
            for(int n=0;n<steps_for_pressure;n++){
                SolvePoisson<<<blocks,threads>>>(u,v,p,np,sm_args,g_args);

                float* temp_p = p;
                p = np;
                np = temp_p;

            }
            CorrectVelocity<<<blocks,threads>>>(u, v, p,a_img, sm_args, g_args);
            cudaDeviceSynchronize();
            k++;
        }

        ComputeVorticity<<<gridDimColors, blockDimColors>>>(u, v, w, sm_args); //Vamos a graficar la vorticidad del campo vectorial, no al campo en si
        ConvertFloatToRGBA<<<gridDimColors, blockDimColors>>>(w, d_rgba_buffer,img_in_cuda, N); //El formato que se usa es float4(son 4 numeros flotantes: rgba)
        cudaDeviceSynchronize();

        
        cudaGraphicsMapResources(1,&cuda_tex_res,0);
        cudaArray_t texture_array;
        cudaGraphicsSubResourceGetMappedArray(&texture_array,cuda_tex_res,0,0);
        
        //Necesitamos copiar el buffer a la textura ya registrada
        cudaMemcpy2DToArray(
            texture_array,
            0,0,
            d_rgba_buffer,         
            N * sizeof(float4),
            N * sizeof(float4), N, 
            cudaMemcpyDeviceToDevice
        );
        cudaGraphicsUnmapResources(1,&cuda_tex_res,0);

        //glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        //glClearColor(0.2f, 0.3f, 0.3f, 1.0f); 
        //glClear(GL_COLOR_BUFFER_BIT);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fboID);
        
        glBlitFramebuffer(
            0, 0, N, N,           
            0, 0, 800, 800,       
            GL_COLOR_BUFFER_BIT,  
            GL_NEAREST
        );
        
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    cudaGraphicsUnregisterResource(cuda_tex_res);
    cudaFree(d_rgba_buffer);
    return 0;
}