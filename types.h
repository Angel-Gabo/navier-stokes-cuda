#ifndef TYPES_H
#define TYPES_H

struct SIM_ARGS {
    int N;
    float dx;
    float dy;
    float dt;
    float xi;
    float yi; 
    float v; //Quite la c y puse v para "denotar" viscocidad, cualquier queja a diosito
    float t; //??? NO SE TOCA
    float rho; //Densidad, se deja en 1
    float g; //Gravedad mijo
    float fl; //??? TAMPOCO SE TOCA
};

struct GPU_ARGS
{
    int bloques_por_lado;
};

struct FParams {
    float xi;
    float yi;
    float dx;
    float dy;
};

#endif