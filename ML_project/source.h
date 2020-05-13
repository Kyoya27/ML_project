#ifndef LIB_LIBRARY_H
#define LIB_LIBRARY_H
typedef struct MLP {
    int* npl;
    int npl_size;
    double*** w;
    double** x;
    double** deltas;
} MLP;
#endif