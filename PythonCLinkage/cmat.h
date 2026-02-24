#ifndef CMAT_H
#define CMAT_H

#include <stddef.h>

void mat_add(double* A, double* B, double* C, size_t size);

void mat_sub(double* A, double* B, double* C, size_t size);

void hadamard(double* A, double* B, double* C, size_t size);

void scalar_mul(double* A, double scalar, double* C, size_t size);

#endif