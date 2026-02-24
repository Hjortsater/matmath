#include "cmat.h"

void mat_add(double* A, double* B, double* C, size_t size) {
    for (size_t i = 0; i < size; i++) C[i] = A[i] + B[i];
}

void mat_sub(double* A, double* B, double* C, size_t size) {
    for (size_t i = 0; i < size; i++) C[i] = A[i] - B[i];
}

void hadamard(double* A, double* B, double* C, size_t size) {
    for (size_t i = 0; i < size; i++) C[i] = A[i] * B[i];
}

void scalar_mul(double* A, double scalar, double* C, size_t size) {
    for (size_t i = 0; i < size; i++) C[i] = A[i] * scalar;
}