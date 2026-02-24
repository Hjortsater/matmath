#include <stddef.h>

void mat_add(const double* A,
             const double* B,
             double* C,
             size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] + B[i];
}

void mat_sub(const double* A,
             const double* B,
             double* C,
             size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] - B[i];
}

void hadamard(const double* A,
              const double* B,
              double* C,
              size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] * B[i];
}

void scalar_mul(const double* A,
                double scalar,
                double* C,
                size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] * scalar;
}