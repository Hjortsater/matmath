#include <Python.h>
#include <stddef.h>
#include "cmat.h"

static int list_to_array(PyObject* list, double* arr, size_t size) {
    if (!PyList_Check(list) || (size_t)PyList_Size(list) != size) return 0;
    for (size_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(list, i);
        arr[i] = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) return 0;
    }
    return 1;
}

static PyObject* array_to_list(double* arr, size_t size) {
    PyObject* result = PyList_New(size);
    if (!result) return NULL;
    for (size_t i = 0; i < size; i++) {
        PyList_SetItem(result, i, PyFloat_FromDouble(arr[i]));
    }
    return result;
}

static int validate_size(PyObject* list1, PyObject* list2, int m, int n, size_t* size) {
    if (!PyList_Check(list1) || !PyList_Check(list2)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be lists.");
        return 0;
    }
    *size = (size_t)(m * n);
    if ((size_t)PyList_Size(list1) != *size || (size_t)PyList_Size(list2) != *size) {
        PyErr_SetString(PyExc_ValueError, "List size does not match provided dimensions.");
        return 0;
    }
    return 1;
}

static int validate_size_single(PyObject* list, int m, int n, size_t* size) {
    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of floats.");
        return 0;
    }
    *size = (size_t)(m * n);
    if ((size_t)PyList_Size(list) != *size) {
        PyErr_SetString(PyExc_ValueError, "List size does not match provided dimensions.");
        return 0;
    }
    return 1;
}

typedef void (*binary_op_func)(double*, double*, double*, size_t);

static PyObject* py_binary_op(PyObject* self, PyObject* args, binary_op_func op) {
    PyObject *list1, *list2;
    int m, n;
    if (!PyArg_ParseTuple(args, "OOii", &list1, &list2, &m, &n)) return NULL;
    size_t size;
    if (!validate_size(list1, list2, m, n, &size)) return NULL;
    double A[size], B[size], C[size];
    if (!list_to_array(list1, A, size) || !list_to_array(list2, B, size)) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert lists to floats.");
        return NULL;
    }
    op(A, B, C, size);
    return array_to_list(C, size);
}

static PyObject* py_scalar_mul(PyObject* self, PyObject* args) {
    PyObject* list;
    double scalar;
    int m, n;
    if (!PyArg_ParseTuple(args, "Oidd", &list, &m, &n, &scalar)) return NULL;
    size_t size;
    if (!validate_size_single(list, m, n, &size)) return NULL;
    double A[size], C[size];
    if (!list_to_array(list, A, size)) {
        PyErr_SetString(PyExc_TypeError, "Failed to convert list to floats.");
        return NULL;
    }
    scalar_mul(A, scalar, C, size);
    return array_to_list(C, size);
}

static PyObject* py_mat_add(PyObject* self, PyObject* args) {
    return py_binary_op(self, args, mat_add);
}

static PyObject* py_mat_sub(PyObject* self, PyObject* args) {
    return py_binary_op(self, args, mat_sub);
}

static PyObject* py_hadamard(PyObject* self, PyObject* args) {
    return py_binary_op(self, args, hadamard);
}

static PyMethodDef CMatMethods[] = {
    {"mat_add", py_mat_add, METH_VARARGS, "Add two matrices"},
    {"mat_sub", py_mat_sub, METH_VARARGS, "Subtract two matrices"},
    {"hadamard", py_hadamard, METH_VARARGS, "Hadamard product of two matrices"},
    {"scalar_mul", py_scalar_mul, METH_VARARGS, "Multiply matrix by scalar"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cmatmodule = {
    PyModuleDef_HEAD_INIT,
    "cmat",
    "C backend for Matrix operations",
    -1,
    CMatMethods
};

PyMODINIT_FUNC PyInit_cmat(void) {
    return PyModule_Create(&cmatmodule);
}