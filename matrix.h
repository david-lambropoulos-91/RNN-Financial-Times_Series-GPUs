#ifndef MATRIX_H
#define MATRIX_H

#include <mkl.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

// represents user-supplied training data
typedef struct DataSet_ 
{
    size_t rows;
    size_t cols;
    float** data;
} DataSet;

// represents a matrix of data in row-major order
typedef struct Matrix_ 
{
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

// create dataset given user data
static DataSet* createDataSet(size_t rows, size_t cols, float** data);

// uses memory of the original data to split dataset into batches
static DataSet** createBatches(DataSet* allData, int numBatches);

// split a dataset into row matrices
static Matrix** splitRows(DataSet* dataset);

// shuffle two datasets, maintaining alignment between their rows
static void shuffleTogether(DataSet* A, DataSet* B);

// destroy dataset
static void destroyDataSet(DataSet* dataset);

// convert dataset to matrix
static Matrix* dataSetToMatrix(DataSet* dataset);

// creates a matrix given data
static Matrix* createMatrix(size_t rows, size_t cols, float* data);

// creates a matrix zeroed out
static Matrix* createMatrixZeroes(size_t rows, size_t cols);

// get an element of a matrix
static float getMatrix(Matrix* mat, size_t row, size_t col);

// set an element of a matrix
static void setMatrix(Matrix* mat, size_t row, size_t col, float val);

// sets the values in $to equal to values in $from
static void copyValuesInto(Matrix* from, Matrix* to);

// prints the entries of a matrix
static void printMatrix(Matrix* input);

// sets each entry in matrix to 0
static void zeroMatrix(Matrix* orig);

// returns transpose of matrix
static Matrix* transpose(Matrix* orig);

// transposes matrix and places data into $origT
static void transposeInto(Matrix* orig, Matrix* origT);

// adds two matrices and returns result
static Matrix* add(Matrix* A, Matrix* b);

// adds $from to $to and places result in $to
static void addTo(Matrix* from, Matrix* to);

// adds $B, a row vector, to each row of $A
static Matrix* addToEachRow(Matrix* A, Matrix* B);

// multiplies every element of $orig by $C
static void scalarMultiply(Matrix* orig, float c);

// multiplies $A and $B (ordering: AB) and returns product matrix
static Matrix* multiply(Matrix* A, Matrix* B);

// multiplies $A and $B (ordering: AB) and places values into $into
static void multiplyInto(Matrix* A, Matrix* B, Matrix* into);

// element-wise multiplcation
static Matrix* hadamard(Matrix* A, Matrix* B);

// places values of hadamard product of $A and $B into $into
static void hadamardInto(Matrix* A, Matrix* B, Matrix* into);

// returns a shallow copy of input matrix
static Matrix* copy(Matrix* orig);

// returns 1 if matrices are equal, 0 otherwise
static int equals(Matrix* A, Matrix* B);

// frees a matrix and its data
static void destroyMatrix(Matrix* matrix);

#endif
