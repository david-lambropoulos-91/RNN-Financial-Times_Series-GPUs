#include "matrix.h"

static DataSet* createDataSet(size_t rows, size_t cols, float** data)
{
    DataSet* dataset = (DataSet*)malloc(sizeof(DataSet));
    dataset->rows = rows;
    dataset->cols = cols;
    dataset->data = data;
    return dataset;
}

DataSet** createBatches(DataSet* allData, int numBatches)
{
    DataSet** batches = (DataSet**)malloc(sizeof(DataSet*) * numBatches);
    int remainder = allData->rows % numBatches;
    int i;
    int curRow = 0;
    for (i = 0; i < numBatches; i++)
    {
        size_t batchSize = allData->rows / numBatches;
        if (remainder-- > 0)
	{
            batchSize++;
        }
        batches[i] = createDataSet(batchSize, allData->cols, allData->data + curRow);
        curRow += batchSize;
    }
    return batches;
}

static Matrix** splitRows(DataSet* dataset)
{
    Matrix** rows = (Matrix**)malloc(sizeof(Matrix*) * dataset->rows);
    int i;
    for (i = 0; i < dataset->rows; i++)
    {
        rows[i] = createMatrix(1, dataset->cols, dataset->data[i]);
    }

    return rows;
}

void shuffleTogether(DataSet* A, DataSet* B)
{
    assert(A->rows == B->rows);
    int i;
    for (i = 0; i < A->rows - 1; i++)
    {
        size_t j = i + rand() / (RAND_MAX / (A->rows - i) + 1);
        float* tmpA = A->data[j];
        A->data[j] = A->data[i];
        A->data[i] = tmpA;
        float* tmpB = B->data[j];
        B->data[j] = B->data[i];
        B->data[i] = tmpB;
    }
}

static void destroyDataSet(DataSet* dataset)
{
    int i;
    for (i = 0; i < dataset->rows; i++)
    {
        free(dataset->data[i]);
    }
    free(dataset->data);
    free(dataset);
}

static Matrix* dataSetToMatrix(DataSet* dataset)
{
    Matrix* convert = (Matrix*)malloc(sizeof(Matrix));
    convert->rows = dataset->rows;
    convert->cols = dataset->cols;
    convert->data = (float*)malloc(sizeof(float) * dataset->rows * dataset->cols);
    int i, j;
    for (i = 0; i < dataset->rows; i++)
    {
        for (j = 0; j < dataset->cols; j++)
	{
            setMatrix(convert, i, j, dataset->data[i][j]);
        }
    }

    return convert;
}

Matrix* createMatrix(size_t rows, size_t cols, float* data)
{
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = data;
    return matrix;
}

Matrix* createMatrixZeroes(size_t rows, size_t cols)
{
    assert(rows > 0 && cols > 0);
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    float* data = (float*)calloc(rows * cols, sizeof(float));
    matrix->data = data;
    return matrix;
}

static float getMatrix(Matrix* mat, size_t row, size_t col)
{
    return mat->data[row * mat->cols + col];
}

static void setMatrix(Matrix* mat, size_t row, size_t col, float val)
{
    mat->data[row * mat->cols + col] = val;
}

void copyValuesInto(Matrix* from, Matrix* to)
{
    assert(from->rows == to->rows && from->cols == to->cols);
    memcpy(to->data, from->data, sizeof(float) * to->rows * to->cols);
}

void printMatrix(Matrix* input)
{
    int i, j;
    for (i = 0; i < input->rows; i++)
    {
        printf("\n");
        for (j = 0; j < input->cols; j++)
	{
            printf("%.2f ", getMatrix(input, i, j));
        }
    }
    printf("\n");
}

void zeroMatrix(Matrix* orig)
{
    memset(orig->data, 0, orig->rows * orig->cols * sizeof(float));
}

Matrix* transpose(Matrix* orig)
{
    float* data = (float*)malloc(sizeof(float) * orig->rows * orig->cols);
    Matrix* transpose = createMatrix(orig->cols, orig->rows, data);
    int i, j;
    for (i = 0; i < orig->rows; i++)
    {
        for (j = 0; j < orig->cols; j++)
	{
            setMatrix(transpose, i, j, getMatrix(orig, i, j));
        }
    }
    return transpose;
}

void transposeInto(Matrix* orig, Matrix* origT)
{
    assert(orig->rows == origT->cols && orig->cols == origT->rows);
    int i, j;
    for (i = 0; i < orig->rows; i++)
    {
        for (j = 0; j < orig->cols; j++)
	{
            setMatrix(origT, j, i, getMatrix(orig, i, j));
        }
    }
}

Matrix* add(Matrix* A, Matrix* B)
{
    assert(A->rows == B->rows && A->cols == B->cols);
    float* data = (float*)malloc(sizeof(float) * A->rows * B->rows);
    Matrix* result = createMatrix(A->rows, A->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++)
    {
        for (j = 0; j < A->cols; j++)
	{
            setMatrix(result, i, j, getMatrix(B, i, j) + getMatrix(A, i, j));
        }
    }

    return result;
}

void addTo(Matrix* from, Matrix* to)
{
    assert(from->rows == to->rows && from->cols == to->cols);
    int i, j;
    for (i = 0; i < from->rows; i++)
    {
        for (j = 0; j < from->cols; j++)
	{
            setMatrix(to, i, j, getMatrix(from, i, j) + getMatrix(to, i, j));
        }
    }
}

// add B to each row of A
Matrix* addToEachRow(Matrix* A, Matrix* B){
    assert(A->cols == B->cols && B->rows == 1);
    float* data = (float*)malloc(sizeof(float) * A->rows * A->cols);
    Matrix* result = createMatrix(A->rows, A->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++){
        for (j = 0; j < A->cols; j++){
            setMatrix(result, i, j, getMatrix(A, i, j) + getMatrix(B, 0, j));
        }
    }
    return result;
}

void scalarMultiply(Matrix* orig, float c){
    int i, j;
    for (i = 0; i < orig->rows; i++){
        for (j = 0; j < orig->cols; j++){
            setMatrix(orig, i, j, getMatrix(orig, i, j) * c);
        }
    }
}

Matrix* multiply(Matrix* A, Matrix* B){
    assert(A->cols == B->rows);
    float* data = (float*)malloc(sizeof(float) * A->rows * B->cols);
    Matrix* result = createMatrix(A->rows, B->cols, data);
    zeroMatrix(result);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->rows, B->cols
    , A->cols, 1, A->data, A->cols, B->data, B->cols, 1, result->data, result->cols);
    return result;
}

void multiplyInto(Matrix* A, Matrix* B, Matrix* into)
{
    zeroMatrix(into);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->rows, B->cols
    , A->cols, 1, A->data, A->cols, B->data, B->cols, 1, into->data, into->cols);
    return;
}

Matrix* hadamard(Matrix* A, Matrix* B)
{
    assert(A->rows == B->rows && A->cols == B->cols);
    float* data = (float*)malloc(sizeof(float) * A->rows * A->cols);
    Matrix* result = createMatrix(A->rows, A->cols, data);
    int i, j;
    for (i = 0; i < A->rows; i++)
    {
        for (j = 0; j < A->cols; j++)
	{
            setMatrix(result, i, j, getMatrix(A, i, j) * getMatrix(B, i, j));
        }
    }
    return result;
}

void hadamardInto(Matrix* A, Matrix* B, Matrix* into)
{
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(A->rows == into->rows && A->cols == into->cols);
    int i, j;
    for (i = 0; i < A->rows; i++)
    {
        for (j = 0; j < A->cols; j++)
	{
            setMatrix(into, i, j, getMatrix(A, i, j) * getMatrix(B, i, j));
        }
    }
}

Matrix* copy(Matrix* orig)
{
    float* data = (float*)malloc(sizeof(float) * orig->rows * orig->cols);
    memcpy(data, orig->data, sizeof(float) * orig->cols * orig->rows);
    return createMatrix(orig->rows, orig->cols, data);
}

int equals(Matrix* A, Matrix* B)
{
    if (A->rows != B->rows)
    {
        return 0;
    }
    if (A->cols != B->cols)
    {
        return 0;
    }
    int i, j;
    for (i = 0; i < A->rows; i++)
    {
        for (j = 0; j < A->cols; j++)
	{
            if (getMatrix(A, i, j) != getMatrix(B, i, j))
	    {
                return 0;
            }
        }
    }
    return 1;
}

void destroyMatrix(Matrix* matrix)
{
    free(matrix->data);
    free(matrix);
}
