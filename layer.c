#include "layer.h"

Layer* createLayer(LAYER_TYPE type, size_t size, Activation activation)
{
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->type = type;
    layer->size = size;
    layer->activation = activation;
    float* row = (float*)malloc(sizeof(float) * size);
    layer->input = createMatrix(1, size, row);
    return layer;
}

Connection* createConnection(Layer* from, Layer* to)
{
    Connection* connection = (Connection*)malloc(sizeof(Connection));
    connection->from = from;
    connection->to = to;
    float* weights_data = (float*)malloc(sizeof(float) * to->size * from->size);
    connection->weights = createMatrix(from->size, to->size, weights_data);
    float* bias_data = (float*)malloc(sizeof(float) * to->size);
    connection->bias = createMatrix(1, to->size, bias_data);
    return connection;
}

// weights are guassian random using fan-in method
// // biases are 0
void initializeConnection(Connection* connection)
{
    int i, j;
    for (i = 0; i < connection->bias->cols; i++)
    {
        setMatrix(connection->bias, 0, i, 0);
    }
    for (i = 0; i < connection->weights->rows; i++)
    {
        for (j = 0; j < connection->weights->cols; j++)
	{
            int neuronsIn = connection->weights->rows;
            setMatrix(connection->weights, i, j, box_muller() / sqrt(neuronsIn));
        }
    }
}

// assuming input of layer is filled with raw input,
// calls activation function on each of them, and
// modifies in-place
void activateLayer(Layer* layer)
{
    if (layer->activation != NULL)
    {
        layer->activation(layer->input);
    }
}

void destroyLayer(Layer* layer)
{
    destroyMatrix(layer->input);
    free(layer);
}

void destroyConnection(Connection* connection)
{
    destroyMatrix(connection->weights);
    destroyMatrix(connection->bias);
    free(connection);
}
