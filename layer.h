#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <stdlib.h>
#include <stddef.h>
#include "activation.h"

// possible types of layers in a network
typedef enum LAYER_TYPE_ 
{
    INPUT,
    HIDDEN,
    OUTPUT
} LAYER_TYPE;

// input matrix will continue to store values
// even after activation occurs, so it is best
// to think of it simply as a store rather than
// strictly as input
typedef struct Layer_ 
{
    LAYER_TYPE type;
    size_t size;
    Activation activation;
    Matrix* input; // (num_examples x size)
} Layer;

// represents a link between two layers
typedef struct Connection_ {
    Layer* from;
    Layer* to;
    Matrix* weights; // (from_size x to_size)
    Matrix* bias; // (1 x to_size)
} Connection;

// returns layer given metadata and configuration
static Layer* createLayer(LAYER_TYPE type, size_t size, Activation activation);

// creates a connection and creates weights and bias matrices
static Connection* createConnection(Layer* from, Layer* to);

// initializes weights and biases within connection
static void initializeConnection(Connection* connection);

// applies activation function to each input in layer
static void activateLayer(Layer* layer);

// frees layer and its input
static void destroyLayer(Layer* layer);

// frees connection, its weights, and biases, but not its layers
static void destroyConnection(Connection* connection);

// returns layer given metadata and configuration
static Layer* createLayer(LAYER_TYPE type, size_t size, Activation activation);

#endif
