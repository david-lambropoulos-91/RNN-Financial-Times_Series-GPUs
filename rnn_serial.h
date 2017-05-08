#ifndef __RNN_SERIAL_H
#define __RNN_SERIAL_H

#include "color_code_me.h"
#include "math.h"
#include "stdlib.h"
#include "malloc.h"
#include "activation.h"
#include "layer.h"
#include "matrix.h"
#include "float.h"
#include "assert.h"

typedef struct neuron_
{
	int layer; // What layer am I in?
	double* weight;	// Table of weights for incoming synapses
	struct neuron_ ** synapses_in;	// table of pointers to the incoing synapses
	int num_synapses_in;	// Number of incoming synapses
	double bias;
	double value;
	double value_prev;
	double error;
	double error_prev;
} neuron;

typedef struct rnn_
{
	size_t numLayers;
	Layer** layers;
	size_t numConnections;
	Connection** connections;	
} rnn;

// constructor to create a network given sizes and functions
// hiddenSizes is an array of sizes, where hiddenSizes[i] is the size
// of the ith hidden layer
// hiddenActivations is an array of activation functions,
// where hiddenActivations[i] is the function of the ith hidden layer
static rnn* createNetwork(size_t numFeatures, size_t numHiddenLayers, size_t* hiddenSizes, Activation* hiddenActivations, size_t numOutputs, Activation outputActivation);

void destroyNetwork(rnn* network);

// will propagate input through entire network
// result will be stored in input field of last layer
// input should be a matrix where each row is an input
static void forwardPass(rnn* network, Matrix* input);

// will propagate input through entire network
// result will be stored in input field of last layer
// input should be a dataset where each row is an input
static void forwardPassDataSet(rnn* network, DataSet* input);

// calculate the cross entropy loss between two datasets with 
// optional regularization (must provide network if using regularization)
// [normal cross entropy] + 1/2(regStrength)[normal l2 reg]
static float crossEntropyLoss(rnn* network, Matrix* prediction, DataSet* actual, float regularizationStrength);

// calculate the mean squared error between two datasets with 
// optional regularization (must provide network if using regularization)
// 1/2[normal mse] + 1/2(regStrength)[normal l2 reg]
static float meanSquaredError(rnn* network, Matrix* prediction, DataSet* actual, float regularizationStrength);

// return matrix of network output
static Matrix* getOuput(rnn* network);

// returns indices corresponding to highest-probability classes for each
// example previously inputted
// assumes final output is in the output layer of network
static int* predict(rnn* network);

// return accuracy (num_correct / num_total) of network on predictions
static float accuracy(rnn* network, DataSet* data, DataSet* classes);

// frees network, its layers, and its connections
static void destroyNetwork(rnn* network);

// write network configuration to a file
static void saveNetwork(rnn* network, char* path);

// read network configuration from a file
static rnn* readNetwork(char* path);
#endif
