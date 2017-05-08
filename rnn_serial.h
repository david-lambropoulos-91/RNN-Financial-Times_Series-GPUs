#ifndef __RNN_SERIAL_H
#define __RNN_SERIAL_H

#include "color_code_me.h"
#include "math.h"
#include "stdlib.h"
#include "malloc.h"
#include "activation.h"

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
	int num_layers;
	neuron** neurons;
	int manyLayers; 	
} rnn;

rnn* create_network();

void destroy_network(rnn* network);

void train_network(rnn* network);


#endif
