#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>
#include "matrix.h"

typedef void (*Activation)(Matrix*);

#define MAX(a,b) (((a)>(b))?(a):(b))

static float sigmoidFunc( float z );

// derivative of sigmoid given output of sigmoid
static float sigmoidDeriv( float z ); 

static float tanHFunc( float z );

static float tanHDeriv( float z );

// raw ReLU function
static float reluFunc(float input);

// derivative of ReLU given output of ReLU
static float reluDeriv(float reluInput);

// applies sigmoid function to each entry of $input
static void sigmoid(Matrix* input);

// applies ReLU function to each entry of input
static void relu(Matrix* input);

// applues tanh function to each entry of input
static void tanH(Matrix* input);

// applies softmax function to each row of $input
static void softmax(Matrix* input);

// applies linear function to each row of $input
static void linear(Matrix* input);

// derivative of linear function given output of linear
static float linearDeriv(float linearInput);

// sample from the unit guassian distribution (mean = 0, variance = 1)
static float box_muller();

// return the string representation of activation function
static const char* getFunctionName(Activation func);

// return the activation function corresponding to a name
static Activation getFunctionByName(const char* name);

// return derivative of activation function
static float (*activationDerivative(Activation func))(float);

#endif
