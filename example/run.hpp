// Declare the functions that the user should provide to use SDCA and SGD algorithms.

#include "data-set.hpp"

// Compute primal objective function according to the loss function.
double primal(DataSet& trainData, SimpleVector& w, double lambda, double gamma);

// Compute dual objective function according to conjugate of the loss function.
double dual(DataSet& trainData, SimpleVector& w, SimpleVector& a, double lambda, double gamma);

// Compute step size for SDCA according to the loss function.
double step(DataSet& trainData, SimpleVector& w, SimpleVector& a, uint r, double lambda, double gamma);

// Compute subgradient of the loss function for SGD.
double subgradient(DataSet& trainData, SimpleVector& w, uint r, double gamma);
