// Declare the SDCA and SGD solvers.

#pragma once

// Include requirements.
#include "data-set.hpp"


// Declare proper types for function pointers.
typedef double (*PrimalFncPtr)(DataSet& trainData, SimpleVector& w, double lambda, double gamma);
typedef double (*DualFncPtr)(DataSet& trainData, SimpleVector& w, SimpleVector& a, double lambda, double gamma);

struct Function {
    PrimalFncPtr primal;
    DualFncPtr dual;
    double lambda;
    double gamma;
};

typedef double (*Subgradient)(DataSet& trainData, SimpleVector& w, uint r, double gamma);
typedef double (*Step)(DataSet& trainData, SimpleVector& w, SimpleVector& a, uint r, double lambda, double gamma);

struct UpdateRule {
    Subgradient subgradient;
    Step step;
};

typedef void (*AlgorithmPtr)(Function f, UpdateRule update, Data& data, double tolerance, uint MAX_EPOCH, std::vector<Result>& result);


// Declare algorithms signiture.
void SDCA(Function f, UpdateRule update, Data& data, double tolerance, uint MAX_EPOCH, std::vector<Result>& result);

void SDCAperm(Function f, UpdateRule update, Data& data, double tolerance, uint MAX_EPOCH, std::vector<Result>& result);

void SDCAcyclic(Function f, UpdateRule update, Data& data, double tolerance, uint MAX_EPOCH, std::vector<Result>& result);

void SGD(Function f, UpdateRule update, Data& data, double tolerance, uint MAX_EPOCH, std::vector<Result>& result);
