// Define the required functions correponding to ordinary and smooth hinge loss.
// Here gamma = 0 corresponds to ordinary hinge loss.

#include "run.hpp"

// Compute primal objective function according to smooth hing loss.
double primal(DataSet& trainData, SimpleVector& w, double lambda, double gamma) {       
    double P = 0.0; 
    uint trainSize = trainData.size;
    for (uint i = 0; i < trainSize; i++) {
        double y = trainData.label[i];
        SparseVector& x = trainData.domain[i];
        double ywx = y * (w * x);
        if (ywx < 1.0) {
            if (ywx < 1 - gamma) {
                P += 1 - ywx - 0.5 * gamma;
            } else {
                P += 0.5 * (1 - ywx) * (1 - ywx) / gamma;
            }
        }
    }
    P /= trainSize;
    P += lambda * w.snorm();
    return P;
}

// Compute dual objective function according to conjugate of smooth hinge loss.
double dual(DataSet& trainData, SimpleVector& w, SimpleVector& a, double lambda, double gamma) {  
    double D = 0.0;
    uint trainSize = trainData.size;
    for (uint i = 0; i < trainSize; i++) {
        double y = trainData.label[i];
        D += y * a[i] - 0.5 * gamma * a[i] * a[i];
    }
    D /= trainSize;
    D -=  lambda * w.snorm();
    return D;
}

// Compute step size for SDCA according to smooth hinge loss.
double step(DataSet& trainData, SimpleVector& w, SimpleVector& a, uint r, double lambda, double gamma) {
    uint trainSize = trainData.size;
    double y = trainData.label[r];
    SparseVector& x = trainData.domain[r];
    double snorm = trainData.domainSquaredNorm[r];
    double ywx = y * (w * x);
    double delta = ((1 - ywx - y * a[r] * gamma) / ((snorm / (2 * lambda * trainSize)) + gamma)) + y * a[r];
    delta = y * std::max(0.0, std::min(1.0, delta)) - a[r];
    return delta;
}

// Compute a subgradient of smooth hinge loss for SGD.
double subgradient(DataSet& trainData, SimpleVector& w, uint r, double gamma) {
    double y = trainData.label[r];
    SparseVector& x = trainData.domain[r];
    double ywx = y * (w * x);
    double g = 0.0;
    if (ywx < 1.0) {
        if (ywx <= 1 - gamma) {
            g = -y;
        } else {
            g = -y * (1 - ywx) / gamma;
        }
    }
    return g;
}
