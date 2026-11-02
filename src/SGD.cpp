// Implemements Stochastic Gradient Descent (SGD) for a smooth hinge-loss.

// Include requirements.
#include "sparse-vector.hpp"
#include "simple-vector.hpp"
#include "data-set.hpp"
#include "solver.hpp"
#include "random-generator.hpp"


void SGD(Function f, UpdateRule update, Data& data, double tolerance, uint MAX_EPOCH, std::vector<Result>& result) {
    // Determine data set size and the dimension of the underlying space.
    DataSet& trainData = data.train;
    DataSet& testData = data.test;
    uint trainSize = trainData.label.size();
    uint dimension = trainData.dimension;

    // Initialize weight vector.
    SimpleVector w(dimension);

    // Convergence loop.
    RandomGenerator rng;
    uint t = 1, epoch_counter = 0;
    while (epoch_counter < MAX_EPOCH) {
        epoch_counter++;
        for (uint i = 0; i < trainSize; i++, t++) {
            // Create a random index.
            uint r = rng.uniform(trainSize);

            // Update weight vector w.
            w.scale(1.0 - 1.0 / t);
            double g = update.subgradient(trainData, w, r, f.gamma);
            if (g != 0.0) {
                SparseVector& x = trainData.domain[r];
                w.add(x, -g / (2 * f.lambda * t));
            }
        }

        // Compute primal objective function.
        double P = f.primal(trainData, w, f.lambda, f.gamma);
    
        // Save statistics.
        result.push_back(Result(epoch_counter, P, testData.errorZeroOne(w)));
    }
}
