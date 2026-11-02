// Implemements Stochastic Dual Coordinate Ascent (SDCA) for a smooth hinge-loss.
// This version is for a random permutation per epoch instead of stochastic choice per iteration.

// Include requirements.
#include "sparse-vector.hpp"
#include "simple-vector.hpp"
#include "data-set.hpp"
#include "solver.hpp"
#include "random-generator.hpp"


void SDCAperm(Function f, UpdateRule update, Data& data, double tolerance, uint MAX_EPOCH, std::vector<Result>& result) {
    // Determine data set size and the dimension of the underlying space.
    DataSet& trainData = data.train;
    DataSet& testData = data.test;
    uint trainSize = trainData.label.size();
    uint dimension = trainData.dimension;

    // Initialize weight vector and dual vector.
    SimpleVector w(dimension);
    SimpleVector a(trainSize);

    // Define some auxiliary variables.
    double D = 0.0, P = 1.0, duality_gap = 1.0;

    // Compute squared norm of domain points.
    trainData.snorm();

    // Initialize a vector of indices.
    RandomGenerator rng;
    std::vector<uint> indices(trainSize);
    for (uint i = 0; i < trainSize; i++) {
        indices[i] = i;
    }

    // Convergence loop.
    uint epoch_counter = 0;
    while (duality_gap > tolerance && epoch_counter < MAX_EPOCH) {
        epoch_counter++;

        // Create a random permutation.
        rng.shuffle(indices);
        for (uint i = 0; i < trainSize; i++) {
            // Choose a random example.
            uint r = indices[i];

            // Update weight vector w and dual vector a.
            double delta = update.step(trainData, w, a, r, f.lambda, f.gamma);
            if (delta != 0.0) {
                SparseVector& x = trainData.domain[r];
                a[r] = a[r] + delta;
                w.add(x, delta / (2 * f.lambda * trainSize));
            }
        }

        // Compute primal and dual objective functions and the duality gap.
        P = f.primal(trainData, w, f.lambda, f.gamma);
        D = f.dual(trainData, w, a, f.lambda, f.gamma);
        duality_gap = P - D;

        // Save statistics.
        result.push_back(Result(epoch_counter, P, D, duality_gap, testData.errorZeroOne(w)));
    }
}
