// Solves the regularized loss minimization problem for a given loss function.

// Include requirements.
#include <format>
#include <print>

#include "solver.hpp"
#include "run.hpp"


int main(int argc, char** argv) {
    // Parse command line.
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << "algorithm dataFile testFile experimentFile" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string algorithm(argv[1]);
    std::string dataFile(argv[2]);
    std::string testFile(argv[3]);
    std::string experimentFile(argv[4]);

    // Read the data. 
    DataSet trainData(dataFile);
    DataSet testData(testFile);
    if (testData.dimension > trainData.dimension) {
        trainData.dimension = testData.dimension;
    }
    Data data = {trainData, testData};

    // Set primal and dual objective functions.
    Function f = {primal, dual};

    // Determine the algorithm.
    AlgorithmPtr alg;
    std::string type = "primal-dual";
    if (algorithm == "SDCA") {
        alg = SDCA;
    } else if (algorithm == "SDCA-perm") {
        alg = SDCAperm;
    } else if (algorithm == "SDCA-cyclic") {
        alg = SDCAcyclic;
    } else if (algorithm == "SGD") {
        alg = SGD;
        type = "primal";
    } else {
        std::cerr << "Use one of the algorithms SDCA, SDCA-perm, SDCA-cylic, SGD.";
        exit(EXIT_FAILURE);
    }

    // Set the update rule.
    UpdateRule update = {subgradient, step};

    // Setup experiments.
    std::ifstream experiment(experimentFile.c_str());
    if (!experiment.good()) {
        std::cerr << "error w/ " << experimentFile << std::endl;
        exit(EXIT_FAILURE);
    }

    uint num_experiments = 0;
    experiment >> num_experiments;
    for (uint e = 0; e < num_experiments; e++) {
        // Determine experiment parameters.
        double tolerance;
        uint MAX_EPOCH;
        experiment >> f.lambda >> f.gamma >> tolerance >> MAX_EPOCH;

        // Run the algoirthm.
        std::vector<Result> result;
        alg(f, update, data, tolerance, MAX_EPOCH, result);

        // Output results.
        std::print("Experiment {:<2}\n\n", e + 1);
        std::print("{:<8} {:<7} {:<10} {:<10}\n", "lambda", "gamma", "tolerance", "MAX_EPOCH");
        std::print("{:<8.6f} {:<7.5f} {:<10.8f} {:<10}\n\n", f.lambda, f.gamma, tolerance, MAX_EPOCH);
        report(result, type);
        std::print("\n\n");
    }
    return EXIT_SUCCESS;
}
