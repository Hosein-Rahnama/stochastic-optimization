// Implements the methods of DataSet class.

// Include requirements.
#include <format>
#include <print>

#include "data-set.hpp"


DataSet::DataSet(std::string& data_filename) : size(0), dimension(0) {
    // Open data file.
    std::ifstream data_file(data_filename.c_str());
    if (!data_file.good()) {
        std::cerr << "error w/ " << data_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Determine the size of data set to allocate memory for it.
    std::string line;
    while (getline(data_file, line)) {
        // Ignore lines which begin with #.
        if (line[0] == '#') {
            continue;
        }
        size++;
    }
    domain.resize(size);
    label.resize(size);

    // Reset the file stream.
    data_file.clear();
    data_file.seekg(0, std::ios::beg);
  
    // Read the data file in SVM-Light format.
    uint i = 0;
    while (getline(data_file, line)) {
        // Ignore lines which begin with #.
        if (line[0] == '#') {
            continue;
        }

        // Erase what comes after #.
        size_t pos = line.find('#');
        if (pos < line.size()) {
            line.erase(pos);
        }

        // Replace : with white space.
        int n = 0;
        for (size_t pos = 0; pos < line.size(); pos++) {
            if (line[pos] == ':') {
                n++; 
                line[pos] = ' ';
            }
        }

        // Read from the string.
        std::istringstream is(line);
        double y;
        is >> y;
        label[i] = y;
        if (label[i] != 1 && label[i] != -1) {
            std::cerr << "Invalid label in SVM-light format." << std::endl;
            exit(EXIT_FAILURE);
        }
        SparseVector x(is, n);
        x.shiftIndex(-1);
        domain[i] = x;
        uint current_dimension = x.maxIndex() + 1;
        if (current_dimension > dimension) {
            dimension = current_dimension;
        }
        i++;
    }
    data_file.close();
}

double DataSet::errorZeroOne(SimpleVector& w) {
    double error = 0.0;
    uint size = label.size();
    for (uint i = 0; i < size; i++) {
        double y = label[i];
        SparseVector x = domain[i];
        if (y * (w * x) < 0.0) {
            error += 1.0;
        }
    }
    error /= size;
    return error;
}

double DataSet::errorHinge(SimpleVector& w) {
    double error = 0.0;
    uint size = label.size();
    for (uint i = 0; i < size; i++) {
        double y = label[i];
        SparseVector x = domain[i];
        double hinge_loss = std::max(0.0, 1 - y * (w * x));
        error += hinge_loss;
    }
    error /= size;
    return error;
}

void DataSet::snorm() {
    domainSquaredNorm.resize(size);
    for (uint i = 0; i < size; i++) {
        domainSquaredNorm[i] = domain[i].snorm();
    }
}

Result::Result(uint epoch, double primal, double error) :
    epoch(epoch), primal(primal), dual(0), gap(0), error(error) {
}

Result::Result(uint epoch, double primal, double dual, double gap, double error) :
    epoch(epoch), primal(primal), dual(dual), gap(gap), error(error) {
}

void report(std::vector<Result>& result, std::string type) {
    if (type == "primal-dual") {
        std::print("{:>5} {:>10} {:>10} {:>10} {:>10}\n", "epoch", "primal", "dual", "gap", "error");
        for (Result& r : result) {
            std::print("{:>5} {:>10.8f} {:>10.8f} {:>10.8f} {:>10.8f}\n", r.epoch, r.primal, r.dual, r.gap, r.error);
        }
    }
    if (type == "primal") {
        std::print("{:>5} {:>10} {:>10}\n", "epoch", "primal", "error");
        for (Result& r : result) {
            std::print("{:>5} {:>10.8f} {:>10.8f}\n", r.epoch, r.primal, r.error);
        }
    }
}
