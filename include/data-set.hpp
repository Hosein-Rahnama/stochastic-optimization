// Declares some utilities for handling datasets.

#pragma once

// Include requirements.
#include "sparse-vector.hpp"
#include "simple-vector.hpp"


class DataSet {
public:
    uint size;
    uint dimension;
    std::vector<SparseVector> domain;
    SimpleVector label;
    SimpleVector domainSquaredNorm;

public:
    // Read data in SVM-Light format from data_filename and store it.
    DataSet(std::string& data_filename);

    // Compute 0-1 error for a given weight vector.
    double errorZeroOne(SimpleVector& w);

    // Compute hinge error for a given weight vector.
    double errorHinge(SimpleVector& w);

    // Compute squared norm of doamin points.
    void snorm();
};

struct Data {
    DataSet& train;
    DataSet& test;
};

class Result {
public:
    uint epoch;
    double primal;
    double dual;
    double gap;
    double error;

public:
    Result(uint epoch, double primal, double error);
    Result(uint epoch, double primal, double dual, double gap, double error);
};

void report(std::vector<Result>& result, std::string type = "primal-dual");
