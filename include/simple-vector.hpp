// Declares an a simple vector class.

#pragma once

// Include requirements.
#include "sparse-vector.hpp"


class SimpleVector {
private:
    std::vector<double>  data;

public:
    // Default constructor does nothing.
    SimpleVector ();

    // Construct a simple vector of zeros with dimension d.
    SimpleVector(uint d);

    // Return size of a simple vector.
    uint size();

    // Resize the a simple vector.
    void resize(uint n);

    // Access ith element.
    double& operator[](uint i);

    // Multiply the simple vector by s.
    void scale(double s);
  
    // Add a simple vector to a sparse vector.
    void add(SparseVector& x, double s);

    // Compute inner product of a simple vector and a sparse vector.
    double operator*(SparseVector& x);

    // Returns the squared norm of a simple vector.
    double snorm();

    void print(std::ostream& os = std::cout);
};
