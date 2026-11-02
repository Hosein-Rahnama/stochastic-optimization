// Declares a very simple sparse vector class based on the STL vector.

#pragma once

// Include requirements.
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


typedef unsigned int uint;

struct IndexValuePair {
public:
    uint index;
    double value;
};

class SparseVector {
private:
    std::vector<IndexValuePair> data;

public:
    typedef std::vector<IndexValuePair>::iterator Iterator;

public:
    // Default constructor does nothing.
    SparseVector();

    // Constructor makes a vector from an istringstream with n elements.
    SparseVector(std::istringstream& is, uint n);

    // define the following begin and end methods for loop support.
    Iterator begin();
    Iterator end();

    // Return the maximal non-zero index.
    uint maxIndex();

    // Shift all of the indices by i.
    void shiftIndex(int i);

    // Returns the squared l2 norm of the vector.
    double snorm();

    // Print the content of a sparse vector to an output stream.
    void print(std::ostream& os = std::cout);
};
