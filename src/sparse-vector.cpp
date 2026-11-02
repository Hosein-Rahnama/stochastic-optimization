// Implements the methods of the SparseVector class.

#include "sparse-vector.hpp"


SparseVector::SparseVector() { };

SparseVector::SparseVector(std::istringstream& is, uint n) : data(std::vector<IndexValuePair>(n)) {
    uint index;
    double value;
    std::vector<IndexValuePair>& v = (*this).data;
    for (uint i = 0; i < n; i++) {
        is >> index >> value;
        v[i] = IndexValuePair(index, value);
    }
}

SparseVector::Iterator SparseVector::begin() { 
    return data.begin();
}

SparseVector::Iterator SparseVector::end() { 
    return data.end();
}

uint SparseVector::maxIndex() {
    uint d = 0;
    SparseVector& v = *this;
    for (IndexValuePair& pair : v) {
        if (pair.index > d) {
            d = pair.index;
        }
    }
    return d;
}

void SparseVector::shiftIndex(int i) {
    SparseVector& v = *this;
    for (IndexValuePair& pair : v) {
        pair.index += i;
    }
}

double SparseVector::snorm() {
    double sum = 0.0;
    SparseVector& v = *this;
    for (IndexValuePair& pair : v) {
        sum += pair.value * pair.value;
    }
    return sum;
}

void SparseVector::print(std::ostream& os) {
    SparseVector& v = *this;
    for (IndexValuePair& pair : v) {
        os << "(" << pair.index << "," << pair.value << ") ";
    }
    os << std::endl;
}
