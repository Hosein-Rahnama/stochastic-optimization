// Implements the methods of the SimpleVector class.

#include "simple-vector.hpp"


SimpleVector::SimpleVector() { };

SimpleVector::SimpleVector(uint d) : data(std::vector<double>(d)) { }

void SimpleVector::scale(double s) { 
    SimpleVector& w = *this;
    uint d = w.size();
    for (uint i = 0; i < d; i++) {
        w[i] *= s;
    }
}

uint SimpleVector::size() {
    return data.size();
}

void SimpleVector::resize(uint n) {
    return data.resize(n);
}

double& SimpleVector::operator[](uint i) {
    return data[i];
}

void SimpleVector::add(SparseVector& x, double s) {
    SimpleVector& w = *this;
    for (IndexValuePair& pair : x) {
        w[pair.index] += s * pair.value;
    }
}

void SimpleVector::print(std::ostream& os) {
    SimpleVector& w = *this;
    uint d = w.size();
    for (uint i = 0; i < d; i++) {
        os << w[i] << " "; 
    }
    os << std::endl;
}

double SimpleVector::snorm() {
    SimpleVector& w = *this;
    double sum = 0;
    uint d = w.size();
    for (uint i = 0; i < d; i++) {
        sum += w[i] * w[i];
    }
    return sum;
}

double SimpleVector::operator*(SparseVector& x) {
    SimpleVector& w = *this;
    double sum = 0.0;
    for (IndexValuePair& pair : x) {
        sum += (pair.value * w[pair.index]);
    }
    return sum;
}
