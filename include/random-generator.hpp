// Declare a random generator class which uses Mersenne Twister.

#pragma once

// Include requirements.
#include <algorithm>
#include <random>

class RandomGenerator {
private:
    std::mt19937 gen;

public:
    explicit RandomGenerator(uint seed = std::random_device{}());

    // Create a random integer in the set {0, 1, ... , m - 1}.
    uint uniform(uint m);

    // Create a random permutation of the input vector.
    void shuffle(std::vector<uint>& v);
};
