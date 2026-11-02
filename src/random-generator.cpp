// Implement methods of RandomGenerator class.

#include "random-generator.hpp"


RandomGenerator::RandomGenerator(uint seed) : gen(seed) { }

uint RandomGenerator::uniform(uint m) {
    std::uniform_int_distribution<int> dist(0, m - 1);
    return dist(gen);
}

void RandomGenerator::shuffle(std::vector<uint>& v) {
    std::shuffle(v.begin(), v.end(), gen);
}
