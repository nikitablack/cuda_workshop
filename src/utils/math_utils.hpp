#pragma once

#include <cstdlib>

inline bool fuzzy_compare(float a, float b, float epsilon = 0.01f)
{
    return std::abs(a - b) < epsilon;
}