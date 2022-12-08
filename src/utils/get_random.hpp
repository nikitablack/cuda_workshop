#pragma once

#include <random>

inline float get_random()
{
    static std::random_device rd{};
    static std::mt19937 gen{rd()};
    static std::uniform_real_distribution<float> distr(0.0f, 1.0f);

    return distr(gen);
}