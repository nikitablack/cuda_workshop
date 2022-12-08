#pragma once

#include <chrono>
#include <iostream>

class Timer
{
public:
    Timer() : m_start{std::chrono::high_resolution_clock::now()} {}

    ~Timer()
    {
        auto end = std::chrono::high_resolution_clock::now();

        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start).count();

        std::cout << t << '\n';
    }

private:
    std::chrono::high_resolution_clock::time_point m_start;
};