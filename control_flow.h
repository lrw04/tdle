#pragma once

#include <spdlog/spdlog.h>

#define PANIC(...)                     \
    do {                               \
        spdlog::critical(__VA_ARGS__); \
        exit(1);                       \
    } while (false)
