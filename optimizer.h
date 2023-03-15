#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "config.h"
#include "graph.h"

// TODO: Adam

void sgd_iter(
    graph_t *graph,
    const std::vector<std::unordered_map<std::string, tensor_t>>& training_set,
    real learning_rate, real p = 1);
