#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "config.h"
#include "graph.h"

struct optimizer {
    virtual void iter(std::size_t t, const std::vector<input_t>& training_set,
                      real learning_rate) = 0;
};

struct sgd : public optimizer {
    graph_t* g;
    sgd(graph_t* g) : g(g) {}
    virtual void iter(std::size_t t, const std::vector<input_t>& training_set,
                      real learning_rate) override;
};

struct adam : public optimizer {
    graph_t* g;
    real b1, b2, e;
    std::vector<tensor_t> m, v;
    adam(graph_t* g_, real b1_ = 0.9, real b2_ = 0.999, real e_ = 1e-8);
    virtual void iter(std::size_t t, const std::vector<input_t>& training_set,
                      real learning_rate) override;
};

// TODO: Adam

void sgd_iter(
    graph_t* graph,
    const std::vector<std::unordered_map<std::string, tensor_t>>& training_set,
    real learning_rate, real p = 1);
