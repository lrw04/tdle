#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <random>

#include "tensor.h"

struct graph_t;

using input_t = std::unordered_map<std::string, tensor_t>;

struct node_t {
    tensor_t value, adjoint, acc;
    graph_t* graph;
    std::size_t index;
    std::vector<std::size_t> dependencies, successors;
    std::string name;
    bool parameterp;
    virtual void compute(const input_t& input) = 0;
    virtual void differentiate() = 0;
};

struct placeholder : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

struct parameter : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

struct multiplication : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

multiplication *multiply(node_t *a, node_t *b, const std::string& name = "");

struct addition : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

addition *add(node_t *a, node_t *b, const std::string& name = "");

struct log_node : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

log_node *log_tensor(node_t *a, const std::string& name = "");

struct reshape_node : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

reshape_node *reshape(node_t *a, const shape_t& shape, const std::string& name = "");

struct relu_node : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

relu_node *relu(node_t *a, const std::string& name = "");

struct softmax_node : public node_t {
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

softmax_node *softmax(node_t *a, const std::string& name = "");

struct scalar_multiplication : public node_t {
    real a;
    virtual void compute(const input_t& input) override;
    virtual void differentiate() override;
};

scalar_multiplication *multiply(real a, node_t *b, const std::string& name = "");

// TODO: Sigmoid, Convolution, scalar multiplication

struct graph_t {
    std::vector<node_t*> nodes;
    std::vector<std::size_t> order;
    std::size_t size();
    std::mt19937_64 rng;
    std::uniform_real_distribution<real> uniform_dist;
    std::normal_distribution<real> normal_dist;
    void finalize();
    void compute(const input_t& input);
    std::unordered_map<std::string, node_t*> name_tbl;
    placeholder *add_placeholder(const shape_t& shape, const std::string& name);
    parameter *add_parameter(const shape_t& shape, const std::string& name = "");
};

void normal_init(node_t *u, real coeff = 1);

void zero_init(node_t *u);

void zero_init(tensor_t u);
