#include "graph.h"

#include <queue>

#include "control_flow.h"

std::size_t graph_t::size() { return nodes.size(); }

void graph_t::finalize() {
    order.clear();
    std::queue<std::size_t> q;
    std::vector<std::size_t> deg(size());
    for (std::size_t i = 0; i < size(); i++) {
        deg[i] = nodes[i]->dependencies.size();
        if (!deg[i]) {
            q.push(i);
        }
    }
    while (q.size()) {
        order.push_back(q.front());
        for (auto successor : nodes[q.front()]->successors) {
            if (!(--deg[successor])) q.push(successor);
        }
        q.pop();
    }
}

void graph_t::compute(const input_t& input) {
    for (std::size_t i = 0; i < nodes.size(); i++) {
        auto& u = *(nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            u.adjoint.data[j] = 0;
        }
    }
    for (auto node : order) nodes[node]->compute(input);
}

placeholder* graph_t::add_placeholder(const shape_t& shape,
                                      const std::string& name) {
    auto u = new placeholder;
    u->value = new_tensor(shape);
    u->adjoint = new_tensor(shape);
    u->acc = new_tensor(shape);
    u->name = name;
    u->index = nodes.size();
    u->graph = this;
    u->parameterp = false;
    nodes.push_back(u);
    name_tbl[name] = u;
    return u;
}

parameter* graph_t::add_parameter(const shape_t& shape,
                                  const std::string& name) {
    auto u = new parameter;
    u->value = new_tensor(shape);
    u->adjoint = new_tensor(shape);
    u->acc = new_tensor(shape);
    u->name = name;
    u->index = nodes.size();
    u->graph = this;
    u->parameterp = true;
    nodes.push_back(u);
    name_tbl[name] = u;
    return u;
}

void placeholder::compute(const input_t& input) {
    if (!input.count(name))
        PANIC("Input for placeholder node {} is unspecified", name);
    if (input.find(name)->second.shape != value.shape)
        PANIC("Input for placeholder node {} has a wrong shape", name);
    auto in = input.find(name)->second;
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++) value.data[i] = in.data[i];
}

void placeholder::differentiate() {}

void parameter::compute(const input_t& input) {}

void parameter::differentiate() {}

multiplication* multiply(node_t* a, node_t* b, const std::string& name) {
    if (a->graph != b->graph)
        PANIC("Nodes {} and {} are not from the same graph", a->name, b->name);
    if (a->value.shape.size() != 2) PANIC("Node {} is not a matrix", a->name);
    if (b->value.shape.size() != 2) PANIC("Node {} is not a matrix", b->name);
    if (a->value.shape[1] != b->value.shape[0])
        PANIC(
            "Shapes of nodes {} and {} do not match for matrix multiplication",
            a->name, b->name);
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    b->successors.push_back(i);
    auto u = new multiplication;
    shape_t shape{a->value.shape[0], b->value.shape[1]};
    u->value = new_tensor(shape);
    u->adjoint = new_tensor(shape);
    u->acc = new_tensor(shape);
    u->dependencies = {a->index, b->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->graph->name_tbl[name] = u;
    return u;
}

addition* add(node_t* a, node_t* b, const std::string& name) {
    if (a->graph != b->graph)
        PANIC("Nodes {} and {} are not from the same graph", a->name, b->name);
    if (a->value.shape != b->value.shape)
        PANIC("Nodes {} and {} have different shapes", a->name, b->name);
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    b->successors.push_back(i);
    auto u = new addition;
    u->value = new_tensor(a->value.shape);
    u->adjoint = new_tensor(a->value.shape);
    u->acc = new_tensor(a->value.shape);
    u->dependencies = {a->index, b->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->graph->name_tbl[name] = u;
    return u;
}

log_node* log_tensor(node_t* a, const std::string& name) {
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new log_node;
    u->value = new_tensor(a->value.shape);
    u->adjoint = new_tensor(a->value.shape);
    u->acc = new_tensor(a->value.shape);
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->graph->name_tbl[name] = u;
    return u;
}

reshape_node* reshape(node_t* a, const shape_t& shape,
                      const std::string& name) {
    if (shape_to_size(a->value.shape) != shape_to_size(shape))
        PANIC("Size mismatch between nodes {} and {}", a->name, name);
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new reshape_node;
    u->value = new_tensor(shape);
    u->adjoint = new_tensor(shape);
    u->acc = new_tensor(shape);
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->graph->name_tbl[name] = u;
    return u;
}

relu_node* relu(node_t* a, const std::string& name) {
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new relu_node;
    u->value = new_tensor(a->value.shape);
    u->adjoint = new_tensor(a->value.shape);
    u->acc = new_tensor(a->value.shape);
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->graph->name_tbl[name] = u;
    return u;
}

softmax_node* softmax(node_t* a, const std::string& name) {
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new softmax_node;
    u->value = new_tensor(a->value.shape);
    u->adjoint = new_tensor(a->value.shape);
    u->acc = new_tensor(a->value.shape);
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->graph->name_tbl[name] = u;
    return u;
}

scalar_multiplication* multiply(real a, node_t* b, const std::string& name) {
    auto i = b->graph->nodes.size();
    b->successors.push_back(i);
    auto u = new scalar_multiplication;
    u->value = new_tensor(b->value.shape);
    u->adjoint = new_tensor(b->value.shape);
    u->acc = new_tensor(b->value.shape);
    u->dependencies = {b->index};
    u->index = i;
    u->name = name;
    u->graph = b->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->a = a;
    u->graph->name_tbl[name] = u;
    return u;
}

void normal_init(node_t* u, real coeff) {
    if (!u->parameterp) PANIC("Initializing non-parameter node {}", u->name);
    auto size = shape_to_size(u->value.shape);
    for (std::size_t i = 0; i < size; i++)
        u->value.data[i] = coeff * u->graph->normal_dist(u->graph->rng);
}

void zero_init(node_t* u) {
    if (!u->parameterp) PANIC("Initializing non-parameter node {}", u->name);
    auto size = shape_to_size(u->value.shape);
    for (std::size_t i = 0; i < size; i++) u->value.data[i] = 0;
}

void multiplication::compute(const input_t& input) {
    node_t& a = *(graph->nodes[dependencies[0]]);
    node_t& b = *(graph->nodes[dependencies[1]]);
    auto n = a.value.shape[0];
    auto m = a.value.shape[1];
    auto l = b.value.shape[1];
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < l; j++) {
            value.data[i * l + j] = 0;
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        auto il = i * l, im = i * m;
        for (std::size_t j = 0; j < m; j++) {
            auto jl = j * l;
            auto t = a.value.data[im + j];
            for (std::size_t k = 0; k < l; k++) {
                value.data[il + k] += t * b.value.data[jl + k];
            }
        }
    }
}

void multiplication::differentiate() {
    node_t& a = *(graph->nodes[dependencies[0]]);
    node_t& b = *(graph->nodes[dependencies[1]]);
    auto n = a.value.shape[0];
    auto m = a.value.shape[1];
    auto l = b.value.shape[1];
    for (std::size_t i = 0; i < n; i++) {
        auto im = i * m, il = i * l;
        for (std::size_t j = 0; j < m; j++) {
            auto jl = j * l;
            for (std::size_t k = 0; k < l; k++) {
                a.adjoint.data[im + j] += adjoint.data[il + k] * b.value.data[jl + k];
                b.adjoint.data[jl + k] += a.value.data[im + j] * adjoint.data[il + k];
            }
        }
    }
}

void addition::compute(const input_t& input) {
    node_t& a = *(graph->nodes[dependencies[0]]);
    node_t& b = *(graph->nodes[dependencies[1]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++)
        value.data[i] = a.value.data[i] + b.value.data[i];
}

void addition::differentiate() {
    node_t& a = *(graph->nodes[dependencies[0]]);
    node_t& b = *(graph->nodes[dependencies[1]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++) {
        a.adjoint.data[i] += adjoint.data[i];
        b.adjoint.data[i] += adjoint.data[i];
    }
}

void log_node::compute(const input_t& input) {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++) value.data[i] = log(a.value.data[i]);
}

void log_node::differentiate() {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++)
        a.adjoint.data[i] += adjoint.data[i] / a.value.data[i];
}

void reshape_node::compute(const input_t& input) {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++) value.data[i] = a.value.data[i];
}

void reshape_node::differentiate() {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++) a.adjoint.data[i] += adjoint.data[i];
}

void relu_node::compute(const input_t& input) {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++)
        value.data[i] = std::max((real)0, a.value.data[i]);
}

void relu_node::differentiate() {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++)
        a.adjoint.data[i] += a.value.data[i] > 0 ? adjoint.data[i] : 0;
}

void softmax_node::compute(const input_t& input) {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    real sum = 0, max = *std::max_element(a.value.data, a.value.data + size);
    for (std::size_t i = 0; i < size; i++)
        sum += value.data[i] = exp(a.value.data[i] - max);
    for (std::size_t i = 0; i < size; i++) value.data[i] /= sum;
}

void softmax_node::differentiate() {
    node_t& a = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++) {
        for (std::size_t j = 0; j < size; j++) {
            a.adjoint.data[i] +=
                adjoint.data[j] *
                (i == j ? value.data[i] - value.data[i] * value.data[i]
                        : -value.data[i] * value.data[j]);
        }
    }
}

void scalar_multiplication::compute(const input_t& input) {
    node_t& b = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++) value.data[i] = a * b.value.data[i];
}

void scalar_multiplication::differentiate() {
    node_t& b = *(graph->nodes[dependencies[0]]);
    auto size = shape_to_size(value.shape);
    for (std::size_t i = 0; i < size; i++)
        b.adjoint.data[i] += a * adjoint.data[i];
}
