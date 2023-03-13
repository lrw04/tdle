#include "graph.h"

#include <queue>

#include "control_flow.h"

std::size_t graph_t::size() { return nodes.size(); }

void graph_t::compute_order() {
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
    for (auto node : order) nodes[node]->compute(input);
}

placeholder* graph_t::add_placeholder(const shape_t& shape,
                                      const std::string& name) {
    auto out = new_tensor(shape);
    auto adjoint = new_tensor(shape);
    auto u = new placeholder;
    u->value = out;
    u->adjoint = adjoint;
    u->name = name;
    u->index = nodes.size();
    u->graph = this;
    u->parameterp = false;
    nodes.push_back(u);
    return u;
}

parameter* graph_t::add_parameter(const shape_t& shape,
                                  const std::string& name) {
    auto out = new_tensor(shape);
    auto adjoint = new_tensor(shape);
    auto u = new parameter;
    u->value = out;
    u->adjoint = adjoint;
    u->name = name;
    u->index = nodes.size();
    u->graph = this;
    u->parameterp = true;
    nodes.push_back(u);
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
    shape_t out_shape{a->value.shape[0], b->value.shape[1]};
    auto out = new_tensor(out_shape);
    auto adjoint = new_tensor(out_shape);
    u->value = out;
    u->adjoint = adjoint;
    u->dependencies = {a->index, b->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
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
    auto out = new_tensor(a->value.shape);
    auto adjoint = new_tensor(a->value.shape);
    u->value = out;
    u->adjoint = adjoint;
    u->dependencies = {a->index, b->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    return u;
}

log_node* log_tensor(node_t* a, const std::string& name) {
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new log_node;
    auto out = new_tensor(a->value.shape);
    auto adjoint = new_tensor(a->value.shape);
    u->value = out;
    u->adjoint = adjoint;
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    return u;
}

reshape_node* reshape(node_t* a, const shape_t& shape,
                      const std::string& name) {
    if (shape_to_size(a->value.shape) != shape_to_size(shape))
        PANIC("Size mismatch between nodes {} and {}", a->name, name);
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new reshape_node;
    auto out = new_tensor(shape);
    auto adjoint = new_tensor(shape);
    u->value = out;
    u->adjoint = adjoint;
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    return u;
}

relu_node* relu(node_t* a, const std::string& name) {
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new relu_node;
    auto out = new_tensor(a->value.shape);
    auto adjoint = new_tensor(a->value.shape);
    u->value = out;
    u->adjoint = adjoint;
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    return u;
}

softmax_node* softmax(node_t* a, const std::string& name) {
    auto i = a->graph->nodes.size();
    a->successors.push_back(i);
    auto u = new softmax_node;
    auto out = new_tensor(a->value.shape);
    auto adjoint = new_tensor(a->value.shape);
    u->value = out;
    u->adjoint = adjoint;
    u->dependencies = {a->index};
    u->index = i;
    u->name = name;
    u->graph = a->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    return u;
}

scalar_multiplication* multiply(real a, node_t* b, const std::string& name) {
    auto i = b->graph->nodes.size();
    b->successors.push_back(i);
    auto u = new scalar_multiplication;
    auto out = new_tensor(b->value.shape);
    auto adjoint = new_tensor(b->value.shape);
    u->value = out;
    u->adjoint = adjoint;
    u->dependencies = {b->index};
    u->index = i;
    u->name = name;
    u->graph = b->graph;
    u->parameterp = false;
    u->graph->nodes.push_back(u);
    u->a = a;
    return u;
}

void normal_init(node_t* u) {
    if (!u->parameterp) PANIC("Initializing non-parameter node {}", u->name);
    auto size = shape_to_size(u->value.shape);
    for (std::size_t i = 0; i < size; i++)
        u->value.data[i] = u->graph->normal_dist(u->graph->rng);
}

void multiplication::compute(const input_t& input) {
    node_t& a = *(graph->nodes[dependencies[0]]);
    node_t& b = *(graph->nodes[dependencies[1]]);
    for (std::size_t i = 0; i < value.shape[0]; i++) {
        for (std::size_t j = 0; j < value.shape[1]; j++) {
            value({i, j}) = 0;
        }
    }
    for (std::size_t i = 0; i < a.value.shape[0]; i++) {
        for (std::size_t j = 0; j < a.value.shape[1]; j++) {
            for (std::size_t k = 0; k < b.value.shape[1]; k++) {
                value({i, k}) += a.value({i, j}) * b.value({j, k});
            }
        }
    }
}

void multiplication::differentiate() {
    node_t& a = *(graph->nodes[dependencies[0]]);
    node_t& b = *(graph->nodes[dependencies[1]]);
    for (std::size_t i = 0; i < a.value.shape[0]; i++) {
        for (std::size_t j = 0; j < a.value.shape[1]; j++) {
            for (std::size_t k = 0; k < b.value.shape[1]; k++) {
                a.adjoint({i, j}) += adjoint({i, k}) * b.value({j, k});
                b.adjoint({j, k}) += adjoint({i, k}) * a.value({i, j});
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
        a.adjoint.data[i] += 1 / a.value.data[i];
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
    real sum = 0;
    for (std::size_t i = 0; i < size; i++)
        sum += value.data[i] = exp(a.value.data[i]);
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
