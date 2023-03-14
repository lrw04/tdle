#include "optimizer.h"

#include <spdlog/spdlog.h>

#include <iostream>

#include "control_flow.h"

void print_matrix(std::ostream& st, tensor_t t) {
    if (t.shape.size() != 2) PANIC("Not a matrix");
    for (std::size_t i = 0; i < t.shape[0]; i++) {
        for (std::size_t j = 0; j < t.shape[1]; j++) st << t({i, j}) << " ";
        st << std::endl;
    }
}

void print_matrix_transposed(std::ostream& st, tensor_t t) {
    if (t.shape.size() != 2) PANIC("Not a matrix");
    for (std::size_t i = 0; i < t.shape[1]; i++) {
        for (std::size_t j = 0; j < t.shape[0]; j++) st << t({j, i}) << " ";
        st << std::endl;
    }
}

void sgd_iter(
    graph_t* graph,
    const std::vector<std::unordered_map<std::string, tensor_t>>& training_set,
    real learning_rate, real p) {
    if (shape_to_size((*(graph->nodes.rbegin()))->value.shape) != 1)
        PANIC("The last node of graph is not a scalar");
    // spdlog::info("cleared adjoint");
    for (std::size_t i = 0; i < graph->nodes.size(); i++) {
        auto& u = *(graph->nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            u.acc.data[j] = 0;
        }
    }
    std::size_t selected = 0;
    for (const auto& example : training_set) {
        if (graph->uniform_dist(graph->rng) > p) continue;
        selected++;
        // spdlog::info("selected {} for now", selected);
        graph->compute(example);
        *((*graph->nodes.rbegin())->adjoint.data) = 1;
        for (auto it = graph->order.rbegin(); it != graph->order.rend(); it++) {
            auto& u = *(graph->nodes[*it]);
            u.differentiate();
        }
        for (std::size_t i = 0; i < graph->nodes.size(); i++) {
            auto& u = *(graph->nodes[i]);
            auto size = shape_to_size(u.value.shape);
            for (std::size_t j = 0; j < size; j++) {
                u.acc.data[j] += u.adjoint.data[j];
            }
        }
        // print_matrix_transposed(std::cout, graph->name_tbl["y"]->value);
        // print_matrix_transposed(std::cout, graph->name_tbl["log_yp"]->adjoint);
        // print_matrix_transposed(std::cout, graph->name_tbl["yp"]->value);
        // print_matrix_transposed(std::cout, graph->name_tbl["yp"]->adjoint);
        // std::cout << std::endl;
    }
    for (std::size_t i = 0; i < graph->nodes.size(); i++) {
        if (!graph->nodes[i]->parameterp) continue;
        auto& u = *(graph->nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            // std::cerr << u.adjoint.data[j] << std::endl;
            u.value.data[j] -= learning_rate * u.acc.data[j] / selected;
        }
    }
}
