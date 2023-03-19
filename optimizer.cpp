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

void sgd::iter(std::size_t t, const std::vector<input_t>& training_set,
               real learning_rate) {
    if (shape_to_size((*(g->nodes.rbegin()))->value.shape) != 1)
        PANIC("The last node of graph is not a scalar");
    // spdlog::info("cleared adjoint");
    for (std::size_t i = 0; i < g->nodes.size(); i++) {
        auto& u = *(g->nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            u.acc.data[j] = 0;
        }
    }
    std::size_t selected = 0;
    for (const auto& example : training_set) {
        selected++;
        // spdlog::info("selected {} for now", selected);
        g->compute(example);
        *((*g->nodes.rbegin())->adjoint.data) = 1;
        for (auto it = g->order.rbegin(); it != g->order.rend(); it++) {
            auto& u = *(g->nodes[*it]);
            u.differentiate();
        }
        for (std::size_t i = 0; i < g->nodes.size(); i++) {
            auto& u = *(g->nodes[i]);
            auto size = shape_to_size(u.value.shape);
            for (std::size_t j = 0; j < size; j++) {
                u.acc.data[j] += u.adjoint.data[j];
            }
        }
    }
    for (std::size_t i = 0; i < g->nodes.size(); i++) {
        if (!g->nodes[i]->parameterp) continue;
        auto& u = *(g->nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            u.value.data[j] -= learning_rate * u.acc.data[j] / selected;
        }
    }
}

adam::adam(graph_t* g_, real b1_, real b2_, real e_) {
    g = g_;
    b1 = b1_;
    b2 = b2_;
    e = e_;
    for (std::size_t i = 0; i < g->nodes.size(); i++) {
        if (g->nodes[i]->parameterp) {
            m.push_back(new_tensor(g->nodes[i]->value.shape));
            v.push_back(new_tensor(g->nodes[i]->value.shape));
        } else {
            m.push_back(new_tensor({1}));
            v.push_back(new_tensor({1}));
        }
        zero_init(*m.rbegin());
        zero_init(*v.rbegin());
    }
}

void adam::iter(std::size_t t, const std::vector<input_t>& training_set,
                real learning_rate) {
    if (shape_to_size((*(g->nodes.rbegin()))->value.shape) != 1)
        PANIC("The last node of graph is not a scalar");
    // spdlog::info("cleared adjoint");
    for (std::size_t i = 0; i < g->nodes.size(); i++) {
        auto& u = *(g->nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            u.acc.data[j] = 0;
        }
    }
    std::size_t selected = 0;
    for (const auto& example : training_set) {
        selected++;
        // spdlog::info("selected {} for now", selected);
        g->compute(example);
        *((*g->nodes.rbegin())->adjoint.data) = 1;
        for (auto it = g->order.rbegin(); it != g->order.rend(); it++) {
            auto& u = *(g->nodes[*it]);
            u.differentiate();
        }
        for (std::size_t i = 0; i < g->nodes.size(); i++) {
            auto& u = *(g->nodes[i]);
            auto size = shape_to_size(u.value.shape);
            for (std::size_t j = 0; j < size; j++) {
                u.acc.data[j] += u.adjoint.data[j];
            }
        }
    }
    for (std::size_t i = 0; i < g->nodes.size(); i++) {
        if (!g->nodes[i]->parameterp) continue;
        auto& u = *(g->nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            u.acc.data[j] /= selected;
        }
    }
    // acc: g_t
    for (std::size_t i = 0; i < g->nodes.size(); i++) {
        if (!g->nodes[i]->parameterp) continue;
        auto& u = *(g->nodes[i]);
        auto size = shape_to_size(u.value.shape);
        for (std::size_t j = 0; j < size; j++) {
            m[i].data[j] = b1 * m[i].data[j] + (1 - b1) * u.acc.data[j];
            v[i].data[j] =
                b2 * v[i].data[j] + (1 - b2) * u.acc.data[j] * u.acc.data[j];
            u.value.data[j] -= learning_rate * m[i].data[j] / (1 - pow(b1, t)) /
                               (e + sqrt(v[i].data[j] / (1 - pow(b2, t))));
        }
    }
}
