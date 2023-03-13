#include "graph.h"

int main() {
    graph_t g;
    auto x = g.add_placeholder({28 * 28, 1}, "x");
    auto w1 = g.add_parameter({300, 28 * 28}, "w1");
    auto b1 = g.add_parameter({300, 1}, "b1");
    auto w1_x = multiply(w1, x);
    auto w1_x_b1 = add(w1_x, b1);
    auto y1 = relu(w1_x_b1);
    auto w2 = g.add_parameter({100, 300}, "w2");
    auto b2 = g.add_parameter({100, 1}, "b2");
    auto w2_y1 = multiply(w2, y1);
    auto w2_y1_b2 = add(w2_y1, b2);
    auto y2 = relu(w2_y1_b2);
    auto w3 = g.add_parameter({10, 100}, "w3");
    auto b3 = g.add_parameter({10, 1}, "b3");
    auto w3_y2 = multiply(w3, y2);
    auto w3_y2_b3 = add(w3_y2, b3);
    auto yp = softmax(w3_y2_b3);
    auto y = g.add_placeholder({10, 1}, "y");
    auto log_yp = log_tensor(yp);
    auto log_yp_t = reshape(log_yp, {1, 10});
    auto neg_entropy = multiply(log_yp_t, y);
    auto loss = multiply(-1, neg_entropy);
    normal_init(w1);
    normal_init(b1);
    normal_init(w2);
    normal_init(b2);
    normal_init(w3);
    normal_init(b3);
}
