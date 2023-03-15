#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "control_flow.h"
#include "graph.h"
#include "optimizer.h"

std::vector<uint8_t> read_bytes(const std::string& fn) {
    std::ifstream st(fn, std::ios::binary);
    st.seekg(0, std::ios::end);
    auto size = st.tellg();
    st.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    st.read(data.data(), size);
    std::vector<uint8_t> out(size);
    for (int i = 0; i < size; i++) out[i] = data[i];
    return out;
}

std::vector<tensor_t> read_mnist_image(const std::string& fn) {
    auto contents = read_bytes(fn);
    std::vector<tensor_t> out;
    auto size = ((contents[4] * 256 + contents[5]) * 256 + contents[6]) * 256 +
                contents[7];
    for (std::size_t i = 0; i < size; i++) {
        auto start = 16 + i * 28 * 28;
        auto v = new_tensor({28 * 28, 1});
        for (std::size_t j = 0; j < 28 * 28; j++)
            v.data[j] = contents[start + j] / (real)256;
        out.push_back(v);
    }
    return out;
}

std::vector<tensor_t> read_mnist_label(const std::string& fn) {
    auto contents = read_bytes(fn);
    std::vector<tensor_t> out;
    auto size = ((contents[4] * 256 + contents[5]) * 256 + contents[6]) * 256 +
                contents[7];
    for (std::size_t i = 0; i < size; i++) {
        auto start = 8 + i;
        auto v = new_tensor({10, 1});
        for (std::size_t j = 0; j < 10; j++) v.data[j] = 0;
        v.data[contents[start]] = 1;
        out.push_back(v);
    }
    return out;
}

void print_image(std::ostream& st, tensor_t t) {
    auto size = shape_to_size(t.shape);
    for (std::size_t i = 0; i < 28; i++) {
        for (std::size_t j = 0; j < 28; j++)
            st << (t.data[i * 28 + j] > 0.5 ? '#' : ' ');
        st << std::endl;
    }
}

tensor_t read_tensor(const std::string& fn) {
    std::ifstream st(fn);
    shape_t shape;
    while (true) {
        std::size_t x;
        st >> x;
        if (!x) break;
        shape.push_back(x);
    }
    auto size = shape_to_size(shape);
    auto ret = new_tensor(shape);
    for (std::size_t i = 0; i < size; i++) st >> ret.data[i];
    return ret;
}

void write_tensor(const std::string& fn, tensor_t tensor) {
    std::ofstream st(fn);
    for (auto s : tensor.shape) st << s << " ";
    st << "\n";
    auto size = shape_to_size(tensor.shape);
    for (std::size_t i = 0; i < size; i++) st << tensor.data[i] << " ";
    st << "\n";
}

int main() {
    const int l1 = 28 * 28, l2 = 300, l3 = 100, l4 = 10;
    graph_t g;
    g.rng.seed(23809713);
    auto x = g.add_placeholder({l1, 1}, "x");
    auto w1 = g.add_parameter({l2, l1}, "w1");
    auto b1 = g.add_parameter({l2, 1}, "b1");
    auto w1_x = multiply(w1, x);
    auto w1_x_b1 = add(w1_x, b1);
    auto y1 = relu(w1_x_b1);
    auto w2 = g.add_parameter({l3, l2}, "w2");
    auto b2 = g.add_parameter({l3, 1}, "b2");
    auto w2_y1 = multiply(w2, y1);
    auto w2_y1_b2 = add(w2_y1, b2);
    auto y2 = relu(w2_y1_b2);
    auto w3 = g.add_parameter({l4, l3}, "w3");
    auto b3 = g.add_parameter({l4, 1}, "b3");
    auto w3_y2 = multiply(w3, y2);
    auto w3_y2_b3 = add(w3_y2, b3, "w3_y2_b3");
    auto yp = softmax(w3_y2_b3, "yp");
    auto y = g.add_placeholder({l4, 1}, "y");
    auto log_yp = log_tensor(yp, "log_yp");
    auto log_yp_t = reshape(log_yp, {1, l4});
    auto neg_loss = multiply(log_yp_t, y, "neg_loss");
    auto loss = multiply(-1, neg_loss, "loss");
    normal_init(w1, sqrt(1.0 / l1));
    zero_init(b1);
    normal_init(w2, sqrt(1.0 / l2));
    zero_init(b2);
    normal_init(w3, sqrt(1.0 / l3));
    zero_init(b3);
    g.finalize();

    auto training_images = read_mnist_image("train-images.idx3-ubyte");
    auto training_labels = read_mnist_label("train-labels.idx1-ubyte");
    auto test_images = read_mnist_image("t10k-images.idx3-ubyte");
    auto test_labels = read_mnist_label("t10k-labels.idx1-ubyte");
    spdlog::info("Read MNIST data successfully");

    std::vector<input_t> training_set;
    for (int i = 0; i < training_images.size(); i++) {
        input_t input;
        input["x"] = training_images[i];
        input["y"] = training_labels[i];
        training_set.push_back(input);
    }

    for (int i = 0; i < 5; i++) {
        print_image(std::cout, training_images[i]);
        std::cout << std::max_element(training_labels[i].data,
                                      training_labels[i].data + 10) -
                         training_labels[i].data
                  << std::endl;
    }

    int t = 0;
    while (true) {
        t++;
        spdlog::info("Starting SGD iteration {}", t);
        real rate = 0.001;
        sgd_iter(&g, training_set, rate, (real)0.01);
        // spdlog::info("iteration ended");

        int correct = 0, selected = 0;
        real sum = 0;
        for (int i = 0; i < test_images.size(); i++) {
            if (g.uniform_dist(g.rng) > (t % 100 ? 0.01 : 1)) continue;
            selected++;
            input_t input;
            input["x"] = test_images[i];
            input["y"] = test_labels[i];
            g.compute(input);
            auto p = yp->value.data;
            auto pp = test_labels[i].data;
            sum += *(loss->value.data);
            // print_matrix(std::cout, w1->value);
            if (std::max_element(p, p + 10) - p ==
                std::max_element(pp, pp + 10) - pp) {
                correct++;
                std::cout << std::max_element(p, p + 10) - p;
                // spdlog::info("{} correct out of {}", correct, selected);
            } else {
                std::cout << "x";
            }
        }
        std::cout << std::endl;
        spdlog::info("{} correct out of {} ({}%), average loss {}", correct,
                     selected, correct * 100 / selected, sum / selected);
        if (t % 10000 == 0) {
            for (auto u : g.nodes) {
                if (!u->parameterp) continue;
                auto name = std::string("epoch_") + std::to_string(t) + "-" +
                            u->name + ".tsr";
                write_tensor(name, u->value);
            }
        }
    }
}
