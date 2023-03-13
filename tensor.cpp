#include "tensor.h"

#include "control_flow.h"

std::size_t tensor_t::get_offset(const index_t& index) const {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < offsets.size(); i++)
        offset += offsets[i] * index[i];
    return offset;
}

real tensor_t::operator()(const index_t& index) const {
    return data[get_offset(index)];
}

real& tensor_t::operator()(const index_t& index) {
    return data[get_offset(index)];
}

offsets_t shape_to_offsets(shape_t shape) {
    if (shape.empty()) return shape;
    for (auto it = shape.rbegin() + 1; it != shape.rend(); it++)
        *it = *it * *(it - 1);
    for (std::size_t i = 0; i < shape.size() - 1; i++) shape[i] = shape[i + 1];
    shape[shape.size() - 1] = 1;
    return shape;
}

std::size_t shape_to_size(const shape_t& shape) {
    std::size_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

tensor_t new_tensor(const shape_t& shape) {
    real* data = new real[shape_to_size(shape)];
    tensor_t tensor;
    tensor.data = data;
    tensor.shape = shape;
    tensor.offsets = shape_to_offsets(shape);
    return tensor;
}
