#pragma once

#include <vector>

#include "config.h"

using index_t = std::vector<std::size_t>;
using offsets_t = std::vector<std::size_t>;
using shape_t = std::vector<std::size_t>;

struct tensor_t {
    offsets_t offsets;
    shape_t shape;
    real *data;
    std::size_t get_offset(const index_t& index) const;
    real operator()(const index_t& index) const;
    real& operator()(const index_t& index);
};

offsets_t shape_to_offsets(shape_t shape);

std::size_t shape_to_size(const shape_t& shape);

tensor_t new_tensor(const shape_t& shape);
