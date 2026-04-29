#pragma once

#include "root.hpp"

namespace minihypervec {
namespace util {
int32_t persist_string_atomic_fsync(const std::string& path,
                                    const std::string& data);
int32_t read_file_to_string(const std::string& path, std::string& out);
}  // namespace util
}  // namespace minihypervec