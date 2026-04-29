#pragma once

#include "index/params.hpp"
#include "root.hpp"
#include "util/file/dataset.hpp"
#include "util/file/files_rw.hpp"
#include "util/file/groundtruth.hpp"

namespace minihypervec {
namespace util {

namespace benchmark {
struct benchmark_param {
  std::string test_collection_name;
  std::string test_query_path;
  std::string test_groundtruth_path;
  collection::IndexType index_type;
  collection::VecType data_type;
  std::unique_ptr<index::SearchParam> search_param;
};
}  // namespace benchmark

int32_t loadQueryInt8(const std::string& file_path,
                      std::vector<std::vector<int8_t>>& out_vecs);

int32_t loadGroundTruth(const std::string& file_path,
                        std::vector<std::vector<uint64_t>>& out_ids,
                        std::vector<std::vector<float>>& out_dists);

int32_t loadBenchMarkConfig(const std::string& file_path,
                            benchmark::benchmark_param& param);
}  // namespace util
}  // namespace minihypervec