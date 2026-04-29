#pragma once

#include "index/params.hpp"
#include "meta_path.hpp"
#include "util/file/files_rw.hpp"

namespace minihypervec
{
  namespace collection
  {

    struct IndexMeta
    {
      IndexType index_type = IndexType::UNKNOWN;
      std::unique_ptr<index::BuildParam> build_param = nullptr;

      IndexMeta() = default;
      IndexMeta(const IndexMeta &o);
      IndexMeta &operator=(const IndexMeta &o);
      IndexMeta(IndexMeta &&) noexcept = default;
      IndexMeta &operator=(IndexMeta &&) noexcept = default;
    };

    struct CollectionMeta
    {
      using json = nlohmann::json;
      using ordered_json = nlohmann::ordered_json;

      std::string collection_name;
      VecType vec_type = VecType::UNKNOWN;
      IndexMeta index_meta;

      CollectionMeta() = default;
      CollectionMeta(const CollectionMeta &) = default;
      CollectionMeta &operator=(const CollectionMeta &) = default;
      CollectionMeta(CollectionMeta &&) noexcept = default;
      CollectionMeta &operator=(CollectionMeta &&) noexcept = default;

      void printCollectionMeta() const;
      static int32_t saveCollectionMeta(const std::string &collection_meta_path,
                                        const CollectionMeta &collection_meta);
      static int32_t loadCollectionMeta(const std::string &collection_meta_path,
                                        CollectionMeta &collection_meta);
      static int32_t loadCollectionMetaFromJson(const json &j,
                                                CollectionMeta &collection_meta);
      static int32_t saveCollectionMetaToJson(
          ordered_json &j, const CollectionMeta &collection_meta);
    };

  } // namespace collection
} // namespace minihypervec
