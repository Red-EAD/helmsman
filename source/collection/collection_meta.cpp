#include "collection/collection_meta.hpp"

namespace minihypervec
{
  namespace collection
  {
    using json = nlohmann::json;
    using ordered_json = nlohmann::ordered_json;

    IndexMeta::IndexMeta(const IndexMeta &o)
    {
      index_type = o.index_type;
      if (o.build_param)
      {
        build_param = o.build_param->clone();
      }
    }

    IndexMeta &IndexMeta::operator=(const IndexMeta &o)
    {
      if (this != &o)
      {
        index_type = o.index_type;
        if (o.build_param)
        {
          build_param = o.build_param->clone();
        }
        else
        {
          build_param = nullptr;
        }
      }
      return *this;
    }

    void CollectionMeta::printCollectionMeta() const
    {
      std::cout << "CollectionMeta: " << std::endl;
      std::cout << "  Collection Name: " << collection_name << std::endl;
      std::cout << "  Vector Type: " << stringFromVecType(vec_type) << std::endl;
      std::cout << "  Index Type: " << stringFromIndexType(index_meta.index_type)
                << std::endl;
      if (index_meta.build_param)
      {
        index_meta.build_param->printBuildParam();
      }
    }

    int32_t CollectionMeta::loadCollectionMetaFromJson(
        const json &j, CollectionMeta &collection_meta)
    {
      try
      {
        collection_meta.collection_name =
            j.at("collection_name").get<std::string>();
        collection_meta.vec_type =
            vecTypeFromString(j.at("vec_type").get<std::string>());
        collection_meta.index_meta.index_type =
            indexTypeFromString(j.at("index_type").get<std::string>());
        if (j.contains("build_param"))
        {
          switch (collection_meta.index_meta.index_type)
          {
          case IndexType::HV_CONST:
            collection_meta.index_meta.build_param =
                std::make_unique<index::MiniHyperVecConstBuildParam>();
            collection_meta.index_meta.build_param->from_json(
                j.at("build_param"));
            break;
          default:
            std::cerr << "Unsupported index type in loadCollectionMetaFromJson: "
                      << static_cast<uint32_t>(
                             collection_meta.index_meta.index_type)
                      << std::endl;
            return -1;
          }
        }
        else
        {
          std::cerr << "No build_param found in loadCollectionMetaFromJson for "
                       "collection: "
                    << collection_meta.collection_name << std::endl;
          return -1;
        }
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error---CollectionMeta::loadCollectionMetaFromJson: JSON "
                     "parse/load "
                     "failed: "
                  << e.what() << "\n";
        return -1;
      }
      return 0;
    }

    int32_t CollectionMeta::saveCollectionMetaToJson(
        ordered_json &j, const CollectionMeta &collection_meta)
    {
      j["collection_name"] = collection_meta.collection_name;
      j["vec_type"] = stringFromVecType(collection_meta.vec_type);
      j["index_type"] = stringFromIndexType(collection_meta.index_meta.index_type);
      if (collection_meta.index_meta.build_param)
      {
        j["build_param"] = collection_meta.index_meta.build_param->to_json();
      }
      else
      {
        std::cerr << "No build_param to save for collection: "
                  << collection_meta.collection_name << std::endl;
      }
      return 0;
    }

    int32_t CollectionMeta::saveCollectionMeta(
        const std::string &collection_meta_path,
        const CollectionMeta &collection_meta)
    {
      ordered_json j;
      CollectionMeta::saveCollectionMetaToJson(j, collection_meta);
      std::string data = j.dump(2, ' ', false);
      data.push_back('\n');
      return util::persist_string_atomic_fsync(collection_meta_path, data);
    }

    int32_t CollectionMeta::loadCollectionMeta(
        const std::string &collection_meta_path, CollectionMeta &collection_meta)
    {
      std::string content;
      int32_t rc = util::read_file_to_string(collection_meta_path, content);
      if (rc != 0)
        return rc;
      json j = json::parse(content);
      rc = CollectionMeta::loadCollectionMetaFromJson(j, collection_meta);
      return rc;
    }

  } // namespace collection
} // namespace minihypervec