#include "spann_index.h"

namespace spann
{
    namespace
    {
        uint32_t GetFixedClusterSizeOrThrow(
            const std::vector<std::vector<int32_t>> &clusters,
            int expected_size,
            const char *name)
        {
            if (clusters.empty())
            {
                throw std::runtime_error(std::string(name) + " is empty");
            }
            if (expected_size <= 0)
            {
                throw std::runtime_error(std::string(name) + " expected size must be > 0");
            }

            const size_t fixed_size = static_cast<size_t>(expected_size);
            for (size_t i = 0; i < clusters.size(); ++i)
            {
                if (clusters[i].size() != fixed_size)
                {
                    throw std::runtime_error(
                        std::string(name) + " size mismatch at cluster " + std::to_string(i) +
                        ": expect " + std::to_string(fixed_size) +
                        ", got " + std::to_string(clusters[i].size()));
                }
            }

            return static_cast<uint32_t>(fixed_size);
        }
    }

    void SpannIndex::saveHeadIndex(const std::string &head_index_path) const
    {
        std::cout << "Saving head HNSW index to " << head_index_path << " ..." << std::endl;
        head_hnsw_->saveIndex(head_index_path);
    }

    void SpannIndex::saveClusterIds(const std::string &cluster_ids_path) const
    {
        std::cout << "Saving cluster IDs to " << cluster_ids_path << " ..." << std::endl;

        const uint32_t cluster_size =
            GetFixedClusterSizeOrThrow(cluster_ids_, fill_target_size_, "cluster_ids");

        std::ofstream fout(cluster_ids_path, std::ios::binary);
        if (!fout)
        {
            throw std::runtime_error("Failed to open cluster IDs output file: " + cluster_ids_path);
        }

        const uint64_t num_clusters = static_cast<uint64_t>(cluster_ids_.size());
        fout.write(reinterpret_cast<const char *>(&num_clusters), sizeof(uint64_t));
        fout.write(reinterpret_cast<const char *>(&cluster_size), sizeof(uint32_t));

        for (const auto &ids : cluster_ids_)
        {
            std::vector<uint64_t> ids_64(ids.size());
            for (size_t i = 0; i < ids.size(); ++i)
            {
                ids_64[i] = static_cast<uint64_t>(ids[i]);
            }
            fout.write(reinterpret_cast<const char *>(ids_64.data()),
                       static_cast<std::streamsize>(ids_64.size()) * sizeof(uint64_t));
        }

        if (!fout)
        {
            throw std::runtime_error("Failed to write cluster IDs to: " + cluster_ids_path);
        }
    }

    void SpannIndex::saveNorms(const std::string &norm_path) const
    {
        std::cout << "Saving cluster norms to " << norm_path << " ..." << std::endl;

        const uint32_t cluster_size =
            GetFixedClusterSizeOrThrow(norms_, fill_target_size_, "cluster_norms");

        std::ofstream fout(norm_path, std::ios::binary);
        if (!fout)
        {
            throw std::runtime_error("Failed to open norm output file: " + norm_path);
        }

        const uint64_t num_clusters = static_cast<uint64_t>(norms_.size());
        fout.write(reinterpret_cast<const char *>(&num_clusters), sizeof(uint64_t));
        fout.write(reinterpret_cast<const char *>(&cluster_size), sizeof(uint32_t));

        for (const auto &norm_vec : norms_)
        {
            fout.write(reinterpret_cast<const char *>(norm_vec.data()),
                       static_cast<std::streamsize>(norm_vec.size()) * sizeof(int32_t));
        }

        if (!fout)
        {
            throw std::runtime_error("Failed to write cluster norms to: " + norm_path);
        }
    }
} // namespace spann
