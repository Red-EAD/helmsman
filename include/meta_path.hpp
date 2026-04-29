#pragma once
#include "root.hpp"
#include "util/file/files_rw.hpp"
namespace minihypervec
{
    inline static constexpr std::string_view g_path_config = "/mnt/service/minihyper-vec/setup/path_config.json";
    struct PathConfig
    {
        std::string nvme_meta_path = "";
        std::string release_index_path = "";

        static PathConfig *getInstance();
        int32_t init(const std::string &path);
        int32_t save(const std::string &path);
        void printConfig() const;
    };

    namespace nvme
    {
        inline constexpr std::string_view nvme_meta_filename = "nvme_meta.json";
        std::string getNVMeMetaPath() noexcept;
    } // namespace nvme

    namespace release
    {
        namespace constants
        {
            inline constexpr std::string_view collection_meta_filename =
                "collection_meta.json";
            inline constexpr std::string_view hardware_meta_filename = "hardware_meta.json";

            inline constexpr std::string_view index_meta_filename = "_index_meta.json";
            inline constexpr std::string_view rawdata_filename = "_rawdata.bin";
            inline constexpr std::string_view cluster_ids_filename = "_cluster_ids.bin";
            inline constexpr std::string_view cluster_norms_filename = "_cluster_norms.bin";
            inline constexpr std::string_view centroids_index_filename =
                "_centroids_index.bin";

            inline constexpr std::string_view cluster_map_filename = "_cluster_map.bin";
            inline constexpr std::string_view cluster_extra_ids_filename =
                "_cluster_extra_ids.bin";
            inline constexpr std::string_view cluster_extra_norms_filename =
                "_cluster_extra_norms.bin";

            std::string getHardwareMetaPath() noexcept;

            std::string getIndexMetaPath(const std::string &collection_name) noexcept;
            std::string getRawdataPath(const std::string &collection_name) noexcept;
            std::string getClusterIDsPath(const std::string &collection_name) noexcept;
            std::string getClusterNormsPath(const std::string &collection_name) noexcept;
            std::string getCentroidsIndexPath(const std::string &collection_name) noexcept;

            std::string getClusterExtraIDsPath(const std::string &collection_name) noexcept;
            std::string getClusterExtraNormsPath(
                const std::string &collection_name) noexcept;
            std::string getClusterMapPath(const std::string &collection_name) noexcept;
        } // namespace constants
    } // namespace release

    void printAuthorInfo() noexcept;

} // namespace minihypervec