#pragma once

#include "compute/distance_cal.hpp"
#include "index/hnsw_imp.hpp"
#include "index/index_abs.hpp"
#include "nvme/nvme_allocator.hpp"
#include "nvme/nvme_manager.hpp"
#include "prune/prune_tool.hpp"
#include "runtime/cluster/cluster_extra.hpp"
#include "runtime/cluster/cluster_map.hpp"
namespace minihypervec
{
    namespace index
    {
        template <typename T>
        class HyperConstImp;

        template <>
        class HyperConstImp<int32_t> : public IndexAbs<int32_t>
        {
        public:
            collection::CollectionMeta m_collection_meta;
            std::unique_ptr<IndexAbs<int32_t>> centroid_index;
            std::unique_ptr<prune::PruningTool<int32_t>> prune_tool;
            runtime::ClusterMap m_cluster_map;
            runtime::ClusterExtra<int32_t> m_cluster_extra;

        public:
            HyperConstImp() = default;
            ~HyperConstImp() override = default;

        public:
            int32_t deployIndex(
                const std::string &collection_name,
                const resource::DeployTempResource &deploy_resource) override;
            int32_t loadIndex(const std::string &collection_name);
            int32_t searchKnn(const SearchParam &search_param,
                              const std::vector<int8_t> &query,
                              std::vector<std::pair<uint64_t, int32_t>> &res,
                              resource::SearchTempResource *search_resource)
                override;
            collection::IndexType getIndexType() const override;

        public:
            int32_t loadMetaDuringLoad(const std::string &collection_name);
            int32_t loadInmemIndexDuringLoad(const std::string &collection_name);
            int32_t loadClusterExtraDuringLoad(const std::string &collection_name);
            int32_t loadClusterMapDuringLoad(const std::string &collection_name);
            int32_t loadPruneToolDuringLoad(const std::string &collection_name);

        public:
            int32_t findNearestClustersDuringSearch(const SearchParam &search_param,
                                                    const std::vector<int8_t> &query,
                                                    std::vector<uint64_t> &probe_ids);
            int32_t launchLoadClustersFromNVMeDuringSearch(
                const std::vector<uint64_t> &probe_ids,
                const std::vector<runtime::ClusterStripe> &cluster_stripe,
                resource::SearchTempResource *search_resource,
                std::vector<std::pair<uint32_t, size_t>> &nvme_que_submits);
            int32_t prepareExtrasAndRankPairsDuringSearch(
                const std::vector<uint64_t> &probe_ids,
                resource::SearchTempResource *search_resource);
            int32_t pollNVMeCompletionsDuringSearch(
                const std::vector<std::pair<uint32_t, size_t>> &nvme_que_submits);
            int32_t calculateInnerProductDuringSearch(
                const std::vector<int8_t> &query, uint64_t total_clusters,
                resource::SearchTempResource *search_resource);

            int32_t rankFinalTopKDuringSearch(
                const std::vector<int8_t> &query, uint64_t total_clusters,
                const SearchParam &search_param,
                resource::SearchTempResource *search_resource,
                std::vector<std::pair<uint64_t, int32_t>> &res);

        public:
            int32_t loadCollectionMetaDuringDeploy(const std::string &collection_name);
            int32_t initClusterExtraDuringDeploy(const std::string &collection_name);
            int32_t allocateNVMeSpaceDuringDeploy(const std::string &collection_name);
            int32_t flushClustersToNVMeDuringDeploy(
                const std::string &collection_name,
                const resource::DeployTempResource &deploy_resource,
                uint64_t flush_clusters_per_batch = 256);
            int32_t syncIndexDuringDeploy(const std::string &collection_name);

        public:
            index::MiniHyperVecConstBuildParam *getBuildParamPtr() const;
        };

    } // namespace index
} // namespace minihypervec