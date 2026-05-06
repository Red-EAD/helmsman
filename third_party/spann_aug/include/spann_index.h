#pragma once

#include "root.h"
#include "hnswlib/hnswlib.hpp"

namespace spann
{
    struct FillContext
    {
        int cluster_id = -1;
        int needed = 0;
        int topk = 0;
        int radius = 5;

        std::vector<int32_t> selected_ids;
        std::vector<int8_t> selected_vecs; // flat

        FillContext() = default;
        FillContext(int cluster_id, int needed, int topk, int radius)
            : cluster_id(cluster_id), needed(needed), topk(topk), radius(radius) {}
    };
    struct Candidate
    {
        float dist;
        int32_t id;
        std::vector<int8_t> vec;
    };
    struct ClusterData
    {
        int64_t head_id = -1;
        int dim = 0;
        std::vector<int32_t> ids;    // posting ids
        std::vector<int8_t> vectors; // flat, size = ids.size() * dim

        int Count() const
        {
            return static_cast<int>(ids.size());
        }
    };

    class SpannIndex
    {
    public:
#pragma pack(push, 1)
        struct MetaRecord
        {
            int32_t pg;
            uint16_t off;
            int32_t cnt;
            uint16_t pc;
        };
#pragma pack(pop)

        static constexpr int64_t PAGE_SIZE = 4096;

    private:
        // 文件路径
        std::string head_path_;
        std::string head_id_path_;
        std::string disk_path_;

        // 元信息
        int32_t num_heads_ = 0;
        int32_t dim_ = 0;
        int32_t total_docs_ = 0;
        int64_t base_offset_ = 0;

        std::vector<int8_t> head_vecs_; // 质心数组[flat], size = num_heads_ * dim_
        std::vector<int64_t> head_ids_; // 质心向量 id
        std::vector<MetaRecord> meta_;  // 每个聚簇的元信息，size = num_heads_

        // 当前有效 cluster 数据（LoadAllClusters 后可用；PerformFilling 原地修改这里）
        std::vector<std::vector<int32_t>> cluster_ids_;
        std::vector<std::vector<int8_t>> cluster_vecs_;

        // filling 时的只读快照
        std::vector<std::vector<int32_t>> snapshot_cluster_ids_;
        std::vector<std::vector<int8_t>> snapshot_cluster_vecs_;

        // filling 参数
        int fill_target_size_ = 64;
        int fill_neighbor_topk_ = 16;
        bool fill_count_head_ = true;

        // filling 进度
        std::atomic<int> fill_processed_clusters_{0};
        std::atomic<bool> fill_stop_progress_{false};

        // 磁盘 fd
        int fd_ = -1;

        // head HNSW
        std::unique_ptr<hnswlib::L2SpaceI_int8> head_space_;
        std::unique_ptr<hnswlib::HierarchicalNSW<int>> head_hnsw_;

    private:
        void CheckClusterId(int cluster_id) const;
        void LoadHeads();
        void LoadHeadIds();
        void LoadMeta();

        // for filling
        void BuildSnapshotForFilling();
        void FillingWorker(int thread_id, int num_threads);
        std::vector<int32_t> QueryNeighborClustersByHead(int cluster_id, int topk) const;
        void SearchNeighborCandidates(FillContext &ctx) const;

    public:
        SpannIndex(const std::string &head_path, const std::string &head_id_path, const std::string &disk_path);
        ~SpannIndex();

        SpannIndex(const SpannIndex &) = delete;
        SpannIndex &operator=(const SpannIndex &) = delete;

        // 基础信息
        int GetNumHeads() const { return num_heads_; }
        int GetDim() const { return dim_; }
        int GetTotalDocs() const { return total_docs_; }
        int64_t GetBaseOffset() const { return base_offset_; }

        // head
        const int8_t *GetHeadVector(int cluster_id) const;
        int64_t GetHeadId(int cluster_id) const;
        int GetClusterSize(int cluster_id) const;

        // 单簇读取（posting，不含 head）
        int FetchCluster(int32_t cluster_id, std::vector<int8_t> &vecs, std::vector<int32_t> &ids) const;

        // 全量加载所有聚簇到内存
        void LoadAllClusters();
        bool AllClustersLoaded() const;

        // 访问 cluster
        const std::vector<int32_t> &GetClusterIds(int cluster_id) const;
        const std::vector<int8_t> &GetClusterVectors(int cluster_id) const;

        void BuildHeadHNSW(int M, int ef_construction, int ef_search);
        void PerformFilling(int target_size, int neighbor_topk, bool count_head, int num_threads);

        void Summary() const;

        // <===== for save ====>
    private:
        std::vector<std::vector<int32_t>> norms_;

    public:
        int32_t performNorms();
        void saveNorms(const std::string &norm_path) const;
        void saveHeadIndex(const std::string &head_index_path) const;
        void saveClusterIds(const std::string &cluster_ids_path) const;
    };

} // namespace spann
