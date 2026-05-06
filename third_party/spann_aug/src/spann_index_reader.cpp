#include "spann_index.h"
#include "root.h"

namespace spann
{
    static_assert(sizeof(SpannIndex::MetaRecord) == 12, "MetaRecord layout mismatch");

    void SpannIndex::LoadHeads()
    {
        std::ifstream fin(head_path_, std::ios::binary);
        if (!fin)
        {
            throw std::runtime_error("Missing head vector file: " + head_path_);
        }

        int32_t num_heads = 0;
        int32_t dim = 0;

        fin.read(reinterpret_cast<char *>(&num_heads), sizeof(int32_t));
        fin.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));

        if (!fin || num_heads <= 0 || dim <= 0)
        {
            throw std::runtime_error("Invalid head vector file header");
        }

        std::vector<int8_t> raw(static_cast<size_t>(num_heads) * dim);
        fin.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));

        if (!fin)
        {
            throw std::runtime_error(
                "Head vector file truncated: expect " +
                std::to_string(static_cast<size_t>(num_heads) * dim) + " bytes");
        }

        num_heads_ = num_heads;
        dim_ = dim;
        head_vecs_ = std::move(raw);
    }

    void SpannIndex::LoadHeadIds()
    {
        std::ifstream fin(head_id_path_, std::ios::binary);
        if (!fin)
        {
            throw std::runtime_error("Missing head ID file: " + head_id_path_);
        }

        std::vector<int64_t> ids(num_heads_);
        fin.read(reinterpret_cast<char *>(ids.data()),
                 static_cast<std::streamsize>(num_heads_) * sizeof(int64_t));

        if (!fin)
        {
            throw std::runtime_error(
                "Head ID file truncated: expect " + std::to_string(num_heads_) + " int64 values");
        }

        head_ids_ = std::move(ids);
    }

    void SpannIndex::LoadMeta()
    {
        std::ifstream fin(disk_path_, std::ios::binary);
        if (!fin)
        {
            throw std::runtime_error("Missing disk index file: " + disk_path_);
        }

        int32_t num_lists = 0;
        int32_t total_docs = 0;
        int32_t dim = 0;
        int32_t start_page = 0;

        fin.read(reinterpret_cast<char *>(&num_lists), sizeof(int32_t));
        fin.read(reinterpret_cast<char *>(&total_docs), sizeof(int32_t));
        fin.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
        fin.read(reinterpret_cast<char *>(&start_page), sizeof(int32_t));

        if (!fin)
        {
            throw std::runtime_error("Invalid disk index header");
        }

        if (num_lists <= 0)
        {
            throw std::runtime_error("Invalid num_lists: " + std::to_string(num_lists));
        }
        if (dim <= 0)
        {
            throw std::runtime_error("Invalid dim: " + std::to_string(dim));
        }
        if (dim != dim_)
        {
            throw std::runtime_error(
                "Dim mismatch: head dim=" + std::to_string(dim_) +
                ", disk dim=" + std::to_string(dim));
        }
        if (start_page < 0)
        {
            throw std::runtime_error("Invalid start_page: " + std::to_string(start_page));
        }
        if (num_lists != num_heads_)
        {
            throw std::runtime_error(
                "Meta count mismatch: num_lists=" + std::to_string(num_lists) +
                ", num_heads=" + std::to_string(num_heads_));
        }

        std::vector<MetaRecord> meta(num_lists);
        fin.read(reinterpret_cast<char *>(meta.data()),
                 static_cast<std::streamsize>(num_lists) * sizeof(MetaRecord));

        if (!fin)
        {
            throw std::runtime_error(
                "Metadata truncated: expect " + std::to_string(num_lists) + " records");
        }

        total_docs_ = total_docs;
        base_offset_ = static_cast<int64_t>(start_page) * PAGE_SIZE;
        meta_ = std::move(meta);
    }

    const int8_t *SpannIndex::GetHeadVector(int cluster_id) const
    {
        CheckClusterId(cluster_id);
        return head_vecs_.data() + static_cast<size_t>(cluster_id) * dim_;
    }

    int64_t SpannIndex::GetHeadId(int cluster_id) const
    {
        CheckClusterId(cluster_id);
        return head_ids_[cluster_id];
    }

    int SpannIndex::GetClusterSize(int cluster_id) const
    {
        CheckClusterId(cluster_id);
        return meta_[cluster_id].cnt;
    }

    int SpannIndex::FetchCluster(int32_t cluster_id, std::vector<int8_t> &vecs, std::vector<int32_t> &ids) const
    {
        CheckClusterId(cluster_id);

        const MetaRecord &m = meta_[cluster_id];
        const int cnt = m.cnt;
        const size_t row_size = sizeof(int32_t) + static_cast<size_t>(dim_);
        const size_t total_bytes = static_cast<size_t>(cnt) * row_size;
        const int64_t abs_off = base_offset_ + static_cast<int64_t>(m.pg) * PAGE_SIZE + m.off;

        std::vector<int8_t> buffer(total_bytes);
        if (total_bytes > 0)
        {
            ssize_t ret = ::pread(fd_, buffer.data(), total_bytes, abs_off);
            if (ret < 0 || static_cast<size_t>(ret) != total_bytes)
            {
                throw std::runtime_error(
                    "Short read for cluster " + std::to_string(cluster_id) +
                    ": expect " + std::to_string(total_bytes) +
                    ", got " + std::to_string(ret));
            }
        }

        ids.resize(cnt);
        vecs.resize(static_cast<size_t>(cnt) * dim_);
        for (int i = 0; i < cnt; ++i)
        {
            const size_t row_off = static_cast<size_t>(i) * row_size;
            std::memcpy(&ids[i], buffer.data() + row_off, sizeof(int32_t));
            const int8_t *vec_ptr =
                reinterpret_cast<const int8_t *>(buffer.data() + row_off + sizeof(int32_t));
            std::memcpy(vecs.data() + static_cast<size_t>(i) * dim_, vec_ptr, dim_);
        }

        ids.push_back(head_ids_[cluster_id]);
        const int8_t *head_vec = GetHeadVector(cluster_id);
        vecs.insert(vecs.end(), head_vec, head_vec + static_cast<size_t>(dim_));
        return 0;
    }

    void SpannIndex::LoadAllClusters()
    {
        std::cout << "Loading all clusters into memory...\n";

        cluster_ids_.clear();
        cluster_vecs_.clear();
        cluster_ids_.resize(num_heads_);
        cluster_vecs_.resize(num_heads_);

        for (int cid = 0; cid < num_heads_; ++cid)
        {
            FetchCluster(cid, cluster_vecs_[cid], cluster_ids_[cid]);
        }

        std::cout << "All clusters loaded: " << num_heads_ << " clusters\n";
    }

    bool SpannIndex::AllClustersLoaded() const
    {
        return static_cast<int>(cluster_ids_.size()) == num_heads_ &&
               static_cast<int>(cluster_vecs_.size()) == num_heads_;
    }

    const std::vector<int32_t> &SpannIndex::GetClusterIds(int cluster_id) const
    {
        CheckClusterId(cluster_id);
        if (!AllClustersLoaded())
        {
            throw std::runtime_error("Clusters are not fully loaded");
        }
        return cluster_ids_[cluster_id];
    }

    const std::vector<int8_t> &SpannIndex::GetClusterVectors(int cluster_id) const
    {
        CheckClusterId(cluster_id);
        if (!AllClustersLoaded())
        {
            throw std::runtime_error("Clusters are not fully loaded");
        }
        return cluster_vecs_[cluster_id];
    }

    void SpannIndex::Summary() const
    {
        std::cout << "===== SpannIndex Summary =====\n";
        std::cout << "num_heads   : " << num_heads_ << "\n";
        std::cout << "dim         : " << dim_ << "\n";
        std::cout << "total_docs  : " << total_docs_ << "\n";
        std::cout << "base_offset : " << base_offset_ << "\n";
        std::cout << "head_vecs   : shape=(" << head_vecs_.size() << "), dtype=int8\n";
        std::cout << "head_ids    : shape=(" << head_ids_.size() << "), dtype=int64\n";
        std::cout << "meta        : shape=(" << meta_.size() << "), sizeof(MetaRecord)="
                  << sizeof(MetaRecord) << "\n";

        if (AllClustersLoaded())
        {
            std::cout << "cluster_vecs loaded: " << cluster_vecs_.size() << " clusters\n";
            std::cout << "cluster_ids  loaded: " << cluster_ids_.size() << " clusters\n";
        }
        else
        {
            std::cout << "cluster_vecs / cluster_ids not loaded yet\n";
        }
    }
} // namespace spann
