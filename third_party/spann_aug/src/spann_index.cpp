#include "spann_index.h"
#include "root.h"

namespace spann
{
    void SpannIndex::CheckClusterId(int cluster_id) const
    {
        if (cluster_id < 0 || cluster_id >= num_heads_)
        {
            throw std::out_of_range(
                "cluster_id out of range: " + std::to_string(cluster_id) +
                ", valid range = [0, " + std::to_string(num_heads_) + ")");
        }
    }

    SpannIndex::SpannIndex(const std::string &head_path,
                           const std::string &head_id_path,
                           const std::string &disk_path)
        : head_path_(head_path),
          head_id_path_(head_id_path),
          disk_path_(disk_path)
    {
        LoadHeads();
        LoadHeadIds();
        LoadMeta();

        fd_ = ::open(disk_path_.c_str(), O_RDONLY);
        if (fd_ < 0)
        {
            throw std::runtime_error("Failed to open disk file: " + disk_path_);
        }
    }

    SpannIndex::~SpannIndex()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

} // namespace spann
