#include "nvme/nvme_allocator.hpp"

namespace minihypervec
{
  namespace nvme
  {
    NVMeAllocator *NVMeAllocator::getInstance()
    {
      static NVMeAllocator instance;
      return &instance;
    }

    int32_t NVMeAllocator::init()
    {
      const uint32_t ps = pageSize();
      if (ps == 0 || chunk_pages_ == 0)
        return kInvalidArgs;

      dev_states_.clear();
      order_dev_.clear();

      for (auto &d : meta_handler_->g_nvme_meta.devices)
      {
        DeviceState st;
        st.nvme_id = d.nvme_id;
        st.capacity_chunks = d.capacity_pages / chunk_pages_;

        uint64_t used_chunks = (d.used_pages + chunk_pages_ - 1) / chunk_pages_;
        if (used_chunks > st.capacity_chunks)
          used_chunks = st.capacity_chunks;

        st.free_chunks.reserve(st.capacity_chunks - used_chunks);
        for (uint64_t i = st.capacity_chunks; i > used_chunks; --i)
        {
          st.free_chunks.push_back(i - 1);
        }

        d.used_pages = used_chunks * chunk_pages_;
        d.free_pages = d.capacity_pages - d.used_pages;

        order_dev_.push_back(st.nvme_id);
        dev_states_.emplace(st.nvme_id, std::move(st));
      }
      return kOk;
    }

    NVMeDeviceMeta *NVMeAllocator::findDeviceById(uint32_t id)
    {
      for (auto &d : meta_handler_->g_nvme_meta.devices)
      {
        if (d.nvme_id == id)
          return &d;
      }
      return nullptr;
    }

    uint64_t NVMeAllocator::totalFreeChunks() const
    {
      uint64_t sum = 0;
      for (const auto &kv : dev_states_)
        sum += kv.second.free_chunks.size();
      return sum;
    }

    int32_t NVMeAllocator::configure(const AllocatorInitConfig &cfg,
                                     bool call_init)
    {
      if (configured_)
        return kAlreadyConfigured;
      if (!cfg.meta_handler)
        return kInvalidMetaHandler;

      meta_handler_ = cfg.meta_handler;
      if (cfg.fixed.by_bytes)
      {
        const uint32_t ps = pageSize();
        if (cfg.fixed.chunk_bytes == 0 || ps == 0 || (cfg.fixed.chunk_bytes % ps) != 0)
          return kInvalidArgs;
        chunk_pages_ = cfg.fixed.chunk_bytes / ps;
      }
      else
      {
        if (cfg.fixed.chunk_pages == 0)
          return kInvalidArgs;
        chunk_pages_ = cfg.fixed.chunk_pages;
      }

      if (call_init)
      {
        int32_t rc = init();
        if (rc != kOk)
          return rc;
      }

      configured_ = true;
      return kOk;
    }

    int32_t NVMeAllocator::allocate(uint64_t size_bytes, AllocationPlan &plan)
    {
      if (!configured_ || !meta_handler_)
        return kNotConfigured;

      plan = AllocationPlan{};
      if (size_bytes == 0)
        return kInvalidArg;

      const uint32_t ps = pageSize();
      if (ps == 0)
        return kInvalidArg;

      const uint64_t chunk_bytes = chunk_pages_ * static_cast<uint64_t>(ps);
      if (chunk_bytes == 0)
        return kInvalidArg;

      const uint64_t need_chunks =
          (size_bytes / chunk_bytes) + ((size_bytes % chunk_bytes) != 0);
      if (totalFreeChunks() < need_chunks)
        return kNoSpace;

      const uint64_t ndev = order_dev_.size();
      if (ndev == 0)
        return kNoDevices;

      plan.total_pages = need_chunks * chunk_pages_;
      plan.total_bytes = need_chunks * chunk_bytes;
      plan.chunks.reserve(need_chunks);

      uint64_t remain = need_chunks;
      const uint64_t start_rr = 0;

      struct Taken
      {
        uint32_t id;
        uint64_t chunk_idx;
      };
      std::vector<Taken> taken;
      taken.reserve(need_chunks);

      auto rollback = [&]()
      {
        for (auto it = taken.rbegin(); it != taken.rend(); ++it)
        {
          auto s = dev_states_.find(it->id);
          if (s != dev_states_.end())
            s->second.free_chunks.push_back(it->chunk_idx);

          if (auto *dev = findDeviceById(it->id))
          {
            if (dev->used_pages < chunk_pages_)
              dev->used_pages = 0;
            else
              dev->used_pages -= chunk_pages_;
            dev->free_pages = dev->capacity_pages - dev->used_pages;
          }
        }
        plan = AllocationPlan{};
      };

      while (remain > 0)
      {
        bool allocated_in_this_round = false;
        for (uint64_t step = 0; step < ndev && remain > 0; ++step)
        {
          const uint64_t idx = (start_rr + step) % ndev;
          const uint32_t id = order_dev_[idx];

          auto it = dev_states_.find(id);
          if (it == dev_states_.end())
            continue;
          auto &st = it->second;
          if (st.free_chunks.empty())
            continue;

          NVMeDeviceMeta *dev = findDeviceById(id);
          if (!dev)
          {
            rollback();
            return kDeviceMissing;
          }

          const uint64_t chunk_idx = st.free_chunks.back();
          st.free_chunks.pop_back();

          dev->used_pages += chunk_pages_;
          dev->free_pages = dev->capacity_pages - dev->used_pages;
          plan.chunks.push_back(Chunk{id, chunk_idx * chunk_pages_, chunk_pages_});
          taken.push_back(Taken{id, chunk_idx});
          --remain;
          allocated_in_this_round = true;
        }
        if (!allocated_in_this_round)
          break;
      }

      if (remain > 0)
      {
        rollback();
        return kStalled;
      }
      return kOk;
    }

    uint32_t NVMeAllocator::pageSize() const
    {
      if (!meta_handler_)
        return 0u;
      const auto &sys = meta_handler_->g_nvme_meta;
      if (sys.global_meta.page_size != 0)
        return sys.global_meta.page_size;
      for (const auto &d : sys.devices)
        if (d.page_size != 0)
          return d.page_size;
      return 0u;
    }

    uint64_t NVMeAllocator::chunkBytes() const
    {
      return chunk_pages_ * static_cast<uint64_t>(pageSize());
    }

  } // namespace nvme
} // namespace minihypervec