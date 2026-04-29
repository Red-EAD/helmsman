#pragma once

#include "root.hpp"
#include "util/file/files_rw.hpp"

namespace minihypervec
{
  namespace nvme
  {

    struct NVMeDeviceMeta
    {
      uint32_t nvme_id;
      std::string pcie_slot;
      uint64_t capacity_pages;
      uint32_t page_size;
      uint64_t used_pages;
      uint64_t free_pages;
      int32_t life_left;

      struct GlobalMeta
      {
        uint32_t total_devices = 3;
        uint32_t page_size = 512;
        uint32_t queues_per_device = 32;
        uint32_t queue_depth = 1024;
      };
    };

    struct NVMeSystemMeta
    {
      NVMeDeviceMeta::GlobalMeta global_meta;
      std::vector<NVMeDeviceMeta> devices;
    };

    class NVMeMetaHandler
    {
    public:
      NVMeSystemMeta g_nvme_meta;
      std::unordered_map<std::string, NVMeDeviceMeta>
          slot_to_nvme_meta;

    public:
      int32_t init(const std::string &path);
      int32_t sync(const std::string &path);
      void printMeta() const;

    public:
      static NVMeMetaHandler *getInstance();
      static int32_t loadNVMeSystemMetaFromFile(NVMeSystemMeta &meta,
                                                const std::string &path);
      static int32_t saveNVMeSystemMetaToFile(const NVMeSystemMeta &meta,
                                              const std::string &path);
    };

  } // namespace nvme
} // namespace minihypervec