#include "nvme/nvme_meta.hpp"

namespace minihypervec
{
  namespace nvme
  {

    using nlohmann::json;

    static void to_json(json &j, const NVMeDeviceMeta::GlobalMeta &gm)
    {
      j = json{
          {"total_devices", gm.total_devices},
          {"page_size", gm.page_size},
          {"queues_per_device", gm.queues_per_device},
          {"queue_depth", gm.queue_depth},
      };
    }

    static void from_json(const json &j, NVMeDeviceMeta::GlobalMeta &gm)
    {
      gm.total_devices = j.value("total_devices", gm.total_devices);
      gm.page_size = j.value("page_size", gm.page_size);
      gm.queues_per_device = j.value("queues_per_device", gm.queues_per_device);
      gm.queue_depth = j.value("queue_depth", gm.queue_depth);
    }

    static void to_json(json &j, const NVMeDeviceMeta &dm)
    {
      j = json{
          {"nvme_id", dm.nvme_id},
          {"pcie_slot", dm.pcie_slot},
          {"capacity_pages", dm.capacity_pages},
          {"page_size", dm.page_size},
          {"used_pages", dm.used_pages},
          {"free_pages", dm.free_pages},
          {"life_left", dm.life_left},
      };
    }

    static void from_json(const json &j, NVMeDeviceMeta &dm)
    {
      j.at("nvme_id").get_to(dm.nvme_id);
      j.at("pcie_slot").get_to(dm.pcie_slot);
      j.at("capacity_pages").get_to(dm.capacity_pages);
      j.at("page_size").get_to(dm.page_size);
      dm.used_pages = j.value("used_pages", uint64_t{0});
      dm.free_pages = j.value("free_pages", uint64_t{0});
      dm.life_left = j.value("life_left", int32_t{-1});
    }

    static void to_json(json &j, const NVMeSystemMeta &sm)
    {
      j = json{
          {"global_meta", sm.global_meta},
          {"devices", sm.devices},
      };
    }

    static void from_json(const json &j, NVMeSystemMeta &sm)
    {
      sm.global_meta = j.value("global_meta", NVMeDeviceMeta::GlobalMeta{});
      sm.devices = j.value("devices", std::vector<NVMeDeviceMeta>{});
    }

    NVMeMetaHandler *NVMeMetaHandler::getInstance()
    {
      static NVMeMetaHandler inst;
      return &inst;
    }

    int32_t NVMeMetaHandler::init(const std::string &path)
    {
      NVMeSystemMeta meta;
      int32_t rc = loadNVMeSystemMetaFromFile(meta, path);
      if (rc != 0)
        return rc;

      g_nvme_meta = std::move(meta);

      slot_to_nvme_meta.clear();
      slot_to_nvme_meta.reserve(g_nvme_meta.devices.size());
      for (const auto &dev : g_nvme_meta.devices)
      {
        slot_to_nvme_meta.emplace(dev.pcie_slot, dev);
      }
      return 0;
    }

    int32_t NVMeMetaHandler::sync(const std::string &path)
    {
      return saveNVMeSystemMetaToFile(g_nvme_meta, path);
    }

    void NVMeMetaHandler::printMeta() const
    {
      std::cout << "NVMeSystemMeta:\n";
      std::cout << "  global_meta.total_devices="
                << g_nvme_meta.global_meta.total_devices << "\n";
      std::cout << "  global_meta.page_size=" << g_nvme_meta.global_meta.page_size
                << "\n";
      std::cout << "  global_meta.queues_per_device="
                << g_nvme_meta.global_meta.queues_per_device << "\n";
      std::cout << "  global_meta.queue_depth="
                << g_nvme_meta.global_meta.queue_depth << "\n";

      std::cout << "  devices:\n";
      for (const auto &d : g_nvme_meta.devices)
      {
        std::cout << "    - nvme_id=" << d.nvme_id << ", slot=" << d.pcie_slot
                  << ", capacity_pages=" << d.capacity_pages
                  << ", page_size=" << d.page_size
                  << ", used_pages=" << d.used_pages
                  << ", free_pages=" << d.free_pages
                  << ", life_left=" << d.life_left << "\n";
      }
    }

    int32_t NVMeMetaHandler::loadNVMeSystemMetaFromFile(NVMeSystemMeta &meta,
                                                        const std::string &path)
    {
      std::string content;
      int32_t rc = util::read_file_to_string(path, content);
      if (rc != 0)
        return rc;

      try
      {
        json j = json::parse(content);
        meta = j.get<NVMeSystemMeta>();
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error---NVMeMetaHandler::loadNVMeSystemMetaFromFile: JSON "
                     "parse/load failed: "
                  << e.what() << "\n";
        return -EINVAL;
      }

      return 0;
    }

    int32_t NVMeMetaHandler::saveNVMeSystemMetaToFile(const NVMeSystemMeta &meta,
                                                      const std::string &path)
    {
      try
      {
        json j = meta;
        std::string data = j.dump(2, ' ', false);
        data.push_back('\n');
        return util::persist_string_atomic_fsync(path, data);
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error---NVMeMetaHandler::saveNVMeSystemMetaToFile: JSON save "
                     "failed: "
                  << e.what() << "\n";
        return -EINVAL;
      }
    }

  } // namespace nvme
} // namespace minihypervec