#include "meta_path.hpp"
#include "nvme/nvme_controller.hpp"
using namespace minihypervec;
using namespace minihypervec::nvme;

static std::string g_nvme_meta_path = "";
static uint32_t g_nvme_cnt = 0;
static std::vector<CtrlrEntry *> g_all_ctrlr = {};
static std::vector<std::vector<NsEntry *>> g_all_namespace =
    {};
static uint32_t g_page_size = 0;
static uint32_t g_cpu_cnt =
    std::thread::hardware_concurrency();
static uint32_t g_queue_depth = 1024;

static NVMeMetaHandler *g_nvme_meta = nullptr;

static void ConfigRegisterNs(struct spdk_nvme_ctrlr *ctrlr,
                             struct spdk_nvme_ns *ns)
{
  struct NsEntry *entry;
  if (!spdk_nvme_ns_is_active(ns))
  {
    return;
  }
  entry = new NsEntry();
  if (entry == NULL)
  {
    perror("NsEntry malloc");
    exit(1);
  }

  entry->ctrlr = ctrlr;
  entry->ns = ns;

  entry->nvme_id = g_nvme_cnt;

  entry->ns_id = spdk_nvme_ns_get_id(ns) - 1;

  entry->page_size = spdk_nvme_ns_get_sector_size(ns);
  g_page_size = entry->page_size;

  entry->lba_num = spdk_nvme_ns_get_size(ns) / entry->page_size;

  g_all_namespace.back().push_back(entry);
  std::cout << "Registered NVMe ID: " << entry->nvme_id
            << " PCIe Slot: " << g_all_ctrlr[g_nvme_cnt]->pcie_slot
            << " Namespace ID: " << entry->ns_id
            << " LBA Count: " << entry->lba_num
            << " Page Size: " << entry->page_size << std::endl;
}

static bool ConfigProbeCb(void *cb_ctx,
                          const struct spdk_nvme_transport_id *trid,
                          struct spdk_nvme_ctrlr_opts *opts)
{
  printf("Attaching to %s\n", trid->traddr);
  return true;
}

static void ConfigAttachCb(void *cb_ctx,
                           const struct spdk_nvme_transport_id *trid,
                           struct spdk_nvme_ctrlr *ctrlr,
                           const struct spdk_nvme_ctrlr_opts *opts)
{
  int nsid;
  struct CtrlrEntry *entry;
  struct spdk_nvme_ns *ns;
  const struct spdk_nvme_ctrlr_data *cdata;

  entry = new CtrlrEntry();
  g_all_ctrlr.push_back(entry);
  if (entry == NULL)
  {
    perror("CtrlrEntry  malloc");
    exit(1);
  }

  cdata = spdk_nvme_ctrlr_get_data(ctrlr);

  entry->ctrlr = ctrlr;

  entry->pcie_slot = std::string(trid->traddr);
  std::cout << "Attached to PCIe slot: " << entry->pcie_slot << std::endl;

  entry->nvme_id = g_nvme_cnt;
  std::cout << "NVMe id " << entry->nvme_id << std::endl;

  g_all_namespace.push_back({});
  for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
       nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid))
  {
    ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
    if (ns == NULL)
    {
      continue;
    }
    ConfigRegisterNs(ctrlr, ns);
  }
  g_nvme_cnt++;
}

void CleanupAllNvme()
{
  for (auto ctrlr_entry : g_all_ctrlr)
  {
    if (ctrlr_entry && ctrlr_entry->ctrlr)
    {
      std::cout << "Detaching NVMe controller at " << ctrlr_entry->pcie_slot
                << std::endl;
      spdk_nvme_detach(ctrlr_entry->ctrlr);
    }
    delete ctrlr_entry;
  }
  g_all_ctrlr.clear();

  for (auto &ns_list : g_all_namespace)
  {
    for (auto ns_entry : ns_list)
    {
      delete ns_entry;
    }
  }
  g_all_namespace.clear();
}

void ConfigMeta()
{
  g_nvme_meta = NVMeMetaHandler::getInstance();
  NVMeSystemMeta &g_meta = g_nvme_meta->g_nvme_meta;
  g_meta.global_meta.total_devices = g_nvme_cnt;
  g_meta.global_meta.page_size = g_page_size;
  g_meta.global_meta.queues_per_device = g_cpu_cnt;
  g_meta.global_meta.queue_depth = g_queue_depth;

  for (auto &ctrlr_entry : g_all_ctrlr)
  {
    NVMeDeviceMeta device_meta;
    device_meta.nvme_id = ctrlr_entry->nvme_id;
    device_meta.pcie_slot = ctrlr_entry->pcie_slot;
    device_meta.page_size = g_page_size;
    device_meta.capacity_pages = g_all_namespace[device_meta.nvme_id][0]
                                     ->lba_num;
    device_meta.used_pages = 0;
    device_meta.free_pages =
        device_meta.capacity_pages;
    device_meta.life_left = 100;

    g_nvme_meta->slot_to_nvme_meta[device_meta.pcie_slot] = device_meta;
    g_meta.devices.push_back(device_meta);
  }

  NVMeMetaHandler::saveNVMeSystemMetaToFile(g_meta, g_nvme_meta_path);
}

int main()
{
  PathConfig *pc = PathConfig::getInstance();
  pc->init(std::string(g_path_config));
  g_nvme_meta_path = minihypervec::nvme::getNVMeMetaPath();
  NVMeMetaHandler *meta_handler = NVMeMetaHandler::getInstance();

  NVMeSystemMeta &g_meta = meta_handler->g_nvme_meta;
  spdk_env_fini();
  spdk_env_opts opts;
  spdk_env_opts_init(&opts);
  if (spdk_env_init(&opts) < 0)
  {
    printf("Unable to initialize SPDK env\n");
    return -1;
  }

  int32_t res = -1;
  res = spdk_nvme_probe(NULL, NULL, ConfigProbeCb, ConfigAttachCb, NULL);
  if (res != 0)
  {
    std::cerr << "probe NVMe failed" << std::endl;
    spdk_env_fini();
    return -1;
  }
  ConfigMeta();
  CleanupAllNvme();
  spdk_env_fini();
  return 0;
}