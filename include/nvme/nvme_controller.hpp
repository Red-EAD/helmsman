#pragma once

#include "nvme/nvme_meta.hpp"

namespace minihypervec
{
  namespace nvme
  {

    struct CtrlrEntry
    {
      spdk_nvme_ctrlr *ctrlr;
      std::string pcie_slot;
      uint32_t nvme_id;
      uint64_t qp_num = 0;
      std::vector<spdk_nvme_qpair *> qpair;
      std::vector<uint64_t> q_finished;
    };

    struct NsEntry
    {
      spdk_nvme_ctrlr *ctrlr;
      spdk_nvme_ns *ns;
      uint32_t nvme_id;
      uint32_t ns_id;
      uint32_t page_size = 512;
      uint64_t lba_num = 0;
    };

    struct CmdEntry
    {
      uint32_t nvme_id;
      uint32_t ns_id;
      char *buf;
      uint32_t que_id;
    };

    class NVMeCtrl
    {
    public:
      inline static NVMeMetaHandler *nvme_meta = nullptr;
      inline static uint64_t ctrl_total_num = 0;
      inline static uint32_t own_ctrl_num = 0;
      inline static uint32_t alloc_ctrl_num = 0;
      inline static std::unordered_map<uint32_t, CtrlrEntry *> all_ctrlr = {};
      inline static std::unordered_map<uint32_t, std::vector<NsEntry *>> all_namespace = {};

    public:
      CtrlrEntry *m_controller = nullptr;
      std::vector<NsEntry *> m_namespaces = {};

    public:
      NVMeCtrl() = default;
      int32_t initNVMe();
      static int32_t initSpdkEnv();
      static int32_t probeNVMe();
      static void releaseAllDev();

      int32_t writeSubmit(char *src, uint64_t lba_id, uint64_t lba_cnt, uint32_t que_id,
                          uint32_t ns_id = 0);
      int32_t readSubmit(char *dst, uint64_t lba_id, uint64_t lba_cnt, uint32_t que_id,
                         uint32_t ns_id = 0);

      int32_t pollCompletions(uint32_t que_id);

      uint64_t getFinishedQue(uint32_t que_id);
      void resetFinishedQue(uint32_t que_id);

    public:
      int32_t allocIoQue(uint64_t io_que_num, uint32_t que_depth = 1024);
      static void registerNs(struct spdk_nvme_ctrlr *ctrlr,
                             struct spdk_nvme_ns *ns, const std::string &pcie_slot);

      static bool probeCb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                          struct spdk_nvme_ctrlr_opts *opts);
      static void attachCb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                           struct spdk_nvme_ctrlr *ctrlr,
                           const struct spdk_nvme_ctrlr_opts *opts);
      static void writeCb(void *arg, const struct spdk_nvme_cpl *completion);
      static void readCb(void *arg, const struct spdk_nvme_cpl *completion);
    };

  } // namespace nvme
} // namespace minihypervec
