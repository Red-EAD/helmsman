#pragma once

#include "nvme/nvme_controller.hpp"

namespace minihypervec
{
  namespace nvme
  {
    class NVMeManager
    {
    public:
      static NVMeManager *getInstance();

      static int32_t extraMemInit(void *ptr, uint64_t size);
      static void *mallocNVMeHostBuf(uint64_t sz);
      static void freeNVMeHostBuf(void *ptr);

      int32_t initNVMeMeta(const std::string &path);
      int32_t initNVMeEnv();
      int32_t initNVMeDev();
      void releaseNVMeEnv();

      int32_t getNVMeDevNum();
      uint64_t getTotalLbaNum(uint32_t nvme_id, uint32_t ns_id);
      int32_t writeSubmit(uint32_t nvme_id, char *src, uint64_t lba_id,
                          uint64_t lba_cnt, uint32_t que_id, uint32_t ns_id = 0);
      int32_t readSubmit(uint32_t nvme_id, char *dst, uint64_t lba_id,
                         uint64_t lba_cnt, uint32_t que_id, uint32_t ns_id = 0);

      int32_t pollCompletions(uint32_t nvme_id, uint32_t que_id);
      uint64_t getFinishedQue(uint32_t nvme_id, uint32_t que_id);
      void resetFinishedQue(uint32_t nvme_id, uint32_t que_id);
      int32_t allocQue(uint32_t nvme_id, uint64_t que_cnt,
                       std::vector<uint64_t> &ques_id);

    public:
      NVMeMetaHandler *meta_handler = nullptr;
      std::vector<NVMeCtrl *> all_ctrls;
      std::vector<uint64_t> ctrls_alloced_qp;
      uint64_t total_dev_num = 0;
    };

  } // namespace nvme
} // namespace minihypervec