#include "nvme/nvme_manager.hpp"

namespace minihypervec
{
  namespace nvme
  {

    NVMeManager *NVMeManager::getInstance()
    {
      static NVMeManager m_instance;
      return &m_instance;
    }

    int32_t NVMeManager::extraMemInit(void *ptr, uint64_t size)
    {
      int32_t res = 0;
      res = spdk_mem_register(static_cast<void *>(ptr), size);
      if (res != 0)
      {
        printf("extra mem init for nvme failed.\n");
        res = -1;
      }
      return res;
    }

    void *NVMeManager::mallocNVMeHostBuf(uint64_t sz)
    {
      void *res = spdk_zmalloc(sz, 0x200000UL, NULL, SPDK_ENV_SOCKET_ID_ANY,
                               SPDK_MALLOC_DMA);
      return res;
    }

    void NVMeManager::freeNVMeHostBuf(void *ptr) { spdk_free(ptr); }

    int32_t NVMeManager::initNVMeMeta(const std::string &path)
    {
      NVMeMetaHandler *meta_handler = NVMeMetaHandler::getInstance();
      std::cout << path << " init NVMe metadata res: " << path << std::endl;
      int32_t res = meta_handler->init(path);
      if (res != 0)
      {
        std::cerr << "Failed to initialize NVMe metadata from " << path
                  << std::endl;
        return res;
      }
      meta_handler->printMeta();
      return 0;
    }

    int32_t NVMeManager::initNVMeEnv()
    {
      int32_t res = 0;
      if (NVMeCtrl::initSpdkEnv() < 0)
      {
        printf("Unable to initialize SPDK env\n");
        res = -1;
        return res;
      }
      else
      {
        NVMeCtrl::probeNVMe();
      }
      uint64_t probe_dev_num = NVMeCtrl::ctrl_total_num;
      std::cout << "Total nvme ssd num in the machine: " << probe_dev_num
                << std::endl;
      total_dev_num = NVMeCtrl::own_ctrl_num;
      std::cout << "Total nvme ssd num used in NVMeManager of hyper-vec: "
                << total_dev_num << std::endl;
      return res;
    }

    int32_t NVMeManager::initNVMeDev()
    {
      int32_t res = 0;
      uint64_t nvme_num = total_dev_num;
      all_ctrls.resize(nvme_num);
      ctrls_alloced_qp.resize(nvme_num);
      for (uint64_t nvme_id = 0; nvme_id < nvme_num; ++nvme_id)
      {
        all_ctrls[nvme_id] = new NVMeCtrl();
        res = all_ctrls[nvme_id]->initNVMe();
      }
      return res;
    }

    int32_t NVMeManager::getNVMeDevNum()
    {
      return total_dev_num;
    }

    uint64_t NVMeManager::getTotalLbaNum(uint32_t nvme_id, uint32_t ns_id)
    {
      auto ctrl = all_ctrls[nvme_id];
      return ctrl->m_namespaces[ns_id]->lba_num;
    }

    int32_t NVMeManager::writeSubmit(uint32_t nvme_id, char *src, uint64_t lba_id,
                                     uint64_t lba_cnt, uint32_t que_id,
                                     uint32_t ns_id)
    {
      auto ctrl = all_ctrls[nvme_id];
      return ctrl->writeSubmit(src, lba_id, lba_cnt, que_id, ns_id);
    }

    int32_t NVMeManager::readSubmit(uint32_t nvme_id, char *dst, uint64_t lba_id,
                                    uint64_t lba_cnt, uint32_t que_id,
                                    uint32_t ns_id)
    {
      auto ctrl = all_ctrls[nvme_id];
      return ctrl->readSubmit(dst, lba_id, lba_cnt, que_id, ns_id);
    }

    int32_t NVMeManager::pollCompletions(uint32_t nvme_id, uint32_t que_id)
    {
      auto ctrl = all_ctrls[nvme_id];
      return ctrl->pollCompletions(que_id);
    }

    uint64_t NVMeManager::getFinishedQue(uint32_t nvme_id, uint32_t que_id)
    {
      auto ctrl = all_ctrls[nvme_id];
      return ctrl->getFinishedQue(que_id);
    }

    void NVMeManager::resetFinishedQue(uint32_t nvme_id, uint32_t que_id)
    {
      auto ctrl = all_ctrls[nvme_id];
      ctrl->resetFinishedQue(que_id);
    }

    int32_t NVMeManager::allocQue(uint32_t nvme_id, uint64_t que_cnt,
                                  std::vector<uint64_t> &ques_id)
    {
      int32_t res = 0;
      auto ctrl = all_ctrls[nvme_id];
      if (ctrls_alloced_qp[nvme_id] + que_cnt > ctrl->m_controller->qp_num)
      {
        std::cout << ctrls_alloced_qp[nvme_id] << " + " << que_cnt << " > "
                  << ctrl->m_controller->qp_num << std::endl;
        std::cerr << "Unalloc IO Que not enough" << std::endl;
        res = -1;
        return res;
      }
      ques_id.resize(que_cnt);
      for (uint64_t que_alloc = 0; que_alloc < que_cnt; ++que_alloc)
      {
        ques_id[que_alloc] = ctrls_alloced_qp[nvme_id];
        ++ctrls_alloced_qp[nvme_id];
      }
      return res;
    }

    void NVMeManager::releaseNVMeEnv() { NVMeCtrl::releaseAllDev(); }

  } // namespace nvme
} // namespace minihypervec