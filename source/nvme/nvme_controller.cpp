#include "nvme/nvme_controller.hpp"

namespace minihypervec
{
  namespace nvme
  {

    void NVMeCtrl::registerNs(struct spdk_nvme_ctrlr *ctrlr,
                              struct spdk_nvme_ns *ns,
                              const std::string &pcie_slot)
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

      entry->nvme_id = nvme_meta->slot_to_nvme_meta[pcie_slot].nvme_id;
      entry->ctrlr = ctrlr;
      entry->ns = ns;
      entry->ns_id = spdk_nvme_ns_get_id(ns) - 1;
      entry->page_size = spdk_nvme_ns_get_sector_size(ns);

      if (nvme_meta->g_nvme_meta.global_meta.page_size != entry->page_size)
      {
        std::cerr << "Error: NVMe device page size mismatch. Expected "
                  << nvme_meta->g_nvme_meta.global_meta.page_size << " but got "
                  << entry->page_size << std::endl;
        exit(1);
      }

      entry->lba_num = spdk_nvme_ns_get_size(ns) / entry->page_size;

      all_namespace[entry->nvme_id].push_back(entry);

      std::cout << "Registered NVMe ID: " << (uint32_t)entry->nvme_id
                << " PCIe Slot: " << pcie_slot
                << " Namespace ID: " << (uint32_t)entry->ns_id
                << " LBA Count: " << entry->lba_num
                << " Page Size: " << entry->page_size << std::endl;
    }

    bool NVMeCtrl::probeCb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                           struct spdk_nvme_ctrlr_opts *opts)
    {
      printf("Attaching to %s\n", trid->traddr);
      return true;
    }

    void NVMeCtrl::attachCb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                            struct spdk_nvme_ctrlr *ctrlr,
                            const struct spdk_nvme_ctrlr_opts *opts)
    {
      int nsid;
      CtrlrEntry *entry;
      spdk_nvme_ns *ns;
      const spdk_nvme_ctrlr_data *cdata;

      entry = new CtrlrEntry();
      if (entry == NULL)
      {
        perror("CtrlrEntry  malloc");
        exit(1);
      }

      cdata = spdk_nvme_ctrlr_get_data(ctrlr);
      entry->ctrlr = ctrlr;
      entry->pcie_slot = std::string(trid->traddr);
      entry->nvme_id = nvme_meta->slot_to_nvme_meta[entry->pcie_slot].nvme_id;

      std::cout << "Attached to PCIe slot: " << entry->pcie_slot << std::endl;
      std::cout << "NVMe id " << (uint32_t)entry->nvme_id << std::endl;

      all_ctrlr[entry->nvme_id] = entry;
      all_namespace[entry->nvme_id] = {};

      for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
           nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid))
      {
        ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (ns == nullptr)
        {
          continue;
        }
        registerNs(ctrlr, ns, entry->pcie_slot);
      }
      own_ctrl_num++;
    }

    int32_t NVMeCtrl::initSpdkEnv()
    {
      int32_t res = 0;
      spdk_env_opts opts;
      spdk_env_opts_init(&opts);
      opts.name = "MiniHyperVec-SpdkEnv";
      if (spdk_env_init(&opts) < 0)
      {
        printf("Unable to initialize SPDK env\n");
        res = -1;
      }
      return res;
    }

    int32_t NVMeCtrl::probeNVMe()
    {
      int32_t res = -1;
      nvme_meta = NVMeMetaHandler::getInstance();
      ctrl_total_num = nvme_meta->g_nvme_meta.global_meta.total_devices;
      res = spdk_nvme_probe(NULL, NULL, probeCb, attachCb, NULL);
      if (res != 0)
      {
        std::cerr << "probe NVMe failed" << std::endl;
        NVMeCtrl::releaseAllDev();
        spdk_env_fini();
        return -1;
      }
      return res;
    }

    void NVMeCtrl::releaseAllDev()
    {
      std::cout << "start release" << std::endl;
      for (uint32_t cur_nvme_id = 0; cur_nvme_id < own_ctrl_num; ++cur_nvme_id)
      {
        std::cout << "release nvme id: " << all_ctrlr[cur_nvme_id]->nvme_id
                  << " pcie slot: " << all_ctrlr[cur_nvme_id]->pcie_slot
                  << std::endl;
        for (auto *ns : all_namespace[cur_nvme_id])
        {
          std::cout << "release ns id: " << ns->ns_id << std::endl;
          if (ns != nullptr)
          {
            delete ns;
            ns = nullptr;
          }
        }
        for (uint32_t num = 0; num < all_ctrlr[cur_nvme_id]->qpair.size(); ++num)
        {
          spdk_nvme_ctrlr_free_io_qpair(all_ctrlr[cur_nvme_id]->qpair[num]);
        }
        std::cout << "detaching" << std::endl;
        spdk_nvme_detach(all_ctrlr[cur_nvme_id]->ctrlr);
        std::cout << "detached" << std::endl;
        if (all_ctrlr[cur_nvme_id] != nullptr)
        {
          delete all_ctrlr[cur_nvme_id];
          all_ctrlr[cur_nvme_id] = nullptr;
        }
      }
    }

    int32_t NVMeCtrl::allocIoQue(uint64_t qp_num, uint32_t que_depth)
    {
      int32_t res = 0;
      if (m_controller == nullptr)
      {
        res = -1;
        return res;
      }
      if (qp_num > NVMeMetaHandler::getInstance()
                       ->g_nvme_meta.global_meta.queues_per_device)
      {
        res = -1;
        std::cerr << "qp_num should be <= IO_QUEUE_CNT" << std::endl;
        return res;
      }
      m_controller->qp_num = qp_num;
      m_controller->qpair.resize(qp_num);
      m_controller->q_finished.resize(qp_num);
      struct spdk_nvme_io_qpair_opts io_qpair_opts;
      spdk_nvme_ctrlr_get_default_io_qpair_opts(m_controller->ctrlr, &io_qpair_opts,
                                                sizeof(io_qpair_opts));

      io_qpair_opts.io_queue_size = que_depth;
      io_qpair_opts.io_queue_requests = 2 * io_qpair_opts.io_queue_size;
      io_qpair_opts.delay_pcie_doorbell = true;
      for (uint64_t num = 0; num < qp_num; ++num)
      {
        m_controller->qpair[num] = spdk_nvme_ctrlr_alloc_io_qpair(
            m_controller->ctrlr, &io_qpair_opts, sizeof(spdk_nvme_io_qpair_opts));
        m_controller->q_finished[num] = 0;
        if (m_controller->qpair[num] == nullptr)
        {
          printf("ERROR: spdk_nvme_ctrlr_alloc_io_qpair() failed\n");
          return -1;
        }
      }
      return res;
    }

    int32_t NVMeCtrl::initNVMe()
    {
      int32_t res = -1;
      if (alloc_ctrl_num > all_ctrlr.size())
      {
        std::cerr << "Reserve NVMe Device not enough." << std::endl;
        return res;
      }
      m_controller = all_ctrlr[alloc_ctrl_num];
      m_namespaces = all_namespace[alloc_ctrl_num];
      alloc_ctrl_num++;

      uint64_t io_que_num =
          NVMeMetaHandler::getInstance()->g_nvme_meta.global_meta.queues_per_device;
      uint32_t que_depth =
          NVMeMetaHandler::getInstance()->g_nvme_meta.global_meta.queue_depth;

      std::cout << "Allocating " << io_que_num << " I/O queues with depth "
                << que_depth << std::endl;

      res = allocIoQue(io_que_num, que_depth);
      return res;
    }

    void NVMeCtrl::writeCb(void *arg, const struct spdk_nvme_cpl *completion)
    {
      CmdEntry *write_cmd = static_cast<CmdEntry *>(arg);
      all_ctrlr[write_cmd->nvme_id]->q_finished[write_cmd->que_id]++;
      if (write_cmd != nullptr)
      {
        free(write_cmd);
        write_cmd = nullptr;
      }
    }

    void NVMeCtrl::readCb(void *arg, const struct spdk_nvme_cpl *completion)
    {
      CmdEntry *read_cmd = static_cast<CmdEntry *>(arg);
      all_ctrlr[read_cmd->nvme_id]->q_finished[read_cmd->que_id]++;
      if (read_cmd != nullptr)
      {
        free(read_cmd);
        read_cmd = nullptr;
      }
    }

    int32_t NVMeCtrl::writeSubmit(char *src, uint64_t lba_id, uint64_t lba_cnt,
                                  uint32_t que_id, uint32_t ns_id)
    {
      CmdEntry *write_cmd = static_cast<CmdEntry *>(malloc(sizeof(CmdEntry)));
      NsEntry *ns_ptr = m_namespaces[ns_id];
      write_cmd->buf = src;
      write_cmd->nvme_id = ns_ptr->nvme_id;
      write_cmd->ns_id = ns_ptr->ns_id;
      write_cmd->que_id = que_id;

      int32_t rc = spdk_nvme_ns_cmd_write(ns_ptr->ns,
                                          m_controller->qpair[que_id],
                                          src, lba_id,
                                          lba_cnt,
                                          writeCb, write_cmd, 0);
      if (rc != 0)
      {
        fprintf(stderr, "starting write I/O failed\n");
        exit(1);
      }

      return rc;
    }

    int32_t NVMeCtrl::readSubmit(char *dst, uint64_t lba_id, uint64_t lba_cnt,
                                 uint32_t que_id, uint32_t ns_id)
    {
      CmdEntry *read_cmd = static_cast<CmdEntry *>(malloc(sizeof(CmdEntry)));
      NsEntry *ns_ptr = m_namespaces[ns_id];
      read_cmd->buf = dst;
      read_cmd->nvme_id = ns_ptr->nvme_id;
      read_cmd->ns_id = ns_ptr->ns_id;
      read_cmd->que_id = que_id;

      int32_t rc = spdk_nvme_ns_cmd_read(ns_ptr->ns,
                                         m_controller->qpair[que_id],
                                         dst, lba_id,
                                         lba_cnt,
                                         readCb, read_cmd, 0);
      if (rc != 0)
      {
        fprintf(stderr, "starting read I/O failed\n");
        exit(1);
      }

      return rc;
    }

    int32_t NVMeCtrl::pollCompletions(uint32_t que_id)
    {
      int32_t done_cnt = 0;
      if (que_id > m_controller->qpair.size())
      {
        done_cnt = -1;
        std::cerr << "Error que_id, large than max number of qpair." << std::endl;
        return done_cnt;
      }
      done_cnt =
          spdk_nvme_qpair_process_completions(m_controller->qpair[que_id], 0);
      return done_cnt;
    }

    void NVMeCtrl::resetFinishedQue(uint32_t que_id)
    {
      m_controller->q_finished[que_id] = 0;
    }

    uint64_t NVMeCtrl::getFinishedQue(uint32_t que_id)
    {
      return m_controller->q_finished[que_id];
    }
  } // namespace nvme
} // namespace minihypervec