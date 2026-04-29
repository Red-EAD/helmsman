#include "runtime/env/minihypervec_env.hpp"

int main(int argc, char **argv)
{
  if (argc != 2 || argv[1] == nullptr || argv[1][0] == '\0')
  {
    std::cerr << "Usage: " << argv[0] << " <collection_name>\n";
    return 1;
  }

  const std::string collection_name = argv[1];

  auto *minihypervec_env = minihypervec::runtime::MiniHyperVecEnv::getInstance();
  if (minihypervec_env->initForDeploy() != 0)
  {
    std::cerr << "Error: initForDeploy failed\n";
    return 1;
  }
  int ret_code = 0;
  {
    std::cout << "Env initialized. Start to deploy index: " << collection_name << std::endl;
    auto offline_worker_handle = minihypervec::runtime::OfflineWorker::getInstance();
    std::cout << "Start to deploy index..." << std::endl;
    int ret = offline_worker_handle->deployIndex(collection_name);
    if (ret != 0)
    {
      std::cerr << "Deploy index failed, ret=" << ret << "\n";
      ret_code = -1;
    }
    else
    {
      std::cout << "Deploy index success\n";
    }
  }
  minihypervec_env->shutdownForDeploy();
  return ret_code;
}