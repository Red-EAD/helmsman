#include "meta_path.hpp"
namespace minihypervec
{
  PathConfig *PathConfig::getInstance()
  {
    static PathConfig instance;
    return &instance;
  }

  int32_t PathConfig::init(const std::string &path)
  {
    using json = nlohmann::json;
    std::string content;
    std::cout << "Loading PathConfig from " << path << std::endl;
    int32_t rc = util::read_file_to_string(path, content);
    if (rc != 0)
    {
      std::cerr << "Failed to read PathConfig from " << path << ", rc=" << rc
                << std::endl;
      return rc;
    }
    try
    {
      json j = json::parse(content);
      nvme_meta_path = j.at("nvme_meta_path").get<std::string>();
      release_index_path = j.at("release_index_path").get<std::string>();
    }
    catch (const std::exception &e)
    {
      std::cerr << "Failed to parse PathConfig JSON: " << e.what() << std::endl;
      return -1;
    }
    return 0;
  }

  int32_t PathConfig::save(const std::string &path)
  {
    using json = nlohmann::json;
    json j;
    j["nvme_meta_path"] = nvme_meta_path;
    j["release_index_path"] = release_index_path;

    std::string content = j.dump(4);
    int32_t rc = util::persist_string_atomic_fsync(path, content);
    if (rc != 0)
    {
      std::cerr << "Failed to write PathConfig to " << path << ", rc=" << rc
                << std::endl;
      return rc;
    }
    return 0;
  }

  void PathConfig::printConfig() const
  {
    std::cout << "PathConfig:" << std::endl;
    std::cout << "  NVMe Meta Path: " << nvme_meta_path << std::endl;
    std::cout << "  Release Index Path: " << release_index_path << std::endl;
  }

  namespace nvme
  {
    std::string getNVMeMetaPath() noexcept
    {
      std::string s;
      PathConfig *pc = PathConfig::getInstance();
      std::string &nvme_meta_path = pc->nvme_meta_path;
      s.reserve(nvme_meta_path.size() + nvme_meta_filename.size());
      s.append(nvme_meta_path.data(), nvme_meta_path.size());
      s.append(nvme_meta_filename.data(), nvme_meta_filename.size());
      return s;
    }

  }

  namespace release
  {
    namespace constants
    {
      std::string getHardwareMetaPath() noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + hardware_meta_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(hardware_meta_filename.data(), hardware_meta_filename.size());
        return s;
      }
      std::string getIndexMetaPath(const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  index_meta_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(index_meta_filename.data(), index_meta_filename.size());
        return s;
      }
      std::string getRawdataPath(const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  rawdata_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(rawdata_filename.data(), rawdata_filename.size());
        return s;
      }
      std::string getClusterIDsPath(const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  cluster_ids_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(cluster_ids_filename.data(), cluster_ids_filename.size());
        return s;
      }
      std::string getClusterNormsPath(const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  cluster_norms_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(cluster_norms_filename.data(), cluster_norms_filename.size());
        return s;
      }
      std::string getCentroidsIndexPath(const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  centroids_index_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(centroids_index_filename.data(), centroids_index_filename.size());
        return s;
      }

      std::string getClusterExtraIDsPath(
          const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  cluster_extra_ids_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(cluster_extra_ids_filename.data(),
                 cluster_extra_ids_filename.size());
        return s;
      }

      std::string getClusterExtraNormsPath(
          const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  cluster_extra_norms_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(cluster_extra_norms_filename.data(),
                 cluster_extra_norms_filename.size());
        return s;
      }

      std::string getClusterMapPath(const std::string &collection_name) noexcept
      {
        std::string s;
        PathConfig *pc = PathConfig::getInstance();
        std::string &release_index_path = pc->release_index_path;
        s.reserve(release_index_path.size() + collection_name.size() + 1 +
                  cluster_map_filename.size());
        s.append(release_index_path.data(), release_index_path.size());
        s.append(collection_name.data(), collection_name.size());
        s.append("/");
        s.append(collection_name.data(), collection_name.size());
        s.append(cluster_map_filename.data(), cluster_map_filename.size());
        return s;
      }

    }
  }

  void printAuthorInfo() noexcept
  {
    try
    {
      const int fd = STDOUT_FILENO;

      auto write_all = [&](const std::string &s) noexcept
      {
        const char *p = s.c_str();
        size_t left = s.size();
        while (left)
        {
          ssize_t n = ::write(fd, p, left);
          if (n <= 0)
            break;
          p += static_cast<size_t>(n);
          left -= static_cast<size_t>(n);
        }
      };

      if (!::isatty(fd))
      {
        write_all(
            "MiniHyperVec\n"
            "Author: Yuchen Huang\n"
            "Email : ychuang@stu.ecnu.edu.cn\n"
            "        ychuangecnu@gmail.com\n"
            "        huangyuchen2@xiaohongshu.com\n"
            "Author: Baiteng Ma\n"
            "Email : btma@stu.ecnu.edu.cn\n"
            "        mabaiteng@xiaohongshu.com\n");
        return;
      }

      winsize ws{};
      if (::ioctl(fd, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0 || ws.ws_row == 0)
      {
        write_all(
            "MiniHyperVec\n"
            "Author: Yuchen Huang\n"
            "Email : ychuang@stu.ecnu.edu.cn\n"
            "        ychuangecnu@gmail.com\n"
            "        huangyuchen2@xiaohongshu.com\n"
            "Author: Baiteng Ma\n"
            "Email : btma@stu.ecnu.edu.cn\n"
            "        mabaiteng@xiaohongshu.com\n");
        return;
      }

      const int rows = static_cast<int>(ws.ws_row);
      const int cols = static_cast<int>(ws.ws_col);

      const std::vector<std::string> logo = {
          R"( _   _                         __      __                     _    _   _____   _   _   _____  )",
          R"(| | | |                        \ \    / /                    |  \/  | |_   _| | \ | | |_   _| )",
          R"(| |_| | _   _ _ __   ___ _ __   \ \  / /__   ___    ______   | \  / |   | |   |  \| |   | |   )",
          R"(|  _  || | | | '_ \ / _ \ '__|   \ \/ / _ \ / __|  |______|  | |\/| |   | |   |     |   | |   )",
          R"(| | | || |_| | |_) |  __/ |       \  /  __/ (__              | |  | |  _| |_  | |\  |  _| |_  )",
          R"(\_| |_/ \__, | .__/ \___|_|        \/ \___|\___|             |_|  |_| |_____| |_| \_| |_____| )",
          R"(         __/ | |                                                                              )",
          R"(        |___/|_|                                                                              )",
          R"(                              H y p e r   V e c   -   M i n i                                 )"};

      const std::vector<std::string> author = {
          "Author: Yuchen Huang",
          "Department: Engine Architecture of RedNote",
          "Email : ychuang@stu.ecnu.edu.cn",
          "        ychuangecnu@gmail.com",
          "        huangyuchen2@xiaohongshu.com",
          "Author: Baiteng Ma",
          "Department: Engine Architecture of RedNote",
          "Email : btma@stu.ecnu.edu.cn",
          "        mabaiteng@xiaohongshu.com",

      };

      std::vector<std::string> content;
      content.insert(content.end(), logo.begin(), logo.end());
      content.push_back("");
      content.push_back("");
      content.push_back("");
      content.insert(content.end(), author.begin(), author.end());

      int innerMax = 0;
      for (const auto &s : content)
        innerMax = std::max(innerMax, (int)s.size());

      const int dividerIdx = (int)logo.size() + 1;
      content[dividerIdx] = std::string(innerMax, '-');

      const int boxW = innerMax + 4;
      const int boxH = (int)content.size() + 2;

      if (boxW > cols || boxH > rows || cols < 40 || rows < 10)
      {
        write_all(
            "MiniHyperVec\n"
            "Author: Yuchen Huang\n"
            "Email : ychuang@stu.ecnu.edu.cn\n"
            "        ychuangecnu@gmail.com\n"
            "        huangyuchen2@xiaohongshu.com\n"
            "Author: Baiteng Ma\n"
            "Email : btma@stu.ecnu.edu.cn\n"
            "        mabaiteng@xiaohongshu.com\n");
        return;
      }

      const int startRow = std::max(1, (rows - boxH) / 2 + 1);
      const int startCol = std::max(1, (cols - boxW) / 2 + 1);

      auto cursor = [](int r, int c) -> std::string
      {
        return "\033[" + std::to_string(r) + ";" + std::to_string(c) + "H";
      };

      const std::string dim = "\033[2m";
      const std::string bold = "\033[1m";
      const std::string cyan = "\033[1;36m";
      const std::string yel = "\033[1;33m";
      const std::string reset = "\033[0m";

      std::string out;
      out.reserve(8192);

      out += "\033[?25l";
      out += "\033[2J\033[H";

      const std::string top = "+" + std::string(innerMax + 2, '-') + "+";
      out += cursor(startRow, startCol);
      out += dim + top + reset;

      for (int i = 0; i < (int)content.size(); ++i)
      {
        const int r = startRow + 1 + i;

        const bool isLogo = i < (int)logo.size();
        const bool isAuthor = i >= (int)content.size() - (int)author.size();
        const bool isDiv = (i == dividerIdx);

        std::string line = content[i];
        if ((int)line.size() < innerMax)
          line += std::string(innerMax - line.size(), ' ');

        out += cursor(r, startCol);
        out += dim + "|" + reset;
        out += " ";

        if (isDiv)
        {
          out += dim + line + reset;
        }
        else if (isLogo)
        {
          out += cyan + line + reset;
        }
        else if (isAuthor)
        {
          out += yel + bold + line + reset;
        }
        else
        {
          out += line;
        }

        out += " ";
        out += dim + "|" + reset;
      }

      out += cursor(startRow + boxH - 1, startCol);
      out += dim + top + reset;

      out += cursor(rows, 1);
      out += "\033[?25h";

      write_all(out);
    }
    catch (...)
    {
    }
  }
}