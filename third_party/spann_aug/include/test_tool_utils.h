#pragma once

#include "root.h"
#include "spann_index.h"

namespace spann
{
    struct BuildArtifacts
    {
        std::string centroids_index_path;
        std::string cluster_ids_path;
        std::string cluster_norms_path;
        std::string index_meta_path;
    };

    struct ParsedArgs
    {
        std::unordered_map<std::string, std::string> values;

        bool Has(const std::string &key) const
        {
            return values.find(key) != values.end();
        }

        const std::string *Get(const std::string &key) const
        {
            auto it = values.find(key);
            if (it == values.end())
            {
                return nullptr;
            }
            return &it->second;
        }
    };

    struct BuildIndexConfig
    {
        std::string spann_dir;
        std::string output_dir;
        std::string prefix = "spann";
        int fill_target_size = 64;
        int fill_neighbor_topk = 16;
        bool fill_count_head = true;
        int fill_num_threads = 32;
        int hnsw_M = 16;
        int hnsw_ef_construction = 200;
        int hnsw_ef_search = 128;
    };

    inline BuildIndexConfig DefaultBuildIndexConfig()
    {
        return {};
    }

    inline ParsedArgs ParseArgs(int argc, char **argv)
    {
        ParsedArgs parsed;
        for (int i = 1; i < argc; ++i)
        {
            std::string token = argv[i];
            if (!token.empty() && token.rfind("--", 0) == 0)
            {
                const size_t eq_pos = token.find('=');
                if (eq_pos != std::string::npos)
                {
                    parsed.values[token.substr(2, eq_pos - 2)] = token.substr(eq_pos + 1);
                    continue;
                }

                const std::string key = token.substr(2);
                if (key == "help")
                {
                    parsed.values[key] = "1";
                    continue;
                }
                if (i + 1 >= argc)
                {
                    throw std::invalid_argument("Missing value for argument --" + key);
                }
                parsed.values[key] = argv[++i];
                continue;
            }

            throw std::invalid_argument("Unsupported positional argument: " + token);
        }
        return parsed;
    }

    inline int ParseIntArg(const ParsedArgs &args, const std::string &key, int fallback)
    {
        const std::string *value = args.Get(key);
        if (value == nullptr)
        {
            return fallback;
        }
        return std::stoi(*value);
    }

    inline bool ParseBoolArg(const ParsedArgs &args, const std::string &key, bool fallback)
    {
        const std::string *value = args.Get(key);
        if (value == nullptr)
        {
            return fallback;
        }

        if (*value == "1" || *value == "true" || *value == "TRUE")
        {
            return true;
        }
        if (*value == "0" || *value == "false" || *value == "FALSE")
        {
            return false;
        }
        throw std::invalid_argument("Invalid boolean value for --" + key + ": " + *value);
    }

    inline std::string ParseStringArg(const ParsedArgs &args,
                                      const std::string &key,
                                      const std::string &fallback)
    {
        const std::string *value = args.Get(key);
        return value == nullptr ? fallback : *value;
    }

    inline BuildIndexConfig ParseBuildIndexConfig(const ParsedArgs &args)
    {
        BuildIndexConfig config = DefaultBuildIndexConfig();
        config.spann_dir = ParseStringArg(args, "spann-dir", config.spann_dir);
        config.output_dir = ParseStringArg(args, "output-dir", config.output_dir);
        config.prefix = ParseStringArg(args, "prefix", config.prefix);
        config.fill_target_size = ParseIntArg(args, "fill-target-size", config.fill_target_size);
        config.fill_neighbor_topk = ParseIntArg(args, "fill-neighbor-topk", config.fill_neighbor_topk);
        config.fill_count_head = ParseBoolArg(args, "fill-count-head", config.fill_count_head);
        config.fill_num_threads = ParseIntArg(args, "fill-num-threads", config.fill_num_threads);
        config.hnsw_M = ParseIntArg(args, "hnsw-M", config.hnsw_M);
        config.hnsw_ef_construction = ParseIntArg(args, "hnsw-ef-construction", config.hnsw_ef_construction);
        config.hnsw_ef_search = ParseIntArg(args, "hnsw-ef-search", config.hnsw_ef_search);

        if (config.spann_dir.empty())
        {
            throw std::invalid_argument("--spann-dir is required");
        }
        if (config.output_dir.empty())
        {
            throw std::invalid_argument("--output-dir is required");
        }
        return config;
    }

    inline BuildArtifacts MakeArtifacts(const BuildIndexConfig &config)
    {
        const std::string &p = config.prefix;
        return {
            config.output_dir + "/" + p + "_centroids_index.bin",
            config.output_dir + "/" + p + "_cluster_ids.bin",
            config.output_dir + "/" + p + "_cluster_norms.bin",
            config.output_dir + "/" + p + "_index_meta.json"};
    }

    inline bool ShouldShowSummary(const ParsedArgs &args, bool fallback)
    {
        return ParseBoolArg(args, "show-summary", fallback);
    }

    inline void PrintBuildUsage(const char *program)
    {
        std::cout
            << "Usage: " << program << " [options]\n"
            << "  --spann-dir PATH             (required) SPANN index directory\n"
            << "  --output-dir PATH            (required) output directory\n"
            << "  --prefix STRING              output filename prefix (default: spann)\n"
            << "  --fill-target-size INT       target cluster size (default: 64)\n"
            << "  --fill-neighbor-topk INT     neighbor topk for filling (default: 16)\n"
            << "  --fill-count-head true|false count head in cluster size (default: true)\n"
            << "  --fill-num-threads INT       filling threads (default: 32)\n"
            << "  --hnsw-M INT                HNSW M parameter (default: 16)\n"
            << "  --hnsw-ef-construction INT   HNSW ef_construction (default: 200)\n"
            << "  --hnsw-ef-search INT         HNSW ef_search (default: 128)\n"
            << "  --show-summary true|false    show index summary (default: true)\n";
    }

    inline std::unique_ptr<SpannIndex> OpenSpannIndex(const BuildIndexConfig &config)
    {
        return std::make_unique<SpannIndex>(
            config.spann_dir + "/SPTAGHeadVectors.bin",
            config.spann_dir + "/SPTAGHeadVectorIDs.bin",
            config.spann_dir + "/SPTAGFullList.bin");
    }

    inline void PrepareFilledIndex(SpannIndex &index, const BuildIndexConfig &config, bool show_summary)
    {
        if (show_summary)
        {
            index.Summary();
        }
        index.LoadAllClusters();
        index.BuildHeadHNSW(
            config.hnsw_M,
            config.hnsw_ef_construction,
            config.hnsw_ef_search);
        index.PerformFilling(
            config.fill_target_size,
            config.fill_neighbor_topk,
            config.fill_count_head,
            config.fill_num_threads);
    }

    inline void SaveArtifacts(SpannIndex &index, const BuildArtifacts &artifacts)
    {
        index.saveNorms(artifacts.cluster_norms_path);
        index.saveHeadIndex(artifacts.centroids_index_path);
        index.saveClusterIds(artifacts.cluster_ids_path);
    }

    inline void SaveIndexMeta(const BuildIndexConfig &config,
                              const SpannIndex &index,
                              const std::string &output_path)
    {
        const int dim = index.GetDim();
        const int centroid_num = index.GetNumHeads();
        const int max_elements = index.GetTotalDocs();

        std::ofstream ofs(output_path);
        if (!ofs.is_open())
        {
            throw std::runtime_error("Failed to open index meta file: " + output_path);
        }

        ofs << "{\n"
            << "  \"collection_name\": \"" << config.prefix << "\",\n"
            << "  \"vec_type\": \"INT8\",\n"
            << "  \"index_type\": \"HV_CONST\",\n"
            << "  \"build_param\": {\n"
            << "    \"centroid_build_param\": {\n"
            << "      \"build_ef\": " << config.hnsw_ef_construction << ",\n"
            << "      \"dim\": " << dim << ",\n"
            << "      \"inner_M\": " << config.hnsw_M << ",\n"
            << "      \"max_elements\": " << centroid_num << ",\n"
            << "      \"metric\": \"Euclidean\",\n"
            << "      \"search_ef\": " << config.hnsw_ef_search << "\n"
            << "    },\n"
            << "    \"centroid_index_type\": \"HNSW\",\n"
            << "    \"centroid_num\": " << centroid_num << ",\n"
            << "    \"cluster_size\": " << config.fill_target_size << ",\n"
            << "    \"dim\": " << dim << ",\n"
            << "    \"max_elements\": " << max_elements << ",\n"
            << "    \"metric\": \"Euclidean\"\n"
            << "  }\n"
            << "}\n";

        ofs.close();
        std::cout << "index_meta saved: " << output_path << "\n";
    }

} // namespace spann
