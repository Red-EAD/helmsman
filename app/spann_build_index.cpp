#include "test_tool_utils.h"

int main(int argc, char **argv)
{
    using namespace spann;

    try
    {
        const ParsedArgs args = ParseArgs(argc, argv);
        if (args.Has("help"))
        {
            PrintBuildUsage(argv[0]);
            return 0;
        }

        const BuildIndexConfig build_config = ParseBuildIndexConfig(args);
        const BuildArtifacts artifacts = MakeArtifacts(build_config);
        const bool show_summary = ShouldShowSummary(args, true);

        auto index = OpenSpannIndex(build_config);
        PrepareFilledIndex(*index, build_config, show_summary);
        index->performNorms();
        SaveArtifacts(*index, artifacts);
        SaveIndexMeta(build_config, *index, artifacts.index_meta_path);

        std::cout << "\n===== Build Completed =====\n";
        std::cout << "centroids_index : " << artifacts.centroids_index_path << "\n";
        std::cout << "cluster_ids     : " << artifacts.cluster_ids_path << "\n";
        std::cout << "cluster_norms   : " << artifacts.cluster_norms_path << "\n";
        std::cout << "index_meta      : " << artifacts.index_meta_path << "\n";

        return 0;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "build tool failed: " << ex.what() << "\n";
        PrintBuildUsage(argv[0]);
        return 1;
    }
}
