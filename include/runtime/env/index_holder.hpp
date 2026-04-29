#pragma once
#include "index/hyperconst_imp.hpp"

namespace minihypervec
{
    namespace runtime
    {
        class IndexHolder
        {
        private:
            std::string collection_name;
            std::shared_ptr<index::IndexAbs<int32_t>> held_index;

        public:
            static IndexHolder *getInstance();

        public:
            int32_t getIndex(const std::string &collection_name,
                               std::shared_ptr<index::IndexAbs<int32_t>> &index);

            int32_t initIndex(const std::string &collection_name,
                               const std::shared_ptr<index::IndexAbs<int32_t>> &index);
        };
    } // namespace runtime
} // namespace minihypervec