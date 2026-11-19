#pragma once

#include <string>
#include <vector>

namespace insar::app {

// Run the pairwise application in-process. Arguments omit argv[0].
int run_pairwise_cli(const std::vector<std::string>& arguments);

}  // namespace insar::app
