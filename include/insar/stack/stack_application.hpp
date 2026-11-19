#pragma once

#include <string>
#include <vector>

namespace insar::stack {

// Run the stack application in-process. Arguments omit argv[0].
int run_cli(const std::vector<std::string>& arguments);

}  // namespace insar::stack
