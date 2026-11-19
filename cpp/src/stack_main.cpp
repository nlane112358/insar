#include "insar/stack/stack_application.hpp"

#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::vector<std::string> arguments;
    arguments.reserve(argc > 1 ? static_cast<size_t>(argc - 1) : 0);
    for (int i = 1; i < argc; ++i) {
        arguments.emplace_back(argv[i]);
    }
    return insar::stack::run_cli(arguments);
}
