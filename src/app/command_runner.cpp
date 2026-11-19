#include "insar/app/command_runner.hpp"

#include "insar/app/cli_options.hpp"
#include "insar/app/insar_application.hpp"

#include <cstdio>
#include <exception>
#include <vector>

namespace insar::app {

int run_pairwise_cli(const std::vector<std::string>& arguments) {
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.emplace_back("insar_proto");
    storage.insert(storage.end(), arguments.begin(), arguments.end());

    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& value : storage) {
        argv.push_back(value.data());
    }

    try {
        const CliOptions options = CliParser::parse(static_cast<int>(argv.size()), argv.data());
        if (options.show_help) {
            CliParser::print_usage(argv.front());
            return 0;
        }
        return InsarApplication(options).run();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}

}  // namespace insar::app
