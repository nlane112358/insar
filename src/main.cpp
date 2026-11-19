#include "insar/app/cli_options.hpp"
#include "insar/app/insar_application.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char** argv) {
    try {
        const insar::app::CliOptions options = insar::app::CliParser::parse(argc, argv);
        if (options.show_help) {
            insar::app::CliParser::print_usage(argv[0]);
            return 0;
        }

        const insar::app::InsarApplication app(options);
        return app.run();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Error: %s\n", ex.what());
        return EXIT_FAILURE;
    }
}
