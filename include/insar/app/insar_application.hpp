#pragma once

#include "insar/app/cli_options.hpp"
#include "insar/io/slc_source.hpp"

#include <memory>

namespace insar::app {

class InsarApplication {
public:
    explicit InsarApplication(CliOptions options);

    int run() const;

private:
    std::unique_ptr<ISlcSource> create_slc_source() const;

    CliOptions options_;
};

} // namespace insar::app
