#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace furiaopt
{

struct ExampleArgs
{
    std::string config_dir;
    std::string logs_output_dir;
};

inline ExampleArgs parse_example_args(int argc, char** argv)
{
    ExampleArgs args;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--config-dir" && i + 1 < argc)
            args.config_dir = argv[++i];
        else if (a == "--logs-output-dir" && i + 1 < argc)
            args.logs_output_dir = argv[++i];
    }
    if (args.config_dir.empty() || args.logs_output_dir.empty())
    {
        std::cerr << "usage: " << argv[0] << " --config-dir <dir> --logs-output-dir <dir>\n";
        std::exit(1);
    }
    return args;
}

} // namespace furiaopt
