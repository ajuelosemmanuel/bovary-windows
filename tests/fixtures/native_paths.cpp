// No engine, model, or hardware calls: exercise the native path boundary only.
#include "../../src/native_paths.h"
#include "bovary-stream.h"
#include <fstream>
#include <iostream>
#include <iterator>

static int run(const std::vector<std::string> & arguments) {
    const bool check_collisions = arguments.size() == 4 && arguments[3] == "--collisions";
    if (arguments.size() != 3 && !check_collisions) throw std::runtime_error("expected input and output paths");
    const auto source = bovary::path_from_utf8(arguments[1]);
    const auto destination = std::filesystem::absolute(bovary::path_from_utf8(arguments[2]));
    auto partial = destination;
    partial += ".partial";
    if (check_collisions) {
        try {
            bovary::reserve_output(partial);
        } catch (const std::system_error &) {
            try {
                bovary::publish_output(partial, destination);
            } catch (const std::filesystem::filesystem_error &) {
                return 0;
            }
            throw std::runtime_error("publishing replaced an existing destination");
        }
        throw std::runtime_error("reservation replaced an existing partial file");
    }
    if (!std::filesystem::exists(source) || std::filesystem::exists(destination) ||
            std::filesystem::exists(partial) || std::filesystem::space(destination.parent_path()).available == 0)
        throw std::runtime_error("unexpected path fixture state");
    std::ifstream input(source, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open fixture input");
    const std::string content(std::istreambuf_iterator<char>(input), {});
    bovary::reserve_output(partial);
    std::ofstream output(partial, std::ios::binary);
    if (!output || !(output << content)) throw std::runtime_error("cannot write fixture output");
    output.close();
    bovary::publish_output(partial, destination);
    const auto model_environment = bovary_stream_path_environment("BOVARY_STREAM_MODEL");
    const auto trace_environment = bovary_stream_path_environment("BOVARY_STREAM_TRACE");
    if (model_environment != arguments[1] || trace_environment != arguments[2] + ".trace")
        throw std::runtime_error("Unicode path environment was not preserved as UTF-8");
    FILE * trace = ggml_fopen(trace_environment.c_str(), "wb");
    if (!trace) throw std::runtime_error("cannot open Unicode trace output");
    const char trace_text[] = "Unicode trace fixture\n";
    const bool written = fwrite(trace_text, 1, sizeof(trace_text) - 1, trace) == sizeof(trace_text) - 1;
    const int closed = fclose(trace);
    if (!written || closed != 0) throw std::runtime_error("cannot write Unicode trace output");
    trace = ggml_fopen(trace_environment.c_str(), "rb");
    if (!trace) throw std::runtime_error("cannot reopen Unicode trace output");
    char restored[sizeof(trace_text)] = {};
    const size_t bytes = fread(restored, 1, sizeof(trace_text) - 1, trace);
    fclose(trace);
    if (bytes != sizeof(trace_text) - 1 || std::string(restored, bytes) != trace_text)
        throw std::runtime_error("Unicode trace readback differs");
    std::cout << arguments[1] << '\n' << bovary::path_to_utf8(destination) << '\n' << content;
    return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t ** argv) try {
    return run(bovary::utf8_arguments(argc, argv));
#else
int main(int argc, char ** argv) try {
    return run(std::vector<std::string>(argv, argv + argc));
#endif
} catch (const std::exception & error) {
    std::cerr << "bovary-paths-test: " << error.what() << '\n';
    return 1;
}
