#pragma once

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace bovary {

// llama.cpp accepts UTF-8 paths; the Windows CRT's narrow argv uses the active
// ANSI code page. Convert from the wide entry point without that lossy step.
#ifdef _WIN32
inline std::string to_utf8(const wchar_t * value) {
    const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                                        nullptr, 0, nullptr, nullptr);
    if (bytes == 0) throw std::runtime_error("cannot encode Windows argument as UTF-8");
    std::string result(bytes, '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                            result.data(), bytes, nullptr, nullptr) != bytes)
        throw std::runtime_error("cannot encode Windows argument as UTF-8");
    result.resize(bytes - 1);
    return result;
}

inline std::vector<std::string> utf8_arguments(int argc, wchar_t ** argv) {
    std::vector<std::string> result;
    result.reserve(argc);
    for (int i = 0; i < argc; ++i) result.push_back(to_utf8(argv[i]));
    return result;
}
#endif

inline std::filesystem::path path_from_utf8(const std::string & value) {
    return std::filesystem::u8path(value);
}

inline std::string path_to_utf8(const std::filesystem::path & value) {
    return value.u8string();
}

inline void reserve_output(const std::filesystem::path & path) {
#ifdef _WIN32
    FILE * output = _wfopen(path.c_str(), L"wbx");
#else
    FILE * output = std::fopen(path.c_str(), "wbx");
#endif
    if (!output || std::fclose(output) != 0)
        throw std::system_error(errno, std::generic_category(), "cannot reserve partial output");
}

inline void publish_output(const std::filesystem::path & staged, const std::filesystem::path & destination) {
#ifdef _WIN32
    if (!MoveFileW(staged.c_str(), destination.c_str()))
        throw std::filesystem::filesystem_error("cannot publish partial output", staged, destination,
            std::error_code(GetLastError(), std::system_category()));
#else
    std::filesystem::create_hard_link(staged, destination);
    std::filesystem::remove(staged);
#endif
}

} // namespace bovary
