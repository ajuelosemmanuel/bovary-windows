#include "llama.h"
#include "gguf.h"
#include "native_paths.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using metadata_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

static metadata_ptr read_metadata(const std::string & path) {
    gguf_init_params params = { true, nullptr };
    metadata_ptr result(gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!result) throw std::runtime_error("cannot read GGUF metadata");
    return result;
}

static std::string exact_pattern(const std::string & name) {
    std::string result = "^";
    for (char c : name) {
        if (std::strchr(".^$|()[]{}*+?\\", c)) result += '\\';
        result += c;
    }
    return result + "$";
}

static int run(const std::vector<std::string> & arguments) {
    std::string input, output;
    int threads = 4;
    bool dry_run = false;
    for (size_t i = 1; i < arguments.size(); ++i) {
        const std::string key = arguments[i];
        if (key == "--dry-run") { dry_run = true; continue; }
        if (i + 1 == arguments.size()) throw std::runtime_error("missing option value");
        const std::string value = arguments[++i];
        if (key == "--input") input = value;
        else if (key == "--output") output = value;
        else if (key == "--threads") threads = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (input.empty() || output.empty() || threads < 1 || threads > 64)
        throw std::runtime_error("use --input SOURCE --output NEW_FILE [--threads N] [--dry-run]");
    if (const char * stream = std::getenv("BOVARY_STREAM"); stream && std::strcmp(stream, "1") == 0)
        throw std::runtime_error("unset BOVARY_STREAM while converting weights");
    const std::filesystem::path destination = std::filesystem::absolute(bovary::path_from_utf8(output));
    auto partial = destination;
    partial += ".partial";
    const std::string partial_utf8 = bovary::path_to_utf8(partial);
    if (std::filesystem::exists(destination) || std::filesystem::exists(partial))
        throw std::runtime_error("output or partial output already exists; conversion never overwrites files");
    auto metadata = read_metadata(input);
    const auto architecture = gguf_find_key(metadata.get(), "general.architecture");
    if (architecture < 0 || gguf_get_kv_type(metadata.get(), architecture) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(metadata.get(), architecture), "gemma4") != 0)
        throw std::runtime_error("this experiment requires a Gemma4 GGUF");
    const int64_t count = gguf_get_n_tensors(metadata.get());
    if (count < 1 || count > 4096) throw std::runtime_error("unexpected tensor count");
    const std::regex gate_up("^blk\\.([0-9]+)\\.ffn_gate_up_exps\\.weight$");
    std::set<int> converted_layers;
    std::vector<std::string> patterns;
    std::vector<ggml_type> targets;
    patterns.reserve(count);
    targets.reserve(count);
    uint64_t source_payload = 0, expected_payload = 0;
    for (int64_t i = 0; i < count; ++i) {
        const std::string name = gguf_get_tensor_name(metadata.get(), i);
        const ggml_type original = gguf_get_tensor_type(metadata.get(), i);
        if (original != GGML_TYPE_F32 && original != GGML_TYPE_Q4_0 && original != GGML_TYPE_Q8_0)
            throw std::runtime_error("source contains an unexpected tensor type: " + name);
        auto target = original;
        std::smatch match;
        if (std::regex_match(name, match, gate_up)) {
            const int layer = std::stoi(match[1]);
            const auto * ne = gguf_get_tensor_ne(metadata.get(), i);
            if (layer < 0 || layer >= 30 || !converted_layers.insert(layer).second || original != GGML_TYPE_Q4_0 ||
                ne[0] != 2816 || ne[1] != 1408 || ne[2] != 128 || ne[3] != 1)
                throw std::runtime_error("unexpected gate/up expert layout: " + name);
            target = GGML_TYPE_Q3_K;
        }
        const uint64_t bytes = gguf_get_tensor_size(metadata.get(), i);
        source_payload += bytes;
        // Only Q4_0 -> Q3_K changes: blocks of 256 values use 144 -> 110 bytes.
        expected_payload += target == original ? bytes : bytes / 144 * 110;
        patterns.push_back(exact_pattern(name));
        targets.push_back(target);
    }
    if (converted_layers.size() != 30) throw std::runtime_error("expected thirty gate/up expert tensors");
    std::vector<llama_model_tensor_override> overrides;
    overrides.reserve(count + 1);
    for (size_t i = 0; i < patterns.size(); ++i) overrides.push_back({ patterns[i].c_str(), targets[i] });
    overrides.push_back({ nullptr, GGML_TYPE_COUNT });
    auto params = llama_model_quantize_default_params();
    params.ftype = LLAMA_FTYPE_MOSTLY_Q3_K_S;
    params.allow_requantize = true;
    params.pure = true;
    params.token_embedding_type = GGML_TYPE_Q8_0;
    params.tt_overrides = overrides.data();
    llama_model_kv_override metadata_overrides[3] = {};
    metadata_overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    std::strcpy(metadata_overrides[0].key, "general.name");
    std::strcpy(metadata_overrides[0].val_str, "Gemma4-26B-A4B-IT-custom-gateup-Q3_K-rest-source-exact");
    metadata_overrides[1].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
    std::strcpy(metadata_overrides[1].key, "quantize.recipe");
    std::strcpy(metadata_overrides[1].val_str, "Lossy Q4_0 requantization: 30 gate/up experts Q3_K; every other tensor copied from source");
    params.kv_overrides = metadata_overrides;
    params.max_buf_size = 64 * 1024 * 1024;
    params.nthread = threads;
    params.dry_run = dry_run;
    const uint64_t reserve = 1024ull * 1024 * 1024;
    if (!dry_run && std::filesystem::space(destination.parent_path()).available < expected_payload + gguf_get_data_offset(metadata.get()) + reserve)
        throw std::runtime_error("not enough free output space including 1 GiB reserve");
    fprintf(stderr, "experiment=gate_up_Q3_K_rest_source_exact lossy_requantization=true changed_tensors=30 source_payload_bytes=%llu expected_payload_bytes=%llu scratch_budget_bytes=%zu\n",
            (unsigned long long) source_payload, (unsigned long long) expected_payload, params.max_buf_size);
    const auto start = std::chrono::steady_clock::now();
    llama_backend_init();
    if (!dry_run) bovary::reserve_output(partial);
    if (llama_model_quantize(input.c_str(), partial_utf8.c_str(), &params) != 0)
        throw std::runtime_error("conversion failed; partial output is retained for inspection");
    llama_backend_free();
    if (!dry_run) {
        auto actual = read_metadata(partial_utf8);
        if (gguf_get_n_tensors(actual.get()) != count) throw std::runtime_error("converted tensor count differs");
        uint64_t actual_payload = 0;
        for (int64_t i = 0; i < count; ++i) {
            const auto j = gguf_find_tensor(actual.get(), gguf_get_tensor_name(metadata.get(), i));
            if (j < 0 || gguf_get_tensor_type(actual.get(), j) != targets[i] ||
                std::memcmp(gguf_get_tensor_ne(metadata.get(), i), gguf_get_tensor_ne(actual.get(), j), GGML_MAX_DIMS * sizeof(int64_t)))
                throw std::runtime_error("converted tensor type or shape differs from exact plan");
            actual_payload += gguf_get_tensor_size(actual.get(), j);
        }
        if (actual_payload != expected_payload) throw std::runtime_error("converted payload size differs from exact plan");
        bovary::publish_output(partial, destination);
        fprintf(stderr, "final_file_bytes=%llu\n", (unsigned long long) std::filesystem::file_size(destination));
    }
    fprintf(stderr, "elapsed_seconds=%.6f dry_run=%s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), dry_run ? "true" : "false");
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
    fprintf(stderr, "bovary-quantize: %s\n", error.what());
    return 1;
}
