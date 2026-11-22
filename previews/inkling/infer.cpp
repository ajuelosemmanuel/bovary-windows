#include "llama.h"
#include "ggml-backend.h"
#include "native_paths.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <type_traits>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using clock_type = std::chrono::steady_clock;
static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}


static void frame(char kind, const std::string & payload) {
    fprintf(stdout, "%c %zu\n", kind, payload.size());
    if (fwrite(payload.data(), 1, payload.size(), stdout) != payload.size())
        throw std::runtime_error("worker output closed");
    fflush(stdout);
}

static std::string json_string(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}

// Propose a continuation only when the most recent generated tokens also
// occur in this request's prompt. The caller verifies each proposal with
// the model's actual greedy logits before emitting it.
static std::vector<llama_token> draft_from_prompt(const std::vector<llama_token> & prompt,
        const std::vector<int> & generated, int maximum) {
    for (int n = std::min<int>(6, generated.size()); n >= 2; --n) {
        const auto match = std::find_end(prompt.begin(), prompt.end() - 1,
                generated.end() - n, generated.end());
        if (match != prompt.end() - 1) {
            const auto next = match + n;
            return {next, next + std::min<std::ptrdiff_t>(maximum, prompt.end() - next)};
        }
    }
    return {};
}

// A code edit often starts by reproducing a declaration from the current
// request. This only provides draft tokens. It never supplies answer logits.
static std::vector<llama_token> draft_code_prefix(const std::string & user,
        const llama_vocab * vocab, int limit) {
    if (limit <= 0) return {};
    for (size_t at = 0; at < user.size();) {
        const bool declaration = user.compare(at, 4, "def ") == 0 ||
                user.compare(at, 10, "async def ") == 0 || user.compare(at, 6, "class ") == 0;
        if (declaration) {
            const auto size = static_cast<int>(user.size() - at);
            const int count = -llama_tokenize(vocab, user.data() + at, size, nullptr, 0, false, false);
            if (count <= 0) return {};
            std::vector<llama_token> result(count);
            if (llama_tokenize(vocab, user.data() + at, size, result.data(), count, false, false) != count)
                throw std::runtime_error("source draft tokenization failed");
            result.resize(std::min(count, limit));
            return result;
        }
        at = user.find('\n', at);
        if (at == std::string::npos) break;
        ++at;
    }
    return {};
}

static std::string run_request(llama_model * model, const std::string & user,
        int predict, int context, int threads, int prompt_batch, int cache_mib,
        const std::string & choices, const std::string & logits_path, bool worker, bool direct_answer, bool no_op_offload,
        clock_type::time_point start, double model_load) {
    const std::string prompt =
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>" + user + "<|end_message|>" +
        (direct_answer ? "<|message_model|><|content_text|>" : "");
    auto * vocab = llama_model_get_vocab(model);
    int n_prompt = -llama_tokenize(vocab, prompt.data(), (int) prompt.size(), nullptr, 0, true, true);
    if (n_prompt < 1 || n_prompt >= context || predict > context - n_prompt)
        throw std::runtime_error("prompt/generation exceeds context");
    const int generation_limit = predict == 0 ? context - n_prompt : predict;
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.data(), (int) prompt.size(), tokens.data(), n_prompt, true, true) != n_prompt)
        throw std::runtime_error("tokenization failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    if (const char * rightsize = std::getenv("BOVARY_STREAM_RIGHTSIZE_CTX");
            rightsize && std::strcmp(rightsize, "1") == 0) {
        const char * stream = std::getenv("BOVARY_STREAM");
        constexpr char expected_architecture[] = "inkling2_loop";
        char architecture[32];
        if (!stream || std::strcmp(stream, "1") != 0 ||
                llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) != sizeof(expected_architecture) - 1 ||
                std::strcmp(architecture, expected_architecture) != 0) {
            throw std::runtime_error("BOVARY_STREAM_RIGHTSIZE_CTX requires streamed Inkling 2 Loop");
        }
        if (predict > 0) {
            constexpr uint32_t alignment = 256;
            const uint32_t needed = n_prompt + predict;
            const uint32_t rounded = (needed + alignment - 1) / alignment * alignment;
            cp.n_ctx = std::min<uint32_t>(rounded, context);
        }
    }
    cp.n_batch = prompt_batch;
    cp.n_ubatch = prompt_batch;
    int requested_depth = 3;
    if (const char * value = std::getenv("BOVARY_STREAM_LOOKUP_DEPTH")) {
        size_t end = 0;
        requested_depth = std::stoi(value, &end);
        if (value[end] != '\0' || requested_depth < 1 || requested_depth > 16)
            throw std::runtime_error("prompt lookup depth must be 1 to 16");
    }
    // The recurrent batch splitter also needs room for the seed and one
    // prefix token beyond the rollback tail.
    const int lookup_depth = std::min(requested_depth, std::max(0, prompt_batch - 2));
    cp.n_rs_seq = lookup_depth;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.moe_cache_size = (size_t) cache_mib * 1024 * 1024;
    cp.moe_cache_cpu_share = 0;
    if (no_op_offload) cp.op_offload = false;
#ifdef BOVARY_CPU_ONLY
    cp.offload_kqv = false;
    cp.op_offload = false;
#endif
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = false;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context allocation failed");
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_guard(ctx, llama_free);
    if (llama_n_rs_seq(ctx) < (uint32_t)lookup_depth)
        throw std::runtime_error("recurrent rollback unavailable for prompt lookup");
    const double ready = seconds(start);
    auto * batch = llama_batch_ext_init(ctx);
    if (!batch) throw std::runtime_error("batch allocation failed");
    std::unique_ptr<std::remove_pointer_t<decltype(batch)>, decltype(&llama_batch_ext_free)> batch_guard(batch, llama_batch_ext_free);
    auto decode = [&](llama_token token, int pos, bool output) {
        llama_batch_ext_clear(batch);
        int idx = llama_batch_ext_add_token(batch, 0, token);
        if (idx < 0) throw std::runtime_error("batch token allocation failed");
        llama_pos position = pos;
        if (!llama_batch_ext_set_pos(batch, idx, &position) || !llama_batch_ext_set_output_logits(batch, idx, output))
            throw std::runtime_error("batch setup failed");
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) throw std::runtime_error("decode failed");
        llama_synchronize(ctx);
    };
    const auto prompt_start = clock_type::now();
    auto prefill = [&](const std::vector<llama_token> & input, int outputs_from) {
        for (int first = 0; first < int(input.size()); first += prompt_batch) {
            llama_batch_ext_clear(batch);
            const int end = std::min(int(input.size()), first + prompt_batch);
            for (int i = first; i < end; ++i) {
                const int idx = llama_batch_ext_add_token(batch, 0, input[i]);
                if (idx < 0) throw std::runtime_error("prompt batch token allocation failed");
                const llama_pos position = i;
                if (!llama_batch_ext_set_pos(batch, idx, &position) ||
                    !llama_batch_ext_set_output_logits(batch, idx, i >= outputs_from))
                    throw std::runtime_error("prompt batch setup failed");
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0)
                throw std::runtime_error("prompt batch processing failed");
            llama_synchronize(ctx);
        }
    };
    int draft_cap = 48;
    if (const char * setting = std::getenv("BOVARY_STREAM_JOINT_DRAFT")) {
        size_t parsed = 0;
        draft_cap = std::stoi(setting, &parsed);
        if (setting[parsed] != '\0' || draft_cap < 0 || draft_cap > 128)
            throw std::runtime_error("joint draft must be between zero and 128");
    }
    const int room = prompt_batch - (1 + (n_prompt - 1) % prompt_batch);
    auto draft = direct_answer && choices.empty() && logits_path.empty()
            ? draft_code_prefix(user, vocab, std::min({draft_cap, room, generation_limit}))
            : std::vector<llama_token>{};
    const int joint_length = int(draft.size());
    auto joint_input = tokens;
    joint_input.insert(joint_input.end(), draft.begin(), draft.end());
    prefill(joint_input, n_prompt - 1);
    const double prompt_s = seconds(prompt_start);
    const int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> choice_ids;
    std::vector<double> choice_logits, choice_logprobs;
    if (!choices.empty()) {
        float * logits = llama_get_logits_ith(ctx, -1);
        const double maximum = *std::max_element(logits, logits + vocab_size);
        double denominator = 0;
        for (int i = 0; i < vocab_size; ++i) denominator += std::exp((double) logits[i] - maximum);
        const double log_z = maximum + std::log(denominator);
        if (!std::isfinite(log_z)) throw std::runtime_error("nonfinite choice distribution");
        for (char choice : choices) {
            if (choice < 'A' || choice > 'Z' || choices.find(choice) != choices.rfind(choice))
                throw std::runtime_error("choices must be distinct uppercase ASCII letters");
            llama_token id = LLAMA_TOKEN_NULL;
            if (llama_tokenize(vocab, &choice, 1, &id, 1, false, false) != 1)
                throw std::runtime_error("each choice must tokenize to exactly one token");
            choice_ids.push_back(id);
            choice_logits.push_back(logits[id]);
            choice_logprobs.push_back((double) logits[id] - log_z);
        }
    }
    std::ofstream logits_file;
    if (!logits_path.empty()) {
        logits_file.open(bovary::path_from_utf8(logits_path), std::ios::binary);
        if (!logits_file) throw std::runtime_error("cannot open logits output");
    }
    std::vector<int> generated;
    std::vector<double> token_times, token_elapsed;
    bool early_stop = false;
    bool in_text = direct_answer;
    bool answer_ended = false;
    double ttft = 0;
    const auto generation_start = clock_type::now();
    bool verifying_joint = joint_length != 0;
    int joint_accepted = 0, joint_restarts = 0;
    size_t checked = 0;
    // Positive indices address input tokens. The joint batch emits logits
    // only for its trailing verification rows, which we index from the end.
    int32_t logit_row = verifying_joint ? -(joint_length + 1) : -1;
    int proposals = 0, accepted = 0, rollbacks = 0;
    for (int i = 0; i < generation_limit; ++i) {
        float * logits = llama_get_logits_ith(ctx, logit_row);
        if (!logits) throw std::runtime_error("missing logits");
        if (logits_file.is_open()) logits_file.write((const char *) logits, vocab_size * sizeof(float));
        const int id = (int) (std::max_element(logits, logits + vocab_size) - logits);
        if (i == 0) {
            ttft = seconds(start);
            fprintf(stderr, "inkling_first_sample elapsed_s=%.9f token=%d\n", ttft, id);
            fflush(stderr);
        }
        if (llama_vocab_is_eog(vocab, id)) { early_stop = true; break; }
        generated.push_back(id);
        char text[1024];
        int len = llama_token_to_piece(vocab, id, text, sizeof(text), 0, true);
        if (len < 0) throw std::runtime_error("token piece too long");
        const std::string piece(text, len);
        if (piece == "<|end_message|>") {
            answer_ended = in_text && direct_answer;
            in_text = false;
        } else if (piece == "<|content_text|>") {
            in_text = true;
        } else if (piece == "<|content_thinking|>" || piece == "<|message_model|>") {
            in_text = false;
        } else if (in_text) {
            if (worker) frame('T', piece);
            else { fwrite(text, 1, len, stdout); fflush(stdout); }
        }
        token_elapsed.push_back(seconds(start));
        if (answer_ended) break;
        if (i + 1 < generation_limit) {
            const auto step = clock_type::now();
            if (checked < draft.size() && id == draft[checked]) {
                // Row 0 followed the seed. Row k follows the kth verified
                // draft token and already contains the next full logits.
                ++checked;
                logit_row = verifying_joint ? int32_t(checked) - joint_length - 1 : int32_t(checked);
                ++accepted;
                if (verifying_joint) ++joint_accepted;
            } else {
                if (checked < draft.size()) {
                    if (verifying_joint) {
                        // The joint prompt batch may exceed the recurrent
                        // rollback window. Restore from verified tokens only.
                        llama_memory_clear(llama_get_memory(ctx), true);
                        auto verified = tokens;
                        verified.insert(verified.end(), generated.begin(), generated.end() - 1);
                        prefill(verified, int(verified.size()) - 1);
                        ++joint_restarts;
                    } else if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt + i, -1)) {
                        throw std::runtime_error("could not roll back rejected prompt lookup");
                    }
                    ++rollbacks;
                }
                verifying_joint = false;
                draft = draft_from_prompt(tokens, generated, std::min(lookup_depth, generation_limit - i - 1));
                checked = 0;
                if (draft.empty()) {
                    decode(id, n_prompt + i, true);
                    logit_row = -1;
                } else {
                    llama_batch_ext_clear(batch);
                    for (int j = 0; j <= int(draft.size()); ++j) {
                        const int idx = llama_batch_ext_add_token(batch, 0, j ? draft[j - 1] : id);
                        const llama_pos pos = n_prompt + i + j;
                        if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos) ||
                                !llama_batch_ext_set_output_logits(batch, idx, true))
                            throw std::runtime_error("prompt lookup batch setup failed");
                    }
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0)
                        throw std::runtime_error("prompt lookup verification failed");
                    llama_synchronize(ctx);
                    logit_row = 0;
                    proposals += int(draft.size());
                }
            }
            token_times.push_back(seconds(step));
        }
    }
    const double generation_s = seconds(generation_start);
    const double total_s = seconds(start);
    std::ostringstream metrics;
    metrics << std::setprecision(17);
    metrics << "{\n\"clock_start_ns\":" << std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count()
        << ",\n\"model_load_s\":" << model_load << ",\n\"context_ready_s\":" << ready
        << ",\n\"prompt_tokens\":" << n_prompt << ",\n\"prompt_batch\":" << prompt_batch << ",\n\"prompt_s\":" << prompt_s
        << ",\n\"prompt_tokens_per_s\":" << n_prompt / prompt_s
        << ",\n\"ttft_process_internal_s\":" << (generated.empty() ? "null" : std::to_string(ttft)) << ",\n\"ttft_boundary\":" << json_string(worker ? "request start through first emitted generation token; excludes worker startup" : "before backend init through first emitted generation token")
        << ",\n\"generation_s\":" << generation_s << ",\n\"total_s\":" << total_s
        << ",\n\"lookup_proposals\":" << proposals << ",\n\"lookup_accepted\":" << accepted
        << ",\n\"lookup_rollbacks\":" << rollbacks
        << ",\n\"joint_length\":" << joint_length << ",\n\"joint_accepted\":" << joint_accepted
        << ",\n\"joint_restarts\":" << joint_restarts
        << ",\n\"actual_generated_tokens\":" << generated.size()
        << ",\n\"requested_generated_tokens\":" << predict << ",\n\"generation_limit\":" << generation_limit
        << ",\n\"finish_reason\":" << json_string(early_stop ? "eos" : answer_ended ? "end_message" : predict == 0 ? "context_limit" : "token_limit")
        << ",\n\"direct_answer\":" << (direct_answer ? "true" : "false")
        << ",\n\"early_stop\":" << (early_stop ? "true" : "false")
        << ",\n\"context\":" << context << ",\n\"allocated_context\":" << llama_n_ctx_seq(ctx)
        << ",\n\"vocab_size\":" << vocab_size << ",\n\"tokens\":[";
    for (size_t i = 0; i < generated.size(); ++i) metrics << (i ? "," : "") << generated[i];
    metrics << "],\n\"decode_step_s\":[";
    for (size_t i = 0; i < token_times.size(); ++i) metrics << (i ? "," : "") << token_times[i];
    metrics << "],\n\"token_elapsed_s\":[";
    for (size_t i = 0; i < token_elapsed.size(); ++i) metrics << (i ? "," : "") << token_elapsed[i];
    metrics << "],\n\"choices\":\"" << choices << "\",\n\"choice_ids\":[";
    for (size_t i = 0; i < choice_ids.size(); ++i) metrics << (i ? "," : "") << choice_ids[i];
    metrics << "],\n\"choice_logits\":[";
    for (size_t i = 0; i < choice_logits.size(); ++i) metrics << (i ? "," : "") << choice_logits[i];
    metrics << "],\n\"choice_logprobs\":[";
    for (size_t i = 0; i < choice_logprobs.size(); ++i) metrics << (i ? "," : "") << choice_logprobs[i];
    metrics << "]\n}\n";
    llama_perf_context_print(ctx);
    return metrics.str();
}

static int run(const std::vector<std::string> & arguments) {
    std::string model_path, prompt_file, logits_path, metrics_path = "metrics.json", load_mode = "mmap", choices;
    int predict = 32, context = 4096, layers = 99, cache_mib = 0, threads = 8, prompt_batch = 1;
#ifdef BOVARY_CPU_ONLY
    layers = 0;
#endif
    bool cpu_experts = false, gpu_embedding = false, worker = false, direct_answer = false, no_op_offload = false;
    for (size_t i = 1; i < arguments.size(); ++i) {
        std::string key = arguments[i];
        if (key == "--require-cpu-only") {
#ifndef BOVARY_CPU_ONLY
            throw std::runtime_error("CPU-only request requires the CPU-only executable; CUDA backend initialization refused");
#endif
            continue;
        }
        if (key == "--protocol-version") { printf("bovary-worker 2\n"); return 0; }
        if (key == "--worker") { worker = true; continue; }
        if (key == "--direct-answer") { direct_answer = true; continue; }
        if (key == "--no-op-offload") { no_op_offload = true; continue; }
        if (key == "--cpu-experts") { cpu_experts = true; continue; }
        if (key == "--gpu-embedding") { gpu_embedding = true; continue; }
        if (i + 1 == arguments.size()) throw std::runtime_error("missing option value");
        std::string value = arguments[++i];
        if      (key == "--model") model_path = value;
        else if (key == "--prompt-file") prompt_file = value;
        else if (key == "--metrics") metrics_path = value;
        else if (key == "--logits") logits_path = value;
        else if (key == "--predict") predict = std::stoi(value);
        else if (key == "--context") context = std::stoi(value);
        else if (key == "--gpu-layers") layers = std::stoi(value);
        else if (key == "--cache-mib") cache_mib = std::stoi(value);
        else if (key == "--threads") threads = std::stoi(value);
        else if (key == "--prompt-batch") prompt_batch = std::stoi(value);
        else if (key == "--load-mode") load_mode = value;
        else if (key == "--choices") choices = value;
        else throw std::runtime_error("unknown option: " + key);
    }
    if (model_path.empty() || (!worker && prompt_file.empty()) || predict < 0 || context < 64 ||
        context > 262144 || cache_mib < 0 || threads < 1) throw std::runtime_error("invalid arguments");
    if (prompt_batch < 1 || prompt_batch > 256)
        throw std::runtime_error("prompt batch must be from 1 to 256");
    const char * stream = std::getenv("BOVARY_STREAM");
#ifdef BOVARY_CPU_ONLY
    if (layers != 0 || cache_mib != 0 || gpu_embedding || cpu_experts ||
        (stream && std::string(stream) == "1"))
        throw std::runtime_error("CPU-only build requires zero GPU layers/cache and no streaming or GPU overrides");
#endif
    if (stream && std::string(stream) == "1" && (cache_mib == 0 || !cpu_experts || layers < 30))
        throw std::runtime_error("streaming requires CPU expert placement, positive cache and all common layers offloaded");
    if (worker && (!logits_path.empty() || !choices.empty()))
        throw std::runtime_error("worker mode forbids persistent logit dumps and choice diagnostics");
    std::string user;
    constexpr size_t max_prompt_bytes = 1024 * 1024;
    if (!worker) {
        std::ifstream input(bovary::path_from_utf8(prompt_file), std::ios::binary);
        if (!input) throw std::runtime_error("cannot open prompt");
        user.resize(max_prompt_bytes + 1);
        input.read(user.data(), static_cast<std::streamsize>(user.size()));
        if (input.bad() || (input.fail() && !input.eof()))
            throw std::runtime_error("cannot read prompt");
        user.resize(static_cast<size_t>(input.gcount()));
        if (user.empty() || user.size() > max_prompt_bytes)
            throw std::runtime_error("prompt file must contain 1 byte to 1 MiB");
    }
#ifdef _WIN32
    if (worker) {
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
    }
#endif
    const auto start = clock_type::now();
#ifndef BOVARY_CPU_ONLY
    ggml_backend_load_all();
#endif
    llama_backend_init();
#ifdef BOVARY_CPU_ONLY
    if (ggml_backend_reg_count() != 1 ||
        std::string(ggml_backend_reg_name(ggml_backend_reg_get(0))) != "CPU")
        throw std::runtime_error("CPU-only build has an unexpected backend registry");
    fprintf(stderr, "bovary_cpu: cpu_only=1 backend=CPU dynamic_backend_loading=0 gpu_layers=0 expert_cache=0\n");
#endif
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = layers;
    if      (load_mode == "mmap") mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    else if (load_mode == "none") mp.load_mode = LLAMA_LOAD_MODE_NONE;
    else if (load_mode == "direct") mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    else throw std::runtime_error("load mode must be mmap, none or direct");
    mp.use_extra_bufts = false;
    llama_model_tensor_buft_override overrides[3] = {};
    size_t override_count = 0;
    if (cpu_experts) {
        auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) throw std::runtime_error("CPU backend absent");
        overrides[override_count++] = {"ffn_(gate|up|down|gate_up)_exps", ggml_backend_dev_buffer_type(dev)};
    }
    if (gpu_embedding) {
        auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) throw std::runtime_error("GPU backend absent for embedding placement");
        overrides[override_count++] = {"^token_embd\\.weight$", ggml_backend_dev_buffer_type(dev)};
    }
    if (override_count) mp.tensor_buft_overrides = overrides;
    auto * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) throw std::runtime_error("model load failed");
    const double model_load = seconds(start);
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_guard(model, llama_model_free);
    char arch[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) <= 0 ||
            std::string(arch) != "inkling2_loop")
        throw std::runtime_error("Inkling experimental worker requires inkling2_loop");
    if (worker) {
        frame('R', "{\"protocol\":2,\"model_load_s\":" + std::to_string(model_load) + "}");
        for (;;) {
            std::string header;
            char c;
            while (std::cin.get(c) && c != '\n') {
                if (header.size() >= 100) throw std::runtime_error("worker header exceeds limit");
                header.push_back(c);
            }
            if (!std::cin && header.empty()) break;
            if (header == "QUIT") break;
            std::istringstream in(header);
            std::string command, extra;
            size_t bytes = 0;
            int count = 0;
            if (!(in >> command >> count >> bytes) || command != "RUN" || (in >> extra) ||
                    bytes == 0 || bytes > max_prompt_bytes || count < 0 || count >= context)
                throw std::runtime_error("invalid worker request header");
            user.assign(bytes, '\0');
            if (!std::cin.read(user.data(), bytes)) throw std::runtime_error("truncated worker input");
            const auto request_start = clock_type::now();
            try {
                auto metrics = run_request(model, user, count, context, threads, prompt_batch,
                        cache_mib, "", "", true, direct_answer, no_op_offload, request_start, 0);
                frame('D', metrics);
            } catch (const std::exception & e) {
                const std::string message = e.what();
                const char * code = message == "prompt/generation exceeds context" ? "context_overflow" : "worker_request";
                frame('E', "{\"code\":" + json_string(code) + ",\"message\":" + json_string(message) + "}");
            }
        }
    } else {
        const auto metrics = run_request(model, user, predict, context, threads, prompt_batch,
                cache_mib, choices, logits_path, false, direct_answer, no_op_offload, start, model_load);
        std::ofstream output(bovary::path_from_utf8(metrics_path));
        if (!output || !(output << metrics)) throw std::runtime_error("cannot write metrics output");
    }
    model_guard.reset();
    llama_backend_free();
    return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t ** argv) try {
    return run(bovary::utf8_arguments(argc, argv));
#else
int main(int argc, char ** argv) try {
    return run(std::vector<std::string>(argv, argv + argc));
#endif
} catch (const std::exception & e) {
    fprintf(stderr, "bovary-infer: %s\n", e.what());
    return 1;
}
