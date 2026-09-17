// Fixed-partition teacher-forced scoring through the pinned SparkInfer runtime.
//
// SparkInfer's own qwen3_gguf_score prints only the candidate's top-k, which cannot score a
// partition chosen *before* the candidate existed. This tool links the unmodified runtime library
// and drives the exact same load + cache_prefix + forward_token path, but reports, for every
// scored position, the candidate's log-probability of each BF16 reference token id.
//
// Usage: bittrellis_refscore <checkpoint> <tokens.bin> <refids.bin> <out.bin> <prefix_len>
//        bittrellis_refscore <checkpoint> --jobs <jobs.txt>
//   jobs.txt: one stream per line, "<tokens.bin> <refids.bin> <out.bin> <prefix_len>". The model is
//   loaded once; each stream starts from position 0 (GDN state is reset there) or from its own
//   cache_prefix, with the KV sequence re-allocated per stream.
//   tokens.bin  int32 [N]                token stream
//   refids.bin  int32 K, then int32 [M*K] reference token ids for positions prefix_len..prefix_len+M-1
//   out.bin     "BTRS" | int32 M | int32 K | int32 argmax[M] | float32 lp_target[M] | float32 lp_ref[M*K]
// Stdout: "OK <M> positions" or "[FAIL] ...".
#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"
#include "qwen_checkpoint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool read_i32(const char* path, std::vector<int>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)n);
    const bool ok = fread(out.data(), 4, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

struct Job { std::string tok, ref, out; int prefix; };

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("usage: %s <checkpoint> <tokens.bin> <refids.bin> <out.bin> <prefix_len> | <checkpoint> --jobs <jobs.txt>\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    std::vector<Job> jobs;
    if (std::string(argv[2]) == "--jobs") {
        FILE* jf = fopen(argv[3], "r");
        if (!jf) { printf("[FAIL] cannot read %s\n", argv[3]); return 1; }
        char a[4096], b[4096], c[4096];
        int pl;
        while (fscanf(jf, "%4095s %4095s %4095s %d", a, b, c, &pl) == 4) jobs.push_back({a, b, c, pl});
        fclose(jf);
    } else if (argc >= 6) {
        jobs.push_back({argv[2], argv[3], argv[4], atoi(argv[5])});
    } else {
        printf("[FAIL] bad arguments\n");
        return 2;
    }
    if (jobs.empty()) { printf("[FAIL] no jobs\n"); return 1; }
    std::vector<std::vector<int>> all_toks(jobs.size()), all_ref(jobs.size());
    int max_n = 0;
    for (size_t j = 0; j < jobs.size(); j++) {
        if (!read_i32(jobs[j].tok.c_str(), all_toks[j]) || !read_i32(jobs[j].ref.c_str(), all_ref[j]) || all_ref[j].empty()) {
            printf("[FAIL] cannot read inputs of job %zu\n", j);
            return 1;
        }
        max_n = std::max(max_n, (int)all_toks[j].size());
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[FAIL] no GPU\n"); return 1; }

    sparkinfer::GGUF g;
    sparkinfer::Qwen35Config cfg;
    QwenCheckpointKind ckind = QwenCheckpointKind::Gguf;
    std::string err;
    if (!qwen_checkpoint_open(path, cfg, g, ckind, err)) { printf("[FAIL] %s\n", err.c_str()); return 1; }
    cfg.max_seq = std::max(2048, max_n + 16);

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads; kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    // Same KV policy as qwen3_gguf_score: SPARKINFER_KV_INT8 forces it, else int8 from 4096 tokens.
    // Every HPC-01 stream is >= 4096 tokens, so one process uses the same policy for all of them.
    { const char* e = getenv("SPARKINFER_KV_INT8");
      kvc.int8_kv = e ? (e[0] != '0') : (cfg.hybrid ? (max_n >= 4096) : true); }
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!qwen_checkpoint_load(model, path, ckind)) { printf("[FAIL] load\n"); return 1; }

    const int V = cfg.vocab;
    std::vector<float> lg(V);
    long nonfinite = 0;
    for (size_t j = 0; j < jobs.size(); j++) {
        const std::vector<int>& toks = all_toks[j];
        const std::vector<int>& ref = all_ref[j];
        const int prefix = jobs[j].prefix;
        const int K = ref[0];
        const int N = (int)toks.size();
        const int M = N - 1 - prefix;
        if (K <= 0 || M <= 0 || (long)ref.size() != 1 + (long)M * K) {
            printf("[FAIL] job %zu: refids hold %zu values, want 1 + %d*%d\n", j, ref.size(), M, K);
            return 1;
        }
        if (!kv.allocate(0, cfg.max_seq)) { printf("[FAIL] KV allocate\n"); return 1; }
        if (prefix > 0) {
            std::vector<int> pre(toks.begin(), toks.begin() + prefix);
            if (!model.cache_prefix(pre)) { printf("[FAIL] cache_prefix(%d)\n", prefix); return 1; }
        }
        std::vector<int> am(M);
        std::vector<float> lpt(M), lpr((size_t)M * K);
        for (int m = 0; m < M; m++) {
            const int i = prefix + m;
            am[m] = model.forward_token(toks[i], i);
            model.copy_logits(lg.data());
            double mx = -INFINITY;
            for (int v = 0; v < V; v++) {
                if (!std::isfinite(lg[v])) { nonfinite++; continue; }
                mx = std::max(mx, (double)lg[v]);
            }
            double se = 0.0;
            for (int v = 0; v < V; v++) if (std::isfinite(lg[v])) se += std::exp((double)lg[v] - mx);
            const double lse = mx + std::log(se);
            lpt[m] = (float)((double)lg[toks[i + 1]] - lse);
            const int* ids = ref.data() + 1 + (size_t)m * K;
            for (int k = 0; k < K; k++) lpr[(size_t)m * K + k] = (float)((double)lg[ids[k]] - lse);
        }
        kv.free(0);
        FILE* f = fopen(jobs[j].out.c_str(), "wb");
        if (!f) { printf("[FAIL] cannot write %s\n", jobs[j].out.c_str()); return 1; }
        fwrite("BTRS", 1, 4, f);
        fwrite(&M, 4, 1, f);
        fwrite(&K, 4, 1, f);
        fwrite(am.data(), 4, (size_t)M, f);
        fwrite(lpt.data(), 4, (size_t)M, f);
        fwrite(lpr.data(), 4, (size_t)M * K, f);
        fclose(f);
        printf("JOB %zu %d positions\n", j, M);
        fflush(stdout);
    }
    printf("OK %zu jobs nonfinite_logits=%ld\n", jobs.size(), nonfinite);
    return 0;
}
