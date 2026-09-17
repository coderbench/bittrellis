// Fixed-partition teacher-forced scoring through llama.cpp, with the same inputs and output format
// as tools/sparkinfer_refscore.cpp, so the GGUF reference point is measured by the same code path.
//
// Usage: llamacpp_score <model.gguf> <tokens.bin> <refids.bin> <out.bin> <prefix_len>
//   tokens.bin  int32 [N]
//   refids.bin  int32 K, then int32 [M*K] BF16 reference ids for positions prefix_len..N-2
//   out.bin     "BTRS" | int32 M | int32 K | int32 argmax[M] | float32 lp_target[M] | float32 lp_ref[M*K]
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <tokens.bin> <refids.bin> <out.bin> <prefix_len>\n", argv[0]);
        return 2;
    }
    const int prefix = atoi(argv[5]);
    std::vector<int> toks, ref;
    if (!read_i32(argv[2], toks) || !read_i32(argv[3], ref) || ref.empty()) { printf("[FAIL] inputs\n"); return 1; }
    const int K = ref[0];
    const int N = (int) toks.size();
    const int M = N - 1 - prefix;
    if (K <= 0 || M <= 0 || (long) ref.size() != 1 + (long) M * K) { printf("[FAIL] refids size\n"); return 1; }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("[FAIL] load\n"); return 1; }
    auto cp = llama_context_default_params();
    const int n_batch = 1024;
    cp.n_ctx = (uint32_t) (N + 64);
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { printf("[FAIL] context\n"); return 1; }
    const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<int> am(M);
    std::vector<float> lpt(M), lpr((size_t) M * K);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int start = 0; start < N - 1; start += n_batch) {
        const int end = std::min(N - 1, start + n_batch);
        batch.n_tokens = 0;
        for (int i = start; i < end; i++) {
            const int b = batch.n_tokens++;
            batch.token[b] = toks[i];
            batch.pos[b] = i;
            batch.n_seq_id[b] = 1;
            batch.seq_id[b][0] = 0;
            batch.logits[b] = i >= prefix;
        }
        if (llama_decode(ctx, batch) != 0) { printf("[FAIL] decode at %d\n", start); return 1; }
        for (int i = std::max(start, prefix); i < end; i++) {
            const int m = i - prefix;
            const float * lg = llama_get_logits_ith(ctx, i - start);
            double mx = lg[0];
            int arg = 0;
            for (int v = 1; v < V; v++) if (lg[v] > mx) { mx = lg[v]; arg = v; }
            double se = 0.0;
            for (int v = 0; v < V; v++) se += std::exp((double) lg[v] - mx);
            const double lse = mx + std::log(se);
            am[m] = arg;
            lpt[m] = (float) ((double) lg[toks[i + 1]] - lse);
            const int * ids = ref.data() + 1 + (size_t) m * K;
            for (int k = 0; k < K; k++) lpr[(size_t) m * K + k] = (float) ((double) lg[ids[k]] - lse);
        }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    FILE * f = fopen(argv[4], "wb");
    if (!f) { printf("[FAIL] write\n"); return 1; }
    fwrite("BTRS", 1, 4, f);
    fwrite(&M, 4, 1, f);
    fwrite(&K, 4, 1, f);
    fwrite(am.data(), 4, (size_t) M, f);
    fwrite(lpt.data(), 4, (size_t) M, f);
    fwrite(lpr.data(), 4, (size_t) M * K, f);
    fclose(f);
    printf("OK %d positions\n", M);
    return 0;
}
