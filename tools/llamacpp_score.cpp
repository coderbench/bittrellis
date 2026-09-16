// Teacher-forced top-k scoring through llama.cpp, in the exact output format of SparkInfer's
// qwen3_gguf_score, so a GGUF reference point is measured by the same KL code as candidates.
//
// Usage: llamacpp_score <model.gguf> <topk> <prefix_len> <ids_file>
//   ids_file: whitespace-separated token ids. Positions i >= prefix_len (i + 1 < N) are scored.
// Output:  S i=<i> tgt=<t> am=<argmax> lp=<logprob_tgt> top=id:lp,...
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <topk> <prefix_len> <ids_file>\n", argv[0]);
        return 2;
    }
    const int topk = atoi(argv[2]);
    const int prefix = atoi(argv[3]);
    std::vector<llama_token> toks;
    {
        std::ifstream f(argv[4]);
        long v;
        while (f >> v) toks.push_back((llama_token) v);
    }
    const int N = (int) toks.size();
    if (N < 2) { fprintf(stderr, "need >= 2 tokens\n"); return 1; }

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
    const int K = std::min(topk, V);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<int> idx(V);
    double nll = 0.0;
    int scored = 0, match = 0;
    for (int start = 0; start < N - 1; start += n_batch) {
        const int end = std::min(N - 1, start + n_batch);  // last token never needs a forward
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
            const float * lg = llama_get_logits_ith(ctx, i - start);
            float mx = lg[0];
            for (int v = 1; v < V; v++) mx = std::max(mx, lg[v]);
            double se = 0.0;
            for (int v = 0; v < V; v++) se += std::exp((double) lg[v] - mx);
            const double lse = mx + std::log(se);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            const int tgt = toks[i + 1];
            const double lp = (double) lg[tgt] - lse;
            nll -= lp;
            scored++;
            match += idx[0] == tgt;
            printf("S i=%d tgt=%d am=%d lp=%.6f top=", i, tgt, idx[0], lp);
            for (int k = 0; k < K; k++) printf("%s%d:%.6f", k ? "," : "", idx[k], (double) lg[idx[k]] - lse);
            printf("\n");
        }
    }
    printf("PPL %.5f over %d positions\n", std::exp(nll / std::max(1, scored)), scored);
    printf("ARGMATCH %d/%d %.4f\n", match, scored, (double) match / std::max(1, scored));
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
