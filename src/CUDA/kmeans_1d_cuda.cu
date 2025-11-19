// kmeans_1d_cuda.cu
//
// K-means 1D (CUDA), versão otimizada:
//
// - Mantém X, C e assign na GPU durante todo o loop.
// - Funde assignment + acumulação + SSE em um único kernel (assign_accumulate).
// - SSE por iteração acumulado em um escalar na GPU (atomicAdd) e copiado
//   como double para o host a cada iteração (8 bytes).
// - Update dos centróides também na GPU, com a mesma política de cluster vazio
//   do código sequencial: centróide vazio recebe X[0].
// - Interface de linha de comando compatível com o naive:
//
//   ./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-6]
//                     [assign.csv] [centroids.csv] [sse.csv]
//
//   nvcc -O2 -arch=sm_75 -DBLOCK_SIZE=256 kmeans_1d_cuda.cu -o kmeans_1d_cuda
//

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>

// -------------------- util de checagem CUDA --------------------
#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if(_e != cudaSuccess){                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// -------------------- parâmetros de execução --------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef MAX_CONST_K
#define MAX_CONST_K 4096
#endif

__constant__ double dC_const[MAX_CONST_K];

// -------------------- kernels --------------------

// Kernel único: assignment + acumulação de somas/contagens + SSE.
// 1 thread por ponto i.
__global__
void assign_accumulate(const double* __restrict__ X,
                       const double* __restrict__ C,
                       int* __restrict__ assign,
                       double* __restrict__ sum,
                       int* __restrict__ cnt,
                       double* __restrict__ sse,
                       int N, int K, int use_constC)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    double xi = X[i];
    int best = -1;
    double bestd = 1e300;

    // Varre todos os K centróides
    for(int c = 0; c < K; ++c){
        double cc = use_constC ? dC_const[c] : C[c];
        double diff = xi - cc;
        double d = diff * diff;
        if(d < bestd){ bestd = d; best = c; }
    }

    assign[i] = best;

    // Acumula soma e contagem para o cluster escolhido
    atomicAdd(&sum[best], xi);
    atomicAdd(&cnt[best], 1);

    // Acumula SSE global
    atomicAdd(sse, bestd);
}

// Atualiza centróides: média se cnt>0; caso contrário, C[c] = X[0].
__global__
void update_centroids(double* __restrict__ C,
                      const double* __restrict__ sum,
                      const int* __restrict__ cnt,
                      const double* __restrict__ X,
                      int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= K) return;

    int n = cnt[c];
    if(n > 0){
        C[c] = sum[c] / (double)n;
    } else {
        C[c] = X[0]; // mesma política do baseline sequencial
    }
}

// -------------------- util CSV 1D: 1 número por linha --------------------
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }
    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){
            fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path);
            free(A); fclose(f); exit(1);
        }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

static void write_sse_series_csv(const char *path, const double *sse_hist, int I){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<I;i++) fprintf(f, "%.10f\n", sse_hist[i]);
    fclose(f);
}

// ---------- CSV de métricas (CUDA) ----------
// Formato: Tempo(ms),Tamanho,SSE_Final,Iteracoes,BlockSize
static void append_results_cuda_csv(const char *path,
                                    double tempo_ms,
                                    int tamanho,
                                    double sse_final,
                                    int iters,
                                    int block_size)
{
    int need_header = 0;
    struct stat st;
    if(stat(path, &st) != 0 || st.st_size == 0){
        need_header = 1;
    }

    FILE *f = fopen(path, "a");
    if(!f){
        fprintf(stderr, "Erro ao abrir %s para append\n", path);
        return;
    }

    if(need_header){
        fprintf(f, "Tempo(ms),Tamanho,SSE_Final,Iteracoes,BlockSize\n");
    }

    fprintf(f, "%.6f,%d,%.10f,%d,%d\n",
            tempo_ms, tamanho, sse_final, iters, block_size);

    fclose(f);
}

// -------------------- loop K-means (host) --------------------
static void kmeans_1d_cuda(const double *hX, double *hC, int *hAssign,
                           int N, int K, int max_iter, double eps,
                           int *iters_out, double *sse_out,
                           double *sse_history)
{
    // Aloca buffers na GPU
    double *dX = nullptr, *dC = nullptr;
    double *dSum = nullptr, *dSSE = nullptr;
    int *dAssign = nullptr, *dCnt = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&dX,    (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dC,    (size_t)K * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dAssign, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dSum,  (size_t)K * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dCnt,  (size_t)K * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dSSE,  sizeof(double)));

    // Copia dados iniciais para a GPU (apenas uma vez)
    CUDA_CHECK(cudaMemcpy(dX, hX, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));

    // Configuração de grid
    dim3 block(BLOCK_SIZE);
    dim3 gridPts((N + block.x - 1) / block.x);
    dim3 gridK((K + block.x - 1) / block.x);

    int use_constC = (K <= MAX_CONST_K) ? 1 : 0;
    if(use_constC){
        CUDA_CHECK(cudaMemcpyToSymbol(dC_const, hC,
                                      (size_t)K * sizeof(double),
                                      0, cudaMemcpyHostToDevice));
    }

    double prev_sse = 1e300;
    double sse = 0.0;
    int it = 0;

    for(it = 0; it < max_iter; ++it){
        // Zera acumuladores na GPU (sum, cnt, sse)
        CUDA_CHECK(cudaMemset(dSum, 0, (size_t)K * sizeof(double)));
        CUDA_CHECK(cudaMemset(dCnt, 0, (size_t)K * sizeof(int)));
        CUDA_CHECK(cudaMemset(dSSE, 0, sizeof(double)));

        // Assignment + acumulação + SSE em um único kernel
        assign_accumulate<<<gridPts, block>>>(dX, dC, dAssign,
                                              dSum, dCnt, dSSE,
                                              N, K, use_constC);
        CUDA_CHECK(cudaGetLastError());

        // Copia SSE escalar para o host
        CUDA_CHECK(cudaMemcpy(&sse, dSSE, sizeof(double),
                              cudaMemcpyDeviceToHost));

        if(sse_history) sse_history[it] = sse;

        // Critério de parada por variação relativa do SSE
        double denom = (prev_sse > 0.0) ? prev_sse : 1.0;
        double rel = fabs(sse - prev_sse) / denom;
        if(rel < eps){
            ++it; // conta a iteração atual
            break;
        }

        // Atualiza centróides na GPU
        update_centroids<<<gridK, block>>>(dC, dSum, dCnt, dX, K);
        CUDA_CHECK(cudaGetLastError());

        // Atualiza memória constante com novos centróides, se estiver em uso
        if(use_constC){
            CUDA_CHECK(cudaMemcpyToSymbol(dC_const, dC,
                                          (size_t)K * sizeof(double),
                                          0, cudaMemcpyDeviceToDevice));
        }

        prev_sse = sse;
    }

    *iters_out = it;
    *sse_out = sse;

    // Copia resultados finais para o host
    CUDA_CHECK(cudaMemcpy(hAssign, dAssign,
                          (size_t)N * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC, dC,
                          (size_t)K * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Libera recursos da GPU
    cudaFree(dX);
    cudaFree(dC);
    cudaFree(dAssign);
    cudaFree(dSum);
    cudaFree(dCnt);
    cudaFree(dSSE);
}

// -------------------- main --------------------
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-6] [assign.csv] [centroids.csv] [sse.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-6;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;
    const char *outSSE      = (argc>7)? argv[7] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    double *sse_history = (double*)malloc((size_t)max_iter * sizeof(double));
    if(!assign || !sse_history){
        fprintf(stderr,"Sem memoria para assign/sse_history\n");
        free(X); free(C);
        if(assign) free(assign);
        if(sse_history) free(sse_history);
        return 1;
    }

    clock_t t0 = clock();
    int iters = 0; double sse = 0.0;
    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, &iters, &sse, sse_history);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    // Saídas compatíveis com as versões Serial/OpenMP
    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);
    write_sse_series_csv(outSSE, sse_history, iters);

    // Append dos resultados no CSV de métricas da GPU
    append_results_cuda_csv("Resultados/cuda.csv",
                            ms,        // Tempo(ms)
                            N,         // Tamanho
                            sse,       // SSE_Final
                            iters,     // Iteracoes
                            BLOCK_SIZE // BlockSize
                            );

    free(assign); free(X); free(C); free(sse_history);
    return 0;
}

