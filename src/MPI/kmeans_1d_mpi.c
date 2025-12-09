/* kmeans_1d_mpi.c (baseado em kmeans_1d_naive.c)
   K-means 1D (C99), versão com MPI:
   - Mesma interface de linha de comando da versão sequencial.
   - Se rodar com 1 processo (size == 1), usa a versão sequencial original.
   - Se rodar com P > 1, distribui os pontos entre os processos e faz
     assignment/update em paralelo com MPI.

   Compilar (exemplo):
      mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm

   Uso:
      mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv \
                                  [max_iter=50] [eps=1e-4] \
                                  [assign.csv] [centroids.csv] [sse.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <mpi.h>   /* [MPI] Inclusão da biblioteca MPI */

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
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

static void write_sse_series_csv(const char *path, const double *sse_history, int num_iters){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    fprintf(f, "SSE\n");
    for(int i=0; i<num_iters; i++){
        fprintf(f, "%.6f\n", sse_history[i]);
    }
    fclose(f);
}

/* ---------- Função para verificar se arquivo existe ---------- */
static int file_exists(const char *path){
    struct stat buffer;
    return (stat(path, &buffer) == 0);
}

/* ---------- Função para append dos resultados ---------- */
static void append_results_csv(const char *path, double tempo_ms, int tamanho, double sse, int iteracoes){
    int write_header = !file_exists(path);
    
    FILE *f = fopen(path, "a");
    if(!f){ 
        fprintf(stderr,"Erro ao abrir %s para escrita\n", path); 
        return; 
    }
    
    if(write_header){
        fprintf(f, "Tempo(ms),Tamanho,SSE_Final,Iteracoes\n");
    }
    fprintf(f, "%.3f,%d,%.6f,%d\n", tempo_ms, tamanho, sse, iteracoes);
    fclose(f);
}

/* ---------- k-means 1D (versão sequencial original) ---------- */
static double assignment_step_1d(const double *X, const double *C,
                                 int *assign, int N, int K){
    double sse = 0.0;
    for(int i=0;i<N;i++){
        int best = -1;
        double bestd = 1e300;
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

static void update_step_1d(const double *X, double *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt   = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    for(int i=0;i<N;i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0];
    }
    free(sum); free(cnt);
}

static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out,
                      double *sse_history)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d(X, C, assign, N, K);
        if(sse_history) sse_history[it] = sse;
        
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){
            it++;
            break;
        }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out   = sse;
}

/* ---------- k-means 1D versão MPI ---------- */
/* [MPI] Esta função replica a lógica de kmeans_1d, mas:
   - distribui os pontos entre os processos por intervalo de índices;
   - calcula assignment e somatórios localmente;
   - usa MPI_Reduce e MPI_Allreduce para obter SSE_global e somas globais;
   - mantém o mesmo critério de parada da versão sequencial.
*/
static void kmeans_1d_mpi(const double *X, double *C, int *assign,
                          int N, int K, int max_iter, double eps,
                          int *iters_out, double *sse_out,
                          double *sse_history, /* usado apenas em rank 0 */
                          int rank, int size)
{
    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it;

    /* [MPI] Cálculo do intervalo de pontos de cada processo */
    int base = N / size;
    int rem  = N % size;

    int start, end;
    if(rank < rem){
        start = rank * (base + 1);
        end   = start + (base + 1);
    } else {
        start = rank * base + rem;
        end   = start + base;
    }
    int local_N = end - start;
    (void)local_N; /* só para evitar warning se não usamos explicitamente */

    /* [MPI] Vetores locais e globais para somas e contagens */
    double *sum_local  = (double*)calloc((size_t)K, sizeof(double));
    double *sum_global = (double*)calloc((size_t)K, sizeof(double));
    int    *cnt_local  = (int*)calloc((size_t)K, sizeof(int));
    int    *cnt_global = (int*)calloc((size_t)K, sizeof(int));
    if(!sum_local || !sum_global || !cnt_local || !cnt_global){
        fprintf(stderr,"[%d] Sem memoria nos vetores MPI\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(it=0; it<max_iter; it++){
        /* Zera acumuladores locais a cada iteração */
        for(int c=0; c<K; c++){
            sum_local[c] = 0.0;
            cnt_local[c] = 0;
        }

        double sse_local = 0.0;

        /* [MPI] 1) Assignment local + somatórios locais */
        for(int i = start; i < end; i++){
            int best = -1;
            double bestd = 1e300;
            double xi = X[i];

            for(int c=0; c<K; c++){
                double diff = xi - C[c];
                double d = diff*diff;
                if(d < bestd){
                    bestd = d;
                    best  = c;
                }
            }
            assign[i] = best;
            sse_local += bestd;
            sum_local[best] += xi;
            cnt_local[best] += 1;
        }

        /* [MPI] 2) Redução de SSE para o processo 0 */
        MPI_Reduce(&sse_local, &sse_global, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        /* [MPI] 3) Somatórios globais para todos (Allreduce) */
        MPI_Allreduce(sum_local,  sum_global,  K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cnt_local,  cnt_global,  K, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

        int stop = 0;
        if(rank == 0){
            if(sse_history) sse_history[it] = sse_global;

            double rel = fabs(sse_global - prev_sse) /
                         (prev_sse > 0.0 ? prev_sse : 1.0);

            if(rel < eps){
                stop = 1;
            }
        }

        /* [MPI] 4) Broadcast do flag de parada */
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(stop){
            /* Mesmo comportamento da versão sequencial:
               - não atualiza centróides na última iteração que dispara o critério;
               - it conta esta iteração. */
            if(rank == 0){
                *iters_out = it + 1;
                *sse_out   = sse_global;
            }
            break;
        }

        /* [MPI] 5) Atualização dos centróides com somas globais */
        if(rank == 0){
            for(int c=0; c<K; c++){
                if(cnt_global[c] > 0)
                    C[c] = sum_global[c] / (double)cnt_global[c];
                else
                    C[c] = X[0];  /* mesma estratégia naive do sequencial */
            }
            prev_sse = sse_global;
        }

        /* [MPI] 6) Broadcast dos centróides atualizados e prev_sse */
        MPI_Bcast(C,       K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&prev_sse, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* Se saiu porque atingiu max_iter sem convergência, garante saída coerente */
    if(it == max_iter){
        if(rank == 0){
            /* sse_global guarda o SSE da última iteração realizada */
            *iters_out = max_iter;
            *sse_out   = sse_global;
        }
    }

    /* [MPI] Broadcast dos resultados finais para todos (caso alguém queira usar) */
    MPI_Bcast(iters_out, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(sse_out,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(sum_local);
    free(sum_global);
    free(cnt_local);
    free(cnt_global);
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    int rank = 0, size = 1;

    /* [MPI] Inicialização do MPI e descoberta de rank/size */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 3){
        if(rank == 0){
            printf("Uso: %s dados.csv centroides_iniciais.csv "
                   "[max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [sse.csv]\n",
                   argv[0]);
            printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabecalho.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int    max_iter   = (argc>3)? atoi(argv[3]) : 50;
    double eps        = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;
    const char *outSSE      = (argc>7)? argv[7] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        if(rank == 0){
            fprintf(stderr,"Parametros invalidos: max_iter>0 e eps>0\n");
        }
        MPI_Finalize();
        return 1;
    }

    int N = 0, K = 0;
    double *X = NULL;
    double *C = NULL;

    /* [MPI] Apenas rank 0 le os arquivos; depois faremos broadcast */
    if(rank == 0){
        X = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
    }

    /* [MPI] Broadcast de N e K para todos */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* [MPI] Alocacao dos vetores X e C nos demais ranks */
    if(rank != 0){
        X = (double*)malloc((size_t)N * sizeof(double));
        C = (double*)malloc((size_t)K * sizeof(double));
    }
    if(!X || !C){
        fprintf(stderr,"[%d] Sem memoria para X ou C\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* [MPI] Broadcast dos dados X e C a partir do rank 0 */
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){
        fprintf(stderr,"[%d] Sem memoria para assign\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* [MPI] sse_history somente no rank 0 (quem grava o CSV) */
    double *sse_history = NULL;
    if(rank == 0){
        sse_history = (double*)malloc((size_t)max_iter * sizeof(double));
        if(!sse_history){
            fprintf(stderr,"[0] Sem memoria para sse_history\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* [MPI] Medida de tempo com MPI_Wtime (tempo de parede) */
    double t0 = MPI_Wtime();

    int iters = 0;
    double sse = 0.0;

    if(size == 1){
        /* [MPI] Caso trivial: 1 processo -> usa versao sequencial original. */
        kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse, sse_history);
    } else {
        /* [MPI] Versao paralela: distribui pontos entre os processos. */
        kmeans_1d_mpi(X, C, assign, N, K, max_iter, eps,
                      &iters, &sse, sse_history, rank, size);
    }

    double t1 = MPI_Wtime();
    double ms = 1000.0 * (t1 - t0);

    if(rank == 0){
        printf("K-means 1D (MPI)\n");
        printf("N=%d K=%d max_iter=%d eps=%g P=%d\n", N, K, max_iter, eps, size);
        printf("Iteracoes: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

        /* [MPI] Escrita dos arquivos de saida apenas no rank 0 */
        write_assign_csv(outAssign, assign, N);
        write_centroids_csv(outCentroid, C, K);
        write_sse_series_csv(outSSE, sse_history, iters);

        /* [MPI] Guarda resultados em arquivo separado da versao sequencial */
        append_results_csv("Resultados/mpi.csv", ms, N, sse, iters);
    }

    free(assign);
    free(X);
    free(C);
    if(rank == 0 && sse_history) free(sse_history);

    MPI_Finalize();
    return 0;
}
