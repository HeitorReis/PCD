#!/bin/bash

# Remove resultados anteriores
rm -f Resultados/sequencial.csv
rm -f Resultados/resultados.csv
rm -f Resultados/cuda.csv

# Compilação Serial
echo "======================================"
echo "Compilando kmeans_1d_naive..."
echo "======================================"

gcc -O2 -std=c99 src/Original/kmeans_1d_naive.c -o src/Original/kmeans_1d_naive -lm

if [ $? -ne 0 ]; then
    echo "Erro na compilação!"
    exit 1
fi

echo "✓ Compilação bem-sucedida!"
echo ""

# Cria diretórios de resultados
mkdir -p Resultados/Original

# Parâmetros comuns
MAX_ITER=500
EPS=1e-4
REPETICOES=10

# Tamanhos conhecidos dos datasets
N_PEQUENO=10000
N_MEDIO=100000
N_GRANDE=1000000

echo "======================================"
echo "Executando K-means Serial (Naive)"
echo "Repetições: $REPETICOES por dataset"
echo "======================================"
echo ""

# Loop de repetições para versão serial
for REP in $(seq 1 $REPETICOES); do
    echo "╔════════════════════════════════════════╗"
    echo "║ Repetição $REP/$REPETICOES (Serial)"
    echo "╚════════════════════════════════════════╝"
    echo ""

    # Dataset PEQUENO (N=10^4, K=4)
    echo ">>> Dataset PEQUENO (N=10,000, K=4) - Rep $REP"

    ./src/Original/kmeans_1d_naive \
        Geracao_dados/inputs/pequeno/dados.csv \
        Geracao_dados/inputs/pequeno/centroides_iniciais.csv \
        $MAX_ITER $EPS \
        Resultados/Original/assign_pequeno_r${REP}.csv \
        Resultados/Original/centroids_pequeno_r${REP}.csv \
        Resultados/Original/sse_pequeno_r${REP}.csv

    echo ""

    # Dataset MÉDIO (N=10^5, K=8)
    echo ">>> Dataset MÉDIO (N=100,000, K=8) - Rep $REP"

    ./src/Original/kmeans_1d_naive \
        Geracao_dados/inputs/medio/dados.csv \
        Geracao_dados/inputs/medio/centroides_iniciais.csv \
        $MAX_ITER $EPS \
        Resultados/Original/assign_medio_r${REP}.csv \
        Resultados/Original/centroids_medio_r${REP}.csv \
        Resultados/Original/sse_medio_r${REP}.csv

    echo ""

    # Dataset GRANDE (N=10^6, K=16)
    echo ">>> Dataset GRANDE (N=1,000,000, K=16) - Rep $REP"

    ./src/Original/kmeans_1d_naive \
        Geracao_dados/inputs/grande/dados.csv \
        Geracao_dados/inputs/grande/centroides_iniciais.csv \
        $MAX_ITER $EPS \
        Resultados/Original/assign_grande_r${REP}.csv \
        Resultados/Original/centroids_grande_r${REP}.csv \
        Resultados/Original/sse_grande_r${REP}.csv

    echo ""
done

echo "======================================"
echo "✓ Execuções Serial concluídas!"
echo "======================================"
echo ""

# Compilação OpenMP
echo "======================================"
echo "Compilando kmeans_1d_omp..."
echo "======================================"

gcc -O2 -std=c99 src/OpenMP/kmeans_1d_omp.c -o src/OpenMP/kmeans_1d_omp -lm -fopenmp

if [ $? -ne 0 ]; then
    echo "Erro na compilação!"
    exit 1
fi

echo "✓ Compilação bem-sucedida!"
echo ""

# Cria diretórios de resultados
rm -rf Resultados/OpenMP
mkdir -p Resultados/OpenMP

# Array de threads para testar
THREADS=(1 2 4 6 8 10 12 16 18 20 28 36 44)

echo "======================================"
echo "Executando K-means OpenMP"
echo "Repetições: $REPETICOES por configuração"
echo "======================================"
echo ""

# Loop por número de threads
for NUM_THREADS in "${THREADS[@]}"; do
    echo "╔════════════════════════════════════════╗"
    echo "║ Testando com $NUM_THREADS thread(s)"
    echo "╚════════════════════════════════════════╝"
    echo ""

    export OMP_NUM_THREADS=$NUM_THREADS

    # Loop de repetições
    for REP in $(seq 1 $REPETICOES); do
        echo ">>> Repetição $REP/$REPETICOES - Threads: $NUM_THREADS"
        echo ""

        # Dataset PEQUENO (N=10^4, K=4)
        echo " • PEQUENO (N=10,000, K=4)"

        ./src/OpenMP/kmeans_1d_omp \
            Geracao_dados/inputs/pequeno/dados.csv \
            Geracao_dados/inputs/pequeno/centroides_iniciais.csv \
            $MAX_ITER $EPS \
            Resultados/OpenMP/assign_pequeno_t${NUM_THREADS}_r${REP}.csv \
            Resultados/OpenMP/centroids_pequeno_t${NUM_THREADS}_r${REP}.csv \
            Resultados/OpenMP/sse_pequeno_t${NUM_THREADS}_r${REP}.csv \

        # Dataset MÉDIO (N=10^5, K=8)
        echo " • MÉDIO (N=100,000, K=8)"

        ./src/OpenMP/kmeans_1d_omp \
            Geracao_dados/inputs/medio/dados.csv \
            Geracao_dados/inputs/medio/centroides_iniciais.csv \
            $MAX_ITER $EPS \
            Resultados/OpenMP/assign_medio_t${NUM_THREADS}_r${REP}.csv \
            Resultados/OpenMP/centroids_medio_t${NUM_THREADS}_r${REP}.csv \
            Resultados/OpenMP/sse_medio_t${NUM_THREADS}_r${REP}.csv

        # Dataset GRANDE (N=10^6, K=16)
        echo " • GRANDE (N=1,000,000, K=16)"

        ./src/OpenMP/kmeans_1d_omp \
            Geracao_dados/inputs/grande/dados.csv \
            Geracao_dados/inputs/grande/centroides_iniciais.csv \
            $MAX_ITER $EPS \
            Resultados/OpenMP/assign_grande_t${NUM_THREADS}_r${REP}.csv \
            Resultados/OpenMP/centroids_grande_t${NUM_THREADS}_r${REP}.csv \
            Resultados/OpenMP/sse_grande_t${NUM_THREADS}_r${REP}.csv

        echo ""
    done

    echo "----------------------------------------"
    echo ""
done

echo "======================================"
echo "✓ Execuções OpenMP concluídas!"
echo "======================================"
echo "Resultados salvos em: Resultados/OpenMP/"
echo ""

###############################################################################
#                               BLOCO CUDA                                    #
###############################################################################

echo "======================================"
echo "Compilando e executando kmeans_1d_cuda..."
echo "======================================"
echo ""

# Cria diretório de resultados CUDA
rm -rf Resultados/CUDA
mkdir -p Resultados/CUDA

# Configurações de BLOCK_SIZE para testar (analogia ao número de threads do OpenMP)
CUDA_BLOCKS=(128 256 512)

echo "======================================"
echo "Executando K-means CUDA"
echo "Repetições: $REPETICOES por configuração"
echo "BLOCK_SIZEs: ${CUDA_BLOCKS[*]}"
echo "======================================"
echo ""

for BLOCK_SIZE in "${CUDA_BLOCKS[@]}"; do
    echo "╔════════════════════════════════════════╗"
    echo "║ Testando CUDA com BLOCK_SIZE=$BLOCK_SIZE"
    echo "╚════════════════════════════════════════╝"
    echo ""

    # Compila a versão CUDA para este BLOCK_SIZE
    nvcc -O2 -arch=sm_75 -DBLOCK_SIZE=$BLOCK_SIZE \
        src/CUDA/kmeans_1d_cuda.cu -o src/CUDA/kmeans_1d_cuda

    if [ $? -ne 0 ]; then
        echo "Erro na compilação CUDA (BLOCK_SIZE=$BLOCK_SIZE)!"
        exit 1
    fi

    # Loop de repetições
    for REP in $(seq 1 $REPETICOES); do
        echo ">>> Repetição $REP/$REPETICOES - BLOCK_SIZE: $BLOCK_SIZE"
        echo ""

        #####################
        # Dataset PEQUENO   #
        #####################
        echo " • PEQUENO (N=10,000, K=4)"

        ./src/CUDA/kmeans_1d_cuda \
            Geracao_dados/inputs/pequeno/dados.csv \
            Geracao_dados/inputs/pequeno/centroides_iniciais.csv \
            $MAX_ITER $EPS \
            Resultados/CUDA/assign_pequeno_b${BLOCK_SIZE}_r${REP}.csv \
            Resultados/CUDA/centroids_pequeno_b${BLOCK_SIZE}_r${REP}.csv \
            Resultados/CUDA/sse_pequeno_b${BLOCK_SIZE}_r${REP}.csv

        #####################
        # Dataset MÉDIO     #
        #####################
        echo " • MÉDIO (N=100,000, K=8)"

        ./src/CUDA/kmeans_1d_cuda \
            Geracao_dados/inputs/medio/dados.csv \
            Geracao_dados/inputs/medio/centroides_iniciais.csv \
            $MAX_ITER $EPS \
            Resultados/CUDA/assign_medio_b${BLOCK_SIZE}_r${REP}.csv \
            Resultados/CUDA/centroids_medio_b${BLOCK_SIZE}_r${REP}.csv \
            Resultados/CUDA/sse_medio_b${BLOCK_SIZE}_r${REP}.csv

        #####################
        # Dataset GRANDE    #
        #####################
        echo " • GRANDE (N=1,000,000, K=16)"

        ./src/CUDA/kmeans_1d_cuda \
            Geracao_dados/inputs/grande/dados.csv \
            Geracao_dados/inputs/grande/centroides_iniciais.csv \
            $MAX_ITER $EPS \
            Resultados/CUDA/assign_grande_b${BLOCK_SIZE}_r${REP}.csv \
            Resultados/CUDA/centroids_grande_b${BLOCK_SIZE}_r${REP}.csv \
            Resultados/CUDA/sse_grande_b${BLOCK_SIZE}_r${REP}.csv

        echo ""
    done

    echo "----------------------------------------"
    echo ""
done

echo "======================================"
echo "✓ Todas as execuções CUDA concluídas!"
echo "======================================"
echo "Resultados salvos em: Resultados/CUDA/"
echo "CSV agregado deverá ser gerado pelo próprio kmeans_1d_cuda em Resultados/cuda.csv"
echo ""

###############################################################################
#                               FIM BLOCO CUDA                                #
###############################################################################

# Gera gráficos
echo "======================================"
echo "Gerando gráficos..."
echo "======================================"

python3 graficos.py
