# Análise de Desempenho do K-Means 1D com OpenMP e CUDA

Este projeto consiste na implementação e análise de desempenho de um algoritmo K-Means unidimensional. Ele compara uma versão sequencial "ingênua" com uma versão paralela otimizada com OpenMP e com uma versão em CUDA para paralelização em GPU.

O projeto inclui:
*   Código-fonte em C para as versões sequencial e paralela do K-Means.
*   Um script em Python para gerar conjuntos de dados de teste.
*   Um script de shell (`script.sh`) para automatizar todo o processo de compilação, execução e coleta de métricas.
*   Um script em Python (`graficos.py`) para gerar gráficos de desempenho (Tempo de Execução, Speedup) a partir dos resultados coletados.

## Estrutura do Projeto

O repositório está organizado da seguinte forma:

*   `Geracao_dados/`: Contém scripts Python para gerar os conjuntos de dados.
    *   `gerador_gray_16.py`: Script principal para criar os datasets.
    *   `inputs/`: Subdiretório onde os dados (`.csv`) são salvos após a geração.

*   `src/`: Contém o código-fonte em C do algoritmo K-Means.
    *   `Original/`: Implementação sequencial (`kmeans_1d_naive.c`).
    *   `OpenMP/`: Implementação paralela com OpenMP (`kmeans_1d_omp.c`).
    *   `CUDA/`: Implementação paralela com CUDA (`kmeans_1d_cuda.cu`).

*   `Resultados/`: Diretório onde todos os artefatos de saída são salvos.
    *   `Original/`, `OpenMP/` e `CUDA/`: Resultados brutos de cada execução (atribuições, centróides, SSE).
    *   `Graficos/`: Gráficos de tempo de execução e speedup gerados pelo `graficos.py`.
    *   `sequencial.csv` e `resultados.csv`: Arquivos com as métricas de desempenho consolidadas.

*   `Relatorio/`: Contém os arquivos LaTeX para a elaboração do relatório do projeto.

*   `script.sh`: Script de shell que automatiza todo o processo: compilação, execução dos benchmarks e geração dos gráficos.

*   `graficos.py`: Script Python responsável por ler os arquivos de resultados (`.csv`) e gerar os gráficos de análise de desempenho.

## Pré-requisitos

Antes de executar, certifique-se de que você tem os seguintes softwares instalados:

*   **GCC**: Compilador C com suporte a OpenMP.
*   **cuda_toolkit**: Contem o compilador nvcc para CUDA
*   **Python 3**: Para os scripts de geração de dados e gráficos.
*   **Bibliotecas Python**: `pandas`, `numpy`, `matplotlib`.
    ```bash
    pip install pandas numpy matplotlib
    ```

## Como Executar o Projeto

A execução é dividida em duas etapas principais: geração dos dados de entrada e a execução dos testes de benchmark.

### 1. Gerar os Dados de Entrada

Primeiro, execute o script Python para criar os datasets (pequeno, médio e grande) que serão usados pelos algoritmos.

```bash
python3 Geracao_dados/gerador_gray_16.py
```

Este comando criará os arquivos `dados.csv` e `centroides_iniciais.csv` dentro dos diretórios `Geracao_dados/inputs/pequeno`, `Geracao_dados/inputs/medio` e `Geracao_dados/inputs/grande`.

### 2. Executar o Script de Benchmark

O script `script.sh` automatiza todo o fluxo de trabalho. Ele irá:
1.  Compilar as versões sequencial, OpenMP e CUDA do código C.
2.  Executar a versão sequencial 10 vezes para cada dataset para obter uma média de tempo de base.
3.  Executar a versão OpenMP 5 vezes para cada dataset, variando o número de threads (de 1 a 44).
4.  Executar a versão CUDA 10 vezes para cada dataset, variando o BLOCK SIZE (126, 256 e 512)
5.  Gerar os gráficos de desempenho e validação a partir dos resultados.

Para executar o script, dê permissão de execução e rode-o:

```bash
chmod +x script.sh
./script.sh
```

### 3. Verificar os Resultados

Após a conclusão do script, os resultados estarão organizados da seguinte forma:
*   **`Resultados/sequencial.csv`**: Métricas agregadas das execuções sequenciais.
*   **`Resultados/resultados.csv`**: Métricas agregadas das execuções paralelas com OpenMP.
*   **`Resultados/Graficos/`**: Pasta contendo todos os gráficos gerados, como "Tempo vs Threads" e "Speedup vs Threads".
*   **`Resultados/Original/`**, **`Resultados/OpenMP/`** e **`Resultados/CUDA/`**: Pastas com os resultados brutos de cada execução individual (atribuições, centróides e SSE por iteração).
