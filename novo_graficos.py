import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

SEQUENCIAL_PATH = "Resultados/sequencial.csv"
OMP_PATH = "Resultados/resultados.csv"
CUDA_PATH = "Resultados/cuda.csv"
MPI_PATH = "Resultados/mpi.csv"   # [MPI] CSV agregado da versão MPI

OUT_DIR = "Resultados/Graficos"
INPUTS_DIR = "Geracao_dados/inputs"
CENTROIDS_DIR = "Resultados/Original"

# [NOVO] Diretório e formatos de "resultados fáceis de ler"
REPORTS_DIR = "Resultados/Relatorios"
REPORT_CSV = os.path.join(REPORTS_DIR, "resumo_resultados.csv")
REPORT_HTML = os.path.join(REPORTS_DIR, "resumo_resultados.html")  # alta compatibilidade (abre em browser/Word)
REPORT_MD = os.path.join(REPORTS_DIR, "resumo_resultados.md")      # opcional: fácil colar no relatório


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ============================================================
#        [NOVO] EXPORTAÇÃO DE RESULTADOS NÃO-IMAGEM
# ============================================================

def _detect_param_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _standardize_df(df, method_name, param_label, param_col=None):
    """
    Padroniza colunas para permitir juntar tudo num único resumo:
    - Method: naive/openmp/mpi/cuda
    - ParamLabel: Threads/Procs/BlockSize/-
    - ParamValue: valor do parâmetro (ou NaN)
    - Tamanho, Tempo(ms), SSE_Final, Iteracoes, Speedup, pontos/s
    """
    out = df.copy()
    out["Method"] = method_name
    out["ParamLabel"] = param_label

    if param_col and param_col in out.columns:
        out["ParamValue"] = out[param_col]
    else:
        out["ParamValue"] = np.nan

    # Mantém apenas colunas "núcleo" + o que já existir
    keep = ["Method", "ParamLabel", "ParamValue",
            "Tamanho", "Tempo(ms)", "SSE_Final", "Iteracoes",
            "Speedup", "pontos/s"]
    # Filtra só as que existem
    keep = [c for c in keep if c in out.columns]
    out = out[keep]
    return out


def _aggregate_summary(df, group_cols):
    """
    Gera resumo por grupo com média e desvio padrão:
    - Tempo_mean/std, Speedup_mean/std, pontos/s_mean/std, Iteracoes_mean/std, SSE_Final_mean/std
    """
    metrics = []
    for col in ["Tempo(ms)", "Speedup", "pontos/s", "Iteracoes", "SSE_Final"]:
        if col in df.columns:
            metrics.append(col)

    if not metrics:
        return pd.DataFrame()

    agg_map = {}
    for m in metrics:
        agg_map[m] = ["mean", "std", "min", "max"]

    g = df.groupby(group_cols).agg(agg_map).reset_index()

    # Achata multi-index de colunas
    new_cols = []
    for c in g.columns:
        if isinstance(c, tuple):
            base, stat = c
            if stat == "":
                new_cols.append(base)
            else:
                new_cols.append(f"{base}_{stat}")
        else:
            new_cols.append(c)
    g.columns = new_cols

    # Ordenação amigável
    for c in ["Tamanho", "ParamValue"]:
        if c in g.columns:
            g = g.sort_values(by=[c] + [x for x in group_cols if x not in [c]])
            break

    return g


def export_readable_reports(seq_path=SEQUENCIAL_PATH,
                            omp_path=OMP_PATH,
                            cuda_path=CUDA_PATH,
                            mpi_path=MPI_PATH,
                            out_dir=REPORTS_DIR):
    """
    [NOVO] Gera arquivos não-imagem, fáceis de ler e compatíveis, com resultados:
    - CSV (resumo_resultados.csv) -> alta compatibilidade (Excel/Sheets)
    - HTML (resumo_resultados.html) -> abre direto no navegador/Word
    - MD  (resumo_resultados.md) -> fácil colar no relatório
    Também gera um CSV "detalhado" consolidado (todas as execuções) se existirem.
    """
    ensure_dir(out_dir)

    frames_all = []
    frames_summary = []

    # -------- Serial / Naive --------
    if os.path.exists(seq_path):
        df_seq = pd.read_csv(seq_path)
        df_seq_std = _standardize_df(df_seq, method_name="naive", param_label="-", param_col=None)
        frames_all.append(df_seq_std)

        # Resumo por tamanho
        sum_seq = _aggregate_summary(df_seq_std, group_cols=["Method", "Tamanho"])
        if not sum_seq.empty:
            frames_summary.append(sum_seq)
    else:
        print(f"[Relatório] Serial não encontrado: {seq_path}")

    # -------- OpenMP --------
    if os.path.exists(omp_path):
        df_omp = pd.read_csv(omp_path)
        param_col = _detect_param_column(df_omp, ["Threads", "thread", "NUM_THREADS"])
        df_omp_std = _standardize_df(df_omp, method_name="openmp", param_label="Threads",
                                     param_col=param_col)
        frames_all.append(df_omp_std)

        sum_omp = _aggregate_summary(df_omp_std, group_cols=["Method", "Tamanho", "ParamValue"])
        if not sum_omp.empty:
            frames_summary.append(sum_omp)
    else:
        print(f"[Relatório] OpenMP não encontrado: {omp_path}")

    # -------- CUDA --------
    if os.path.exists(cuda_path):
        df_cuda = pd.read_csv(cuda_path)
        param_col = _detect_param_column(df_cuda, ["BlockSize", "BLOCK_SIZE", "block", "bs"])
        df_cuda_std = _standardize_df(df_cuda, method_name="cuda", param_label="BlockSize",
                                      param_col=param_col)
        frames_all.append(df_cuda_std)

        sum_cuda = _aggregate_summary(df_cuda_std, group_cols=["Method", "Tamanho", "ParamValue"])
        if not sum_cuda.empty:
            frames_summary.append(sum_cuda)
    else:
        print(f"[Relatório] CUDA não encontrado: {cuda_path}")

    # -------- MPI --------
    if os.path.exists(mpi_path):
        df_mpi = pd.read_csv(mpi_path)
        param_col = _detect_param_column(df_mpi, ["Procs", "Processos", "NP", "NProcs", "Threads"])
        df_mpi_std = _standardize_df(df_mpi, method_name="mpi", param_label="Procs",
                                     param_col=param_col)
        frames_all.append(df_mpi_std)

        sum_mpi = _aggregate_summary(df_mpi_std, group_cols=["Method", "Tamanho", "ParamValue"])
        if not sum_mpi.empty:
            frames_summary.append(sum_mpi)
    else:
        print(f"[Relatório] MPI não encontrado: {mpi_path}")

    # Consolidado detalhado (todas execuções)
    if frames_all:
        df_all = pd.concat(frames_all, ignore_index=True)
        detailed_path = os.path.join(out_dir, "resultados_detalhados_todos.csv")
        df_all.to_csv(detailed_path, index=False)
        print(f"✓ [Relatório] Salvo detalhado: {detailed_path}")
    else:
        df_all = pd.DataFrame()

    # Consolidado resumo (médias/std)
    if frames_summary:
        df_summary = pd.concat(frames_summary, ignore_index=True)
        df_summary.to_csv(REPORT_CSV, index=False)
        print(f"✓ [Relatório] Salvo resumo CSV: {REPORT_CSV}")

        # HTML (alta compatibilidade, fácil de abrir)
        html = []
        html.append("<html><head><meta charset='utf-8'>")
        html.append("<style>")
        html.append("body{font-family:Arial,Helvetica,sans-serif;margin:20px;}")
        html.append("table{border-collapse:collapse;width:100%;}")
        html.append("th,td{border:1px solid #ccc;padding:6px;font-size:12px;}")
        html.append("th{background:#f5f5f5;}")
        html.append("</style></head><body>")
        html.append("<h2>Resumo de Resultados (Naive / OpenMP / MPI / CUDA)</h2>")
        html.append("<p>Gerado automaticamente pelo graficos.py</p>")
        html.append(df_summary.to_html(index=False))
        html.append("</body></html>")

        with open(REPORT_HTML, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"✓ [Relatório] Salvo resumo HTML: {REPORT_HTML}")

        # Markdown (útil para colar no relatório)
        try:
            md = df_summary.to_markdown(index=False)
        except Exception:
            md = df_summary.to_string(index=False)
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"✓ [Relatório] Salvo resumo MD: {REPORT_MD}")
    else:
        print("[Relatório] Não foi possível gerar resumo (nenhum CSV agregado disponível).")


# ============================================================
#            SPEEDUP / MÉTRICAS – OPENMP (CPU)
# ============================================================

def add_speedup(seq_path=SEQUENCIAL_PATH, omp_path=OMP_PATH):
    """
    Lê sequencial.csv e resultados.csv (OpenMP), e acrescenta colunas:
    - Speedup (Tempo_serial / Tempo_omp)
    - pontos/s (throughput usando o tempo paralelo)
    """
    if not os.path.exists(seq_path):
        print(f"Arquivo não encontrado: {seq_path}")
        return 1

    if not os.path.exists(omp_path):
        print(f"Arquivo não encontrado: {omp_path}")
        return 1

    df_seq = pd.read_csv(seq_path)
    df_omp = pd.read_csv(omp_path)

    if 'Tamanho' not in df_seq.columns or 'Tempo(ms)' not in df_seq.columns:
        print("sequencial.csv deve ter colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes")
        return 1

    if 'Tamanho' not in df_omp.columns or 'Tempo(ms)' not in df_omp.columns:
        print("resultados.csv deve ter colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes, Threads")
        return 1

    # Calcula média do tempo serial por tamanho
    df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
    df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})

    df_merged = df_omp.merge(df_seq_agg, on='Tamanho', how='left')
    df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']

    # Nova coluna: throughput (pontos/s) usando o tempo paralelo
    tempo_ms = df_merged['Tempo(ms)'].replace(0, np.nan)
    df_merged['pontos/s'] = (df_merged['Tamanho'] * 1000.0) / tempo_ms

    if 'Tempo_serial(ms)' in df_merged.columns:
        df_merged = df_merged.drop(columns=['Tempo_serial(ms)'])

    df_merged.to_csv(omp_path, index=False)
    print(f"✓ Speedup e 'pontos/s' adicionados em {omp_path}")
    return 0


def plot_tempo_vs_threads(df_omp, out_dir=OUT_DIR):
    """
    Tempo de execução vs Threads (OpenMP) para diferentes tamanhos de entrada.
    """
    ensure_dir(out_dir)

    if 'Threads' not in df_omp.columns:
        print("Sem coluna 'Threads' em resultados.csv. Pulando gráfico Tempo vs Threads.")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_omp['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_omp['Tamanho'].unique())):
        g = df_omp[df_omp['Tamanho'] == N].copy()

        # Média e desvio padrão por número de threads
        agg = g.groupby('Threads').agg({
            'Tempo(ms)': ['mean', 'std']
        }).reset_index()
        agg.columns = ['Threads', 'Tempo_mean', 'Tempo_std']
        agg = agg.sort_values('Threads')

        if agg.empty:
            continue

        x = agg['Threads'].values
        y_mean = agg['Tempo_mean'].values
        y_std = agg['Tempo_std'].values

        plt.errorbar(
            x, y_mean, yerr=y_std,
            marker='o', capsize=5, capthick=1.5,
            label=f'N={N:,}',
            color=colors[i]
        )

    plt.title("Tempo de Execução vs Threads para Diferentes Entradas")
    plt.xlabel("Threads")
    plt.ylabel("Tempo (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "tempo_vs_threads_geral.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")


def plot_speedup_vs_threads(df_omp, df_seq, out_dir=OUT_DIR):
    """
    Speedup (Serial vs OpenMP) vs Threads para diferentes tamanhos de entrada.
    """
    ensure_dir(out_dir)

    if 'Threads' not in df_omp.columns:
        print("Sem coluna 'Threads' em resultados.csv. Pulando gráfico Speedup vs Threads.")
        return

    # Recalcula speedup se necessário
    if 'Speedup' not in df_omp.columns:
        df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
        df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})
        df_merged = df_omp.merge(df_seq_agg, on='Tamanho', how='left')
        df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']
    else:
        df_merged = df_omp.copy()

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_merged['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_merged['Tamanho'].unique())):
        g = df_merged[df_merged['Tamanho'] == N].copy()

        agg = g.groupby('Threads').agg({
            'Speedup': ['mean', 'std']
        }).reset_index()
        agg.columns = ['Threads', 'Speedup_mean', 'Speedup_std']
        agg = agg.sort_values('Threads')

        if agg.empty:
            continue

        x = agg['Threads'].values
        y_mean = agg['Speedup_mean'].values
        y_std = agg['Speedup_std'].values

        plt.errorbar(
            x, y_mean, yerr=y_std,
            marker='o', capsize=5, capthick=1.5,
            label=f'N={N:,}',
            color=colors[i]
        )

    # Linha ideal (exemplo, saturando em 12)
    all_threads = sorted(df_merged['Threads'].unique())
    ideal = np.minimum(all_threads, 12)
    plt.plot(all_threads, ideal, '--', color='gray', alpha=0.8, label='Ideal (satura em 12)')

    plt.title("Speedup vs Threads para Diferentes Tamanhos de Entrada")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "speedup_vs_threads_geral.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")


# ============================================================
#            SPEEDUP / MÉTRICAS – CUDA (GPU)
# ============================================================

def add_speedup_cuda(seq_path=SEQUENCIAL_PATH, cuda_path=CUDA_PATH):
    """
    Lê sequencial.csv e cuda.csv, e acrescenta colunas:
    - Speedup (Tempo_serial / Tempo_cuda)
    - pontos/s (throughput usando o tempo CUDA)
    """
    if not os.path.exists(seq_path):
        print(f"Arquivo não encontrado (serial): {seq_path}")
        return 1

    if not os.path.exists(cuda_path):
        print(f"Arquivo não encontrado (CUDA): {cuda_path}")
        return 1

    df_seq = pd.read_csv(seq_path)
    df_cuda = pd.read_csv(cuda_path)

    if 'Tamanho' not in df_seq.columns or 'Tempo(ms)' not in df_seq.columns:
        print("sequencial.csv deve ter colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes")
        return 1

    if 'Tamanho' not in df_cuda.columns or 'Tempo(ms)' not in df_cuda.columns:
        print("cuda.csv deve ter colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes, BlockSize")
        return 1

    df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
    df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})

    df_merged = df_cuda.merge(df_seq_agg, on='Tamanho', how='left')
    df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']

    tempo_ms = df_merged['Tempo(ms)'].replace(0, np.nan)
    df_merged['pontos/s'] = (df_merged['Tamanho'] * 1000.0) / tempo_ms

    if 'Tempo_serial(ms)' in df_merged.columns:
        df_merged = df_merged.drop(columns=['Tempo_serial(ms)'])

    df_merged.to_csv(cuda_path, index=False)
    print(f"✓ Speedup e 'pontos/s' adicionados em {cuda_path}")
    return 0


def plot_tempo_vs_blocksize(df_cuda, out_dir=OUT_DIR):
    """
    Tempo de execução vs BlockSize (CUDA) para diferentes tamanhos de entrada.
    """
    ensure_dir(out_dir)

    if 'BlockSize' not in df_cuda.columns:
        print("Sem coluna 'BlockSize' em cuda.csv. Pulando gráfico Tempo vs BlockSize (CUDA).")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_cuda['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_cuda['Tamanho'].unique())):
        g = df_cuda[df_cuda['Tamanho'] == N].copy()

        agg = g.groupby('BlockSize').agg({
            'Tempo(ms)': ['mean', 'std']
        }).reset_index()
        agg.columns = ['BlockSize', 'Tempo_mean', 'Tempo_std']
        agg = agg.sort_values('BlockSize')

        if agg.empty:
            continue

        x = agg['BlockSize'].values
        y_mean = agg['Tempo_mean'].values
        y_std = agg['Tempo_std'].values

        plt.errorbar(
            x, y_mean, yerr=y_std,
            marker='o', capsize=5, capthick=1.5,
            label=f'N={N:,}',
            color=colors[i]
        )

    plt.title("Tempo de Execução vs BlockSize (CUDA)")
    plt.xlabel("BlockSize (threads por bloco)")
    plt.ylabel("Tempo (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "tempo_vs_blocksize_cuda.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")


def plot_speedup_vs_blocksize(df_cuda, df_seq, out_dir=OUT_DIR):
    """
    Speedup (Serial vs CUDA) vs BlockSize para diferentes tamanhos de entrada.
    """
    ensure_dir(out_dir)

    if 'BlockSize' not in df_cuda.columns:
        print("Sem coluna 'BlockSize' em cuda.csv. Pulando gráfico Speedup vs BlockSize (CUDA).")
        return

    # Recalcula speedup se necessário
    if 'Speedup' not in df_cuda.columns:
        df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
        df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})
        df_merged = df_cuda.merge(df_seq_agg, on='Tamanho', how='left')
        df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']
    else:
        df_merged = df_cuda.copy()

    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_merged['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_merged['Tamanho'].unique())):
        g = df_merged[df_merged['Tamanho'] == N].copy()

        agg = g.groupby('BlockSize').agg({
            'Speedup': ['mean', 'std']
        }).reset_index()
        agg.columns = ['BlockSize', 'Speedup_mean', 'Speedup_std']
        agg = agg.sort_values('BlockSize')

        if agg.empty:
            continue

        x = agg['BlockSize'].values
        y_mean = agg['Speedup_mean'].values
        y_std = agg['Speedup_std'].values

        plt.errorbar(
            x, y_mean, yerr=y_std,
            marker='o', capsize=5, capthick=1.5,
            label=f'N={N:,}',
            color=colors[i]
        )

    plt.title("Speedup (Serial vs CUDA) vs BlockSize")
    plt.xlabel("BlockSize (threads por bloco)")
    plt.ylabel("Speedup")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "speedup_vs_blocksize_cuda.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")


# ============================================================
#            SPEEDUP / MÉTRICAS – MPI (CPU / DISTRIBUÍDO)
# ============================================================

def add_speedup_mpi(seq_path=SEQUENCIAL_PATH, mpi_path=MPI_PATH):
    """
    Lê sequencial.csv e mpi.csv, e acrescenta colunas:
    - Speedup (Tempo_serial / Tempo_mpi)
    - pontos/s (throughput usando o tempo MPI)

    A lógica é análoga à de add_speedup (OpenMP), mas aplicada ao CSV do MPI.
    """
    if not os.path.exists(seq_path):
        print(f"Arquivo não encontrado (serial): {seq_path}")
        return 1

    if not os.path.exists(mpi_path):
        print(f"Arquivo não encontrado (MPI): {mpi_path}")
        return 1

    df_seq = pd.read_csv(seq_path)
    df_mpi = pd.read_csv(mpi_path)

    if 'Tamanho' not in df_seq.columns or 'Tempo(ms)' not in df_seq.columns:
        print("sequencial.csv deve ter colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes")
        return 1

    if 'Tamanho' not in df_mpi.columns or 'Tempo(ms)' not in df_mpi.columns:
        print("mpi.csv deve ter, no mínimo, colunas: Tempo(ms), Tamanho, SSE_Final, Iteracoes")
        return 1

    df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
    df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})

    df_merged = df_mpi.merge(df_seq_agg, on='Tamanho', how='left')
    df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']

    tempo_ms = df_merged['Tempo(ms)'].replace(0, np.nan)
    df_merged['pontos/s'] = (df_merged['Tamanho'] * 1000.0) / tempo_ms

    if 'Tempo_serial(ms)' in df_merged.columns:
        df_merged = df_merged.drop(columns=['Tempo_serial(ms)'])

    df_merged.to_csv(mpi_path, index=False)
    print(f"✓ Speedup e 'pontos/s' adicionados em {mpi_path}")
    return 0


def _get_mpi_procs_column(df_mpi):
    """
    Tenta descobrir a coluna que representa o número de processos do MPI.
    Aceita alguns nomes comuns para ser robusto à implementação:
    - 'Procs', 'Processos', 'NP', 'NProcs', 'Threads' (se você resolveu reaproveitar).
    """
    candidates = ['Procs', 'Processos', 'NP', 'NProcs', 'Threads']
    for c in candidates:
        if c in df_mpi.columns:
            return c
    return None


def plot_tempo_vs_procs_mpi(df_mpi, out_dir=OUT_DIR):
    """
    Tempo de execução vs número de processos (MPI) para diferentes tamanhos de entrada.
    Estrutura análoga a plot_tempo_vs_threads, mas usando a coluna de processos.
    """
    ensure_dir(out_dir)

    col_procs = _get_mpi_procs_column(df_mpi)
    if col_procs is None:
        print("Não foi encontrada coluna de número de processos em mpi.csv. "
              "Esperado algo como 'Procs', 'Processos', 'NP', 'NProcs' ou 'Threads'. "
              "Pulando gráfico Tempo vs Processos (MPI).")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.cividis(np.linspace(0, 1, len(df_mpi['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_mpi['Tamanho'].unique())):
        g = df_mpi[df_mpi['Tamanho'] == N].copy()

        agg = g.groupby(col_procs).agg({
            'Tempo(ms)': ['mean', 'std']
        }).reset_index()
        agg.columns = [col_procs, 'Tempo_mean', 'Tempo_std']
        agg = agg.sort_values(col_procs)

        if agg.empty:
            continue

        x = agg[col_procs].values
        y_mean = agg['Tempo_mean'].values
        y_std = agg['Tempo_std'].values

        plt.errorbar(
            x, y_mean, yerr=y_std,
            marker='o', capsize=5, capthick=1.5,
            label=f'N={N:,}',
            color=colors[i]
        )

    plt.title("Tempo de Execução vs Número de Processos (MPI)")
    plt.xlabel("Processos MPI")
    plt.ylabel("Tempo (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "tempo_vs_procs_mpi.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")


def plot_speedup_vs_procs_mpi(df_mpi, df_seq, out_dir=OUT_DIR):
    """
    Speedup (Serial vs MPI) vs número de processos para diferentes tamanhos de entrada.
    Estrutura análoga a plot_speedup_vs_threads.
    """
    ensure_dir(out_dir)

    col_procs = _get_mpi_procs_column(df_mpi)
    if col_procs is None:
        print("Não foi encontrada coluna de número de processos em mpi.csv. "
              "Pulando gráfico Speedup vs Processos (MPI).")
        return

    # Recalcula speedup se necessário
    if 'Speedup' not in df_mpi.columns:
        df_seq_agg = df_seq.groupby('Tamanho').agg({'Tempo(ms)': 'mean'}).reset_index()
        df_seq_agg = df_seq_agg.rename(columns={'Tempo(ms)': 'Tempo_serial(ms)'})
        df_merged = df_mpi.merge(df_seq_agg, on='Tamanho', how='left')
        df_merged['Speedup'] = df_merged['Tempo_serial(ms)'] / df_merged['Tempo(ms)']
    else:
        df_merged = df_mpi.copy()

    plt.figure(figsize=(10, 6))
    colors = plt.cm.cividis(np.linspace(0, 1, len(df_merged['Tamanho'].unique())))

    for i, N in enumerate(sorted(df_merged['Tamanho'].unique())):
        g = df_merged[df_merged['Tamanho'] == N].copy()

        agg = g.groupby(col_procs).agg({
            'Speedup': ['mean', 'std']
        }).reset_index()
        agg.columns = [col_procs, 'Speedup_mean', 'Speedup_std']
        agg = agg.sort_values(col_procs)

        if agg.empty:
            continue

        x = agg[col_procs].values
        y_mean = agg['Speedup_mean'].values
        y_std = agg['Speedup_std'].values

        plt.errorbar(
            x, y_mean, yerr=y_std,
            marker='o', capsize=5, capthick=1.5,
            label=f'N={N:,}',
            color=colors[i]
        )

    plt.title("Speedup (Serial vs MPI) vs Número de Processos")
    plt.xlabel("Processos MPI")
    plt.ylabel("Speedup")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Tamanho (N)")
    out_path = os.path.join(out_dir, "speedup_vs_procs_mpi.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Salvo: {out_path}")


# ============================================================
#                     SSE POR ITERAÇÃO
# ============================================================

def _read_sse_series(path):
    """
    Lê série de SSE (um valor por linha ou coluna SSE).
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            sse = df.iloc[:, 0].values.astype(float)
        elif 'SSE' in df.columns:
            sse = df['SSE'].values.astype(float)
        else:
            sse = df.iloc[:, 0].values.astype(float)
        return sse
    except Exception as e:
        print(f"Falha ao ler SSE de {path}: {e}")
        return None


def plot_sse_validacao(labels=('pequeno', 'medio', 'grande'),
                       threads_ref=4,
                       out_dir=OUT_DIR):
    """
    Plota SSE por iteração comparando Serial vs OpenMP.
    Usa a última repetição disponível de cada configuração.
    """
    ensure_dir(out_dir)

    serial_dir = "Resultados/Original"
    omp_dir = "Resultados/OpenMP"

    plt.figure(figsize=(12, 7))
    any_plotted = False

    colors = {
        'pequeno': ('#E15759', '#F28E2B'),
        'medio': ('#4E79A7', '#59A14F'),
        'grande': ('#76B7B2', '#B07AA1')
    }

    for lbl in labels:
        # Serial
        serial_pattern = os.path.join(serial_dir, f"sse_{lbl}_r*.csv")
        serial_files = sorted(glob(serial_pattern))
        serial_path = serial_files[-1] if serial_files else None

        # OpenMP com threads_ref
        omp_pattern = os.path.join(omp_dir, f"sse_{lbl}_t{threads_ref}_r*.csv")
        omp_files = sorted(glob(omp_pattern))
        omp_path = omp_files[-1] if omp_files else None

        if not serial_path and not omp_path:
            print(f"SSE não encontrado para '{lbl}'.")
            print(f" Serial: {serial_pattern}")
            print(f" OpenMP: {omp_pattern}")
            continue

        sse_serial = _read_sse_series(serial_path) if serial_path else None
        sse_omp = _read_sse_series(omp_path) if omp_path else None

        if sse_serial is None and sse_omp is None:
            print(f"Falha ao ler SSE para '{lbl}'. Pulando.")
            continue

        any_plotted = True

        if sse_serial is not None:
            it_serial = np.arange(1, len(sse_serial) + 1)
            plt.plot(
                it_serial, sse_serial,
                label=f'Serial ({lbl})',
                marker='.', markersize=4,
                linestyle='-',
                lw=2,
                color=colors.get(lbl, ('black', 'gray'))[0],
                alpha=0.9
            )

        if sse_omp is not None:
            it_omp = np.arange(1, len(sse_omp) + 1)
            plt.plot(
                it_omp, sse_omp,
                label=f'OpenMP ({lbl}, {threads_ref}t)',
                marker='.', markersize=4,
                linestyle='--',
                lw=2,
                color=colors.get(lbl, ('black', 'gray'))[1],
                alpha=0.9
            )

    if any_plotted:
        plt.title("Convergência do SSE (Serial vs OpenMP)", fontsize=14, fontweight='bold')
        plt.xlabel("Iteração", fontsize=12)
        plt.ylabel("SSE (escala log)", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
        plt.legend(loc='best', framealpha=0.95, fontsize=10)
        plt.yscale('log')
        out_path = os.path.join(out_dir, "sse_por_iteracao_geral.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✓ Salvo: {out_path}")
    else:
        plt.close()
        print("Nenhum gráfico de SSE por iteração gerado (arquivos não encontrados).")


def plot_sse_validacao_cuda(labels=('pequeno', 'medio', 'grande'),
                            blocksize_ref=256,
                            out_dir=OUT_DIR):
    """
    Plota SSE por iteração comparando Serial vs CUDA.
    Usa a última repetição disponível para um BlockSize de referência.
    """
    ensure_dir(out_dir)

    serial_dir = "Resultados/Original"
    cuda_dir = "Resultados/CUDA"

    plt.figure(figsize=(12, 7))
    any_plotted = False

    colors = {
        'pequeno': ('#E15759', '#FFBE7D'),
        'medio': ('#4E79A7', '#9CBAE5'),
        'grande': ('#76B7B2', '#C5DFD5')
    }

    for lbl in labels:
        # Serial
        serial_pattern = os.path.join(serial_dir, f"sse_{lbl}_r*.csv")
        serial_files = sorted(glob(serial_pattern))
        serial_path = serial_files[-1] if serial_files else None

        # CUDA para BlockSize de referência
        cuda_pattern = os.path.join(cuda_dir, f"sse_{lbl}_b{blocksize_ref}_r*.csv")
        cuda_files = sorted(glob(cuda_pattern))
        cuda_path = cuda_files[-1] if cuda_files else None

        if not serial_path and not cuda_path:
            print(f"SSE não encontrado para '{lbl}' (Serial/CUDA).")
            print(f" Serial: {serial_pattern}")
            print(f" CUDA  : {cuda_pattern}")
            continue

        sse_serial = _read_sse_series(serial_path) if serial_path else None
        sse_cuda = _read_sse_series(cuda_path) if cuda_path else None

        if sse_serial is None and sse_cuda is None:
            print(f"Falha ao ler SSE para '{lbl}' (Serial/CUDA). Pulando.")
            continue

        any_plotted = True

        if sse_serial is not None:
            it_serial = np.arange(1, len(sse_serial) + 1)
            plt.plot(
                it_serial, sse_serial,
                label=f'Serial ({lbl})',
                marker='.', markersize=4,
                linestyle='-',
                lw=2,
                color=colors.get(lbl, ('black', 'gray'))[0],
                alpha=0.9
            )

        if sse_cuda is not None:
            it_cuda = np.arange(1, len(sse_cuda) + 1)
            plt.plot(
                it_cuda, sse_cuda,
                label=f'CUDA ({lbl}, BS={blocksize_ref})',
                marker='.', markersize=4,
                linestyle='--',
                lw=2,
                color=colors.get(lbl, ('black', 'gray'))[1],
                alpha=0.9
            )

    if any_plotted:
        plt.title("Convergência do SSE (Serial vs CUDA)", fontsize=14, fontweight='bold')
        plt.xlabel("Iteração", fontsize=12)
        plt.ylabel("SSE (escala log)", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
        plt.legend(loc='best', framealpha=0.95, fontsize=10)
        plt.yscale('log')
        out_path = os.path.join(out_dir, "sse_por_iteracao_cuda.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✓ Salvo: {out_path}")
    else:
        plt.close()
        print("Nenhum gráfico de SSE (Serial vs CUDA) foi gerado (arquivos não encontrados).")


def plot_sse_mpi(labels=('pequeno', 'medio', 'grande'),
                 procs_ref=8,
                 out_dir=OUT_DIR):
    """
    Plota SSE por iteração comparando Serial vs MPI.
    Usa a última repetição disponível para um número fixo de processos (procs_ref).
    """
    ensure_dir(out_dir)

    serial_dir = "Resultados/Original"
    mpi_dir = "Resultados/MPI"

    plt.figure(figsize=(12, 7))
    any_plotted = False

    colors = {
        'pequeno': ('#E15759', '#4E79A7'),
        'medio':   ('#59A14F', '#FF9DA7'),
        'grande':  ('#76B7B2', '#F28E2B')
    }

    for lbl in labels:
        serial_pattern = os.path.join(serial_dir, f"sse_{lbl}_r*.csv")
        serial_files = sorted(glob(serial_pattern))
        serial_path = serial_files[-1] if serial_files else None

        mpi_pattern = os.path.join(mpi_dir, f"sse_{lbl}_p{procs_ref}_r*.csv")
        mpi_files = sorted(mpi_pattern and glob(mpi_pattern) or [])
        mpi_path = mpi_files[-1] if mpi_files else None

        if not serial_path and not mpi_path:
            print(f"SSE não encontrado para '{lbl}' (Serial/MPI).")
            print(f" Serial: {serial_pattern}")
            print(f" MPI   : {mpi_pattern}")
            continue

        sse_serial = _read_sse_series(serial_path) if serial_path else None
        sse_mpi = _read_sse_series(mpi_path) if mpi_path else None

        if sse_serial is None and sse_mpi is None:
            print(f"Falha ao ler SSE para '{lbl}' (Serial/MPI). Pulando.")
            continue

        any_plotted = True

        if sse_serial is not None:
            it_serial = np.arange(1, len(sse_serial) + 1)
            plt.plot(
                it_serial,
                sse_serial,
                label=f"{lbl} - Serial",
                linestyle='-',
                lw=2,
                color=colors.get(lbl, ('black', 'gray'))[0],
                alpha=0.9
            )

        if sse_mpi is not None:
            it_mpi = np.arange(1, len(sse_mpi) + 1)
            plt.plot(
                it_mpi,
                sse_mpi,
                label=f"{lbl} - MPI (P={procs_ref})",
                linestyle='--',
                lw=2,
                color=colors.get(lbl, ('black', 'gray'))[1],
                alpha=0.9
            )

    if any_plotted:
        plt.title("Convergência do SSE (Serial vs MPI)", fontsize=14, fontweight='bold')
        plt.xlabel("Iteração", fontsize=12)
        plt.ylabel("SSE (escala log)", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
        plt.legend(loc='best', framealpha=0.95, fontsize=10)
        plt.yscale('log')
        out_path = os.path.join(out_dir, "sse_por_iteracao_mpi.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"✓ Salvo: {out_path}")
    else:
        plt.close()
        print("Nenhum gráfico de SSE (MPI) gerado (arquivos não encontrados).")


# ============================================================
#          AUXILIARES PARA DISTRIBUIÇÃO E CENTRÓIDES
# ============================================================

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _read_vector_csv_general(path, candidate_cols=('altura_cm', 'x', 'valor', 'value')):
    """
    Lê um vetor 1D a partir de um CSV com 1 coluna numérica ou coluna conhecida.
    """
    try:
        df = pd.read_csv(path)

        for c in candidate_cols:
            if c in df.columns:
                return df[c].astype(float).values

        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return df[c].astype(float).values

        return df.iloc[:, 0].astype(float).values
    except Exception as e:
        print(f"Falha ao ler vetor de {path}: {e}")
        return None


def _find_centroids_for_label(label, centroids_dir):
    candidates = [
        os.path.join(centroids_dir, f"centroids_{label}.csv"),
        os.path.join(centroids_dir, f"centroids_{label}.txt"),
        os.path.join(centroids_dir, f"centroids_{label}.dat"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    try:
        for fname in os.listdir(centroids_dir):
            if fname.startswith(f"centroids_{label}"):
                return os.path.join(centroids_dir, fname)
    except FileNotFoundError:
        return None

    digits = ''.join(ch for ch in label if ch.isdigit())
    if digits:
        try:
            for fname in os.listdir(centroids_dir):
                if fname.startswith("centroids_") and digits in fname:
                    return os.path.join(centroids_dir, fname)
        except FileNotFoundError:
            pass

    return None


def plot_distribuicoes_inputs_e_centroides(inputs_dir=INPUTS_DIR,
                                           centroids_dir=CENTROIDS_DIR,
                                           out_dir=OUT_DIR,
                                           bins=40):
    ensure_dir(out_dir)

    if not os.path.isdir(inputs_dir):
        print(f"Diretório de inputs não encontrado: {inputs_dir}")
        return

    files = sorted([
        f for f in os.listdir(inputs_dir)
        if f.lower().endswith(('.csv', '.txt', '.dat'))
    ])

    if not files:
        print(f"Nenhum arquivo de dados encontrado em {inputs_dir}")
        return

    any_plotted = False

    for fname in files:
        label = os.path.splitext(fname)[0]
        data_path = os.path.join(inputs_dir, fname)

        dados = _read_vector_csv_general(data_path)
        if dados is None or len(dados) == 0:
            print(f"Falha ao ler dados de {data_path}.")
            continue

        cent_path = _find_centroids_for_label(label, centroids_dir)
        centroides = _read_vector_csv_general(cent_path) if cent_path else None

        if not cent_path:
            print(f"Centroides não encontrados para '{label}' em {centroids_dir}. Plotando só a distribuição.")

        plt.figure(figsize=(8, 4.5))
        plt.hist(dados, bins=bins, alpha=0.45, color='#4C78A8', edgecolor='white')

        sample = dados if len(dados) <= 800 else np.random.choice(dados, size=800, replace=False)
        plt.plot(sample, np.zeros_like(sample), '|', color='black', alpha=0.25)

        if centroides is not None and len(centroides) > 0:
            ymax = plt.ylim()[1]
            for i, c in enumerate(sorted(centroides)):
                plt.axvline(c, color='#F58518', linestyle='--', linewidth=2, alpha=0.9)
                plt.text(c, ymax * 0.95, f"C{i}", color='#F58518',
                         ha='center', va='top', fontsize=9, rotation=90)

        plt.title(f"Distribuição e centroides – {label}")
        plt.xlabel("Valor")
        plt.ylabel("Contagem")
        plt.grid(True, alpha=0.25)

        out_path = os.path.join(out_dir, f"distribuicao_centroides_{label}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        any_plotted = True
        print(f"✓ Salvo: {out_path}")

    if not any_plotted:
        print("Nenhum gráfico de distribuição/centroides foi gerado.")


# ============================================================
#                   DIAGNÓSTICO / MAIN
# ============================================================

def diagnosticar_hierarquia():
    print("\n=== Diagnóstico de hierarquia ===")
    print(f"- INPUTS_DIR: {INPUTS_DIR} -> {'ok' if os.path.isdir(INPUTS_DIR) else 'não encontrado'}")
    print(f"- CENTROIDS_DIR: {CENTROIDS_DIR} -> {'ok' if os.path.isdir(CENTROIDS_DIR) else 'não encontrado'}")
    print(f"- OMP_PATH: {OMP_PATH} -> {'ok' if os.path.exists(OMP_PATH) else 'não encontrado'}")
    print(f"- SEQUENCIAL_PATH: {SEQUENCIAL_PATH} -> {'ok' if os.path.exists(SEQUENCIAL_PATH) else 'não encontrado'}")
    print(f"- CUDA_PATH: {CUDA_PATH} -> {'ok' if os.path.exists(CUDA_PATH) else 'não encontrado'}")
    print(f"- MPI_PATH: {MPI_PATH} -> {'ok' if os.path.exists(MPI_PATH) else 'não encontrado'}")

    inputs = sorted(glob(os.path.join(INPUTS_DIR, "*.*")))
    cents = sorted(glob(os.path.join(CENTROIDS_DIR, "centroids_*.*")))
    sse_serial = sorted(glob(os.path.join("Resultados", "Original", "sse_*.csv")))
    sse_omp = sorted(glob(os.path.join("Resultados", "OpenMP", "sse_*.csv")))
    sse_cuda = sorted(glob(os.path.join("Resultados", "CUDA", "sse_*.csv")))
    sse_mpi = sorted(glob(os.path.join("Resultados", "MPI", "sse_*.csv")))

    def _preview(lst, maxn=8):
        if not lst:
            return "nenhum"
        head = [os.path.relpath(p) for p in lst[:maxn]]
        more = f" (+{len(lst) - maxn})" if len(lst) > maxn else ""
        return ", ".join(head) + more

    print(f"- Arquivos em {INPUTS_DIR}: {_preview(inputs)}")
    print(f"- Centróides em {CENTROIDS_DIR} (centroids_*): {_preview(cents)}")
    print(f"- SSE serial (Resultados/Original): {_preview(sse_serial)}")
    print(f"- SSE OpenMP (Resultados/OpenMP): {_preview(sse_omp)}")
    print(f"- SSE CUDA (Resultados/CUDA): {_preview(sse_cuda)}")
    print(f"- SSE MPI (Resultados/MPI): {_preview(sse_mpi)}")
    print("=== Fim do diagnóstico ===\n")


def main():
    ensure_dir(OUT_DIR)
    diagnosticar_hierarquia()

    # 1) Performance OpenMP (CPU)
    have_results = os.path.exists(OMP_PATH) and os.path.exists(SEQUENCIAL_PATH)
    if have_results:
        print("\n" + "=" * 60)
        print("Calculando Speedup (Serial vs OpenMP)...")
        print("=" * 60)
        add_speedup(SEQUENCIAL_PATH, OMP_PATH)

        df_omp = pd.read_csv(OMP_PATH)
        df_seq = pd.read_csv(SEQUENCIAL_PATH)

        print("\n" + "=" * 60)
        print("Gerando gráficos de performance (OpenMP)...")
        print("=" * 60 + "\n")

        plot_tempo_vs_threads(df_omp, OUT_DIR)
        plot_speedup_vs_threads(df_omp, df_seq, OUT_DIR)
    else:
        print("Resultados OpenMP não encontrados (Resultados/resultados.csv ou Resultados/sequencial.csv). "
              "Pulando gráficos de performance OpenMP.")

    # 1b) Performance CUDA (GPU)
    have_cuda = os.path.exists(CUDA_PATH) and os.path.exists(SEQUENCIAL_PATH)
    if have_cuda:
        print("\n" + "=" * 60)
        print("Calculando Speedup (Serial vs CUDA)...")
        print("=" * 60)
        add_speedup_cuda(SEQUENCIAL_PATH, CUDA_PATH)

        df_cuda = pd.read_csv(CUDA_PATH)
        df_seq_cuda = pd.read_csv(SEQUENCIAL_PATH)

        print("\n" + "=" * 60)
        print("Gerando gráficos de performance (CUDA)...")
        print("=" * 60 + "\n")

        plot_tempo_vs_blocksize(df_cuda, OUT_DIR)
        plot_speedup_vs_blocksize(df_cuda, df_seq_cuda, OUT_DIR)
    else:
        print("Resultados CUDA não encontrados (Resultados/cuda.csv). Pulando gráficos de performance CUDA.")

    # 1c) Performance MPI (CPU distribuído)
    have_mpi = os.path.exists(MPI_PATH) and os.path.exists(SEQUENCIAL_PATH)
    if have_mpi:
        print("\n" + "=" * 60)
        print("Calculando Speedup (Serial vs MPI)...")
        print("=" * 60)
        add_speedup_mpi(SEQUENCIAL_PATH, MPI_PATH)

        df_mpi = pd.read_csv(MPI_PATH)
        df_seq_mpi = pd.read_csv(SEQUENCIAL_PATH)

        print("\n" + "=" * 60)
        print("Gerando gráficos de performance (MPI)...")
        print("=" * 60 + "\n")

        plot_tempo_vs_procs_mpi(df_mpi, OUT_DIR)
        plot_speedup_vs_procs_mpi(df_mpi, df_seq_mpi, OUT_DIR)
    else:
        print("Resultados MPI não encontrados (Resultados/mpi.csv). Pulando gráficos de performance MPI.")

    # 2) SSE por iteração (validação Serial vs OpenMP)
    plot_sse_validacao(labels=('pequeno', 'medio', 'grande'),
                       threads_ref=4,
                       out_dir=OUT_DIR)

    # 2b) SSE por iteração (Serial vs CUDA) – BlockSize de referência
    plot_sse_validacao_cuda(labels=('pequeno', 'medio', 'grande'),
                            blocksize_ref=256,
                            out_dir=OUT_DIR)

    # 2c) SSE por iteração (Serial vs MPI) – número de processos de referência
    plot_sse_mpi(labels=('pequeno', 'medio', 'grande'),
                 procs_ref=8,
                 out_dir=OUT_DIR)

    # 3) Distribuição (1D) + centróides finais
    print("\nGerando distribuições com centróides a partir de Geracao_dados/inputs e Resultados/Original/centroids_* ...\n")
    plot_distribuicoes_inputs_e_centroides(inputs_dir=INPUTS_DIR,
                                           centroids_dir=CENTROIDS_DIR,
                                           out_dir=OUT_DIR)

    # [NOVO] Exporta relatórios não-imagem (CSV/HTML/MD) com todos os resultados
    print("\n" + "=" * 60)
    print("Gerando relatórios não-imagem (CSV/HTML/MD)...")
    print("=" * 60)
    export_readable_reports()

    print("\n" + "=" * 60)
    print(f"✓ Todos os gráficos disponíveis foram gerados em {OUT_DIR}")
    print(f"✓ Relatórios não-imagem foram gerados em {REPORTS_DIR}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
