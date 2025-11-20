import pandas as pd
import numpy as np

# -----------------------------------------------------------------
# CONFIGURAÇÃO
# -----------------------------------------------------------------
# Caminho para o seu arquivo .wrg original
file_path = r"DIRETÓRIO .WRG"

# Colunas (com base nos seus títulos do CSV e na estrutura do .wrg)
col_names = [
    'X coord', 'Y coord', 'Z coord', 'Height', 'Weibull A', 'Weibull k',
    'Power Density', 'Number of Sectors'
]
# Adiciona as colunas dos 16 setores (Freq, A, k)
for i in range(1, 17):
    col_names.append(f'f_sec_{i}')
    col_names.append(f'A_sec_{i}')
    col_names.append(f'k_sec_{i}')

# -----------------------------------------------------------------
# LEITURA E PROCESSAMENTO
# -----------------------------------------------------------------
try:
    # 1. Lê o .wrg, tratando espaços como delimitadores
    #    skiprows=1 pula a primeira linha do .wrg (ex: "158 342...")
    df = pd.read_csv(file_path, delim_whitespace=True,
                     skiprows=1, names=col_names)

    # 2. Pula a primeira linha de dados (ex: "...1.0 1.50..."),
    #    que você suspeita ser um dado inválido.
    df = df.iloc[1:].reset_index(drop=True)

    print(
        f"Arquivo .wrg lido com sucesso. Processando {len(df)} pontos de grid...")

    # 3. Define as colunas que queremos calcular a média
    f_cols = [f'f_sec_{i}' for i in range(1, 17)]
    A_cols = [f'A_sec_{i}' for i in range(1, 17)]
    k_cols = [f'k_sec_{i}' for i in range(1, 17)]

    # 4. Calcula a média de cada coluna de setor
    mean_f = df[f_cols].mean()
    mean_A = df[A_cols].mean()
    mean_k = df[k_cols].mean()

    # 5. Aplica a escala (dados do .wrg vêm multiplicados)
    # Frequência: O valor é * 1000 (ex: 709). Devemos normalizar para somar 1.
    # Weibull A: O valor é * 10 (ex: 99 = 9.9 m/s)
    # Weibull k: O valor é * 100 (ex: 469 = 4.69)

    f_list = (mean_f / mean_f.sum()).values
    A_list = (mean_A / 10).values
    k_list = (mean_k / 100).values

    # -----------------------------------------------------------------
    # SAÍDA (PRONTA PARA COPIAR)
    # -----------------------------------------------------------------
    print("\n--- COPIE E COLE ISTO NO SEU CÓDIGO 'calcular_aep.py' ---")

    # Imprime as listas formatadas
    print(
        f"f_norm = {np.array2string(f_list, separator=', ', max_line_width=100)}")
    print(
        f"A_list = {np.array2string(A_list, separator=', ', max_line_width=100, precision=2)}")
    print(
        f"k_list = {np.array2string(k_list, separator=', ', max_line_width=100, precision=2)}")
    print(f"ti_val = 0.10 # ATENÇÃO: TI não está no .wrg, usando 0.10 como padrão.")
    print("------------------------------------------------------------")

except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em: {file_path}")
except Exception as e:
    print(f"Ocorreu um erro ao processar o arquivo: {e}")
    print("Verifique se o pandas está instalado (pip install pandas)")
