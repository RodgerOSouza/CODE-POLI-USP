import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import pandas as pd  # <-- Adicionado para ler o .wrg

# -----------------------------------------------------------------
# BLOCO DE IMPORTAÇÃO (Correto para sua versão)
# -----------------------------------------------------------------
try:
    from py_wake.deficit_models.noj import NOJ
    from py_wake.deficit_models.gaussian import BastankhahGaussian
    from py_wake.flow_map import HorizontalGrid
    from py_wake.wind_turbines import WindTurbine
    from py_wake.site import UniformWeibullSite
    from py_wake.superposition_models import LinearSum
    from py_wake.wind_turbines.power_ct_functions import PowerCtFunctionList, CubePowerSimpleCt

except ImportError as e:
    print("\n--- OCORREU UM ERRO REAL DE IMPORTAÇÃO ---")
    print(f"\nMENSAGEM ORIGINAL DO PYTHON:\n{e}")
    print("\nUma das importações falhou. Verifique se o ambiente 'pywake' está correto.")
    print("Verifique se 'pandas' está instalado (pip install pandas)")
    exit()

# --- NOVA FUNÇÃO PARA LER OS DADOS DE VENTO DO ARQUIVO .WRG ---


def get_wind_data_from_wrg(file_path):
    """
    Lê o arquivo .wrg, calcula a média dos dados de todos os pontos
    e retorna as listas de frequência, Weibull A e Weibull k.
    """
    # Nomes das colunas com base na estrutura .wrg e no seu .csv
    col_names = [
        'X coord', 'Y coord', 'Z coord', 'Height', 'Weibull A', 'Weibull k',
        'Power Density', 'Number of Sectors'
    ]
    for i in range(1, 17):
        col_names.append(f'f_sec_{i}')
        col_names.append(f'A_sec_{i}')
        col_names.append(f'k_sec_{i}')

    try:
        # 1. Lê o .wrg, usando '\s+' (qualquer espaço) como delimitador
        #    skiprows=1 pula a primeira linha do .wrg (ex: "158 342...")
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=1,
                         names=col_names, engine='python')

        # 2. Pula a primeira linha de dados (ex: "...1.0 1.50..."),
        #    que você identificou como dados de cabeçalho inválidos.
        df = df.iloc[1:].reset_index(drop=True)

        # Converte colunas para numérico, tratando erros
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()  # Remove linhas que falharam na conversão

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
        # Frequência: Normaliza a soma para 1
        f_list = (mean_f / mean_f.sum()).values
        # Weibull A: Divide por 10 (ex: 99 = 9.9 m/s)
        A_list = (mean_A / 10).values
        # Weibull k: Divide por 100 (ex: 469 = 4.69)
        k_list = (mean_k / 100).values

        print("Dados de vento médios extraídos do .wrg.")
        return f_list, A_list, k_list

    except FileNotFoundError:
        print(
            f"Erro Crítico: Arquivo de Vento (.wrg) NÃO encontrado em: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo .wrg: {e}")
        return None, None, None

# -----------------------------------------------------------------
# PASSO 1: EXTRAIR LAYOUT DO KML
# (Nenhuma mudança)
# -----------------------------------------------------------------


def get_turbine_layout(kml_file_path):
    x_coords, y_coords = [], []
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    if not os.path.exists(kml_file_path):
        print(f"Erro Crítico: Arquivo KML NÃO encontrado: {kml_file_path}")
        return None, None
    try:
        tree = ET.parse(kml_file_path)
        root = tree.getroot()
        for placemark in root.findall('.//kml:Placemark', ns):
            x_val, y_val = None, None
            for data in placemark.findall('.//kml:Data', ns):
                if data.get('name') == 'X':
                    x_val = float(data.find('kml:value', ns).text)
                if data.get('name') == 'Y':
                    y_val = float(data.find('kml:value', ns).text)
            if x_val is not None and y_val is not None:
                x_coords.append(x_val)
                y_coords.append(y_val)
        print(f"Layout KML carregado: {len(x_coords)} turbinas encontradas.")
        return np.array(x_coords), np.array(y_coords)
    except ET.ParseError as e:
        print(f"Erro ao ler o KML: {e}")
        return None, None


# -----------------------------------------------------------------
# PASSO 2: DEFINIR TURBINA (COM BASE NO KML)
# (Nenhuma mudança)
# -----------------------------------------------------------------
hub_height = 105.0  # (ALT_TORRE)
diameter = 150.0    # (DIAM_ROTOR)
power_mw = 4.2      # (POT_MW)

generic_power_ct_func = CubePowerSimpleCt(
    power_rated=power_mw,
    power_unit='MW',
    ct=0.8
)

my_turbine_type = WindTurbine(
    name='V150-4.2MW (Genérico)',
    diameter=diameter,
    hub_height=hub_height,
    powerCtFunction=generic_power_ct_func
)
print(
    f"Turbina definida: {my_turbine_type.name()} com {my_turbine_type.hub_height()}m de altura (criada genericamente).")

# -----------------------------------------------------------------
# --- PASSO 3 MODIFICADO: DEFINIR O SITE (LENDO O .WRG) ---
# -----------------------------------------------------------------
print("\n--- PASSO 3: Processando dados de vento ---")
wrg_file_path = r"DIRETÓRIO DO .WRG"
f_norm, A_list, k_list = get_wind_data_from_wrg(wrg_file_path)

# TI não está no .wrg, então definimos um padrão
ti_val = 0.10
site = None

if f_norm is not None:
    site = UniformWeibullSite(
        p_wd=f_norm,
        a=A_list,
        k=k_list,
        ti=ti_val
    )
    print("Site (recurso eólico) definido com sucesso (dados médios do .wrg).")

# -----------------------------------------------------------------
# PASSO 4: SIMULAÇÃO E CÁLCULO DE AEP
# (Verifica se KML e Site foram carregados)
# -----------------------------------------------------------------
print("\n--- PASSO 4: Iniciando simulação ---")
kml_file = r"DIRETÓRIO .KML"
x, y = get_turbine_layout(kml_file)

# Verifica se ambos os arquivos de entrada foram lidos com sucesso
if (x is not None and len(x) > 0) and (site is not None):

    # Modelo 1: NOJ (Jensen)
    model_noj = NOJ(
        site=site,
        windTurbines=my_turbine_type,
        k=0.075,
        superpositionModel=LinearSum()
    )

    # Modelo 2: Bastankhah-Gaussian
    model_gaussian = BastankhahGaussian(
        site=site,
        windTurbines=my_turbine_type,
        k=0.075,
        superpositionModel=LinearSum()
    )

    # 3. Calcular o AEP (GWh/ano)
    print("Calculando AEP para o modelo NOJ (Jensen)...")
    aep_noj = model_noj(x, y).aep().sum()

    print("Calculando AEP para o modelo Bastankhah-Gaussian...")
    aep_gaussian = model_gaussian(x, y).aep().sum()

    print("\n--- RESULTADOS DA GERAÇÃO (AEP) ---")
    print(f"Modelo NOJ (Jensen):             {aep_noj:.2f} GWh/ano")
    print(f"Modelo Bastankhah-Gaussian:    {aep_gaussian:.2f} GWh/ano")
    print("AVISO: Valores de AEP calculados com a turbina V150 (4.2MW) GENÉRICA.")

    # -------------------------------------------------------------
    # EXTRA: Salvar os gráficos
    # -------------------------------------------------------------
    print("\nGerando visualizações (salvando arquivos PNG)...")

    plt.figure(figsize=(10, 8))
    my_turbine_type.plot(x, y)
    plt.title("Layout das Turbinas (Extraído do KML)")
    plt.xlabel("Coordenada X (UTM)")
    plt.ylabel("Coordenada Y (UTM)")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig("layout_parque.png")
    print(f"Salvo: layout_parque.png")

    plt.figure(figsize=(8, 8))
    site.plot_wd_distribution(n_wd=12, ws_bins=[0, 5, 10, 15, 20, 25])
    plt.title("Rosa dos Ventos (Dados do .wrg)")
    plt.tight_layout()
    plt.savefig("rosa_dos_ventos_wrg.png")
    print(f"Salvo: rosa_dos_ventos_wrg.png")

    print("Simulando mapa de esteira...")
    sim_res_gaussian = model_gaussian(x, y, wd=[120], ws=[10])

    plt.figure(figsize=(12, 8))

    grid = HorizontalGrid(x=np.linspace(min(x)-500, max(x)+500, 100),
                          y=np.linspace(min(y)-500, max(y)+500, 100))

    flow_map = sim_res_gaussian.flow_map(grid=grid)
    flow_map.plot_wake_map()
    plt.title("Mapa de Velocidade do Vento (Gaussian) - Direção 120°, 10 m/s")
    plt.tight_layout()
    plt.savefig("mapa_esteira_120deg.png")
    print(f"Salvo: mapa_esteira_120deg.png")

    print("\n--- Análise Concluída ---")

else:
    print("\nA simulação não pôde ser executada. Verifique os erros acima (KML ou .wrg).")
