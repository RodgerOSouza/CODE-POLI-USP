import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import pandas as pd

# -----------------------------------------------------------------
# 1. BLOCO DE IMPORTAÇÃO
# -----------------------------------------------------------------
try:
    # MUDANÇA CRUCIAL: Usamos All2AllIterative para permitir bloqueio (upstream)
    from py_wake.wind_farm_models import All2AllIterative

    # Importamos os modelos de DEFICIT
    from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
    from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit  # Bloqueio

    from py_wake.flow_map import HorizontalGrid
    from py_wake.wind_turbines import WindTurbine
    from py_wake.site import UniformWeibullSite
    from py_wake.superposition_models import LinearSum
    from py_wake.wind_turbines.power_ct_functions import CubePowerSimpleCt

except ImportError as e:
    print("\n--- OCORREU UM ERRO REAL DE IMPORTAÇÃO ---")
    print(f"\nMENSAGEM ORIGINAL DO PYTHON:\n{e}")
    exit()

# -----------------------------------------------------------------
# 2. FUNÇÕES DE LEITURA (Mantidas iguais)
# -----------------------------------------------------------------


def get_wind_data_from_wrg(file_path):
    col_names = [
        'X coord', 'Y coord', 'Z coord', 'Height', 'Weibull A', 'Weibull k',
        'Power Density', 'Number of Sectors'
    ]
    for i in range(1, 17):
        col_names.append(f'f_sec_{i}')
        col_names.append(f'A_sec_{i}')
        col_names.append(f'k_sec_{i}')

    try:
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=1,
                         names=col_names, engine='python')
        df = df.iloc[1:].reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        f_cols = [f'f_sec_{i}' for i in range(1, 17)]
        A_cols = [f'A_sec_{i}' for i in range(1, 17)]
        k_cols = [f'k_sec_{i}' for i in range(1, 17)]

        mean_f = df[f_cols].mean()
        mean_A = df[A_cols].mean()
        mean_k = df[k_cols].mean()

        f_list = (mean_f / mean_f.sum()).values
        A_list = (mean_A / 10).values
        k_list = (mean_k / 100).values

        return f_list, A_list, k_list

    except Exception as e:
        print(f"Erro no WRG: {e}")
        return None, None, None


def get_turbine_layout(kml_file_path):
    x_coords, y_coords = [], []
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    if not os.path.exists(kml_file_path):
        print(f"Erro: KML não encontrado: {kml_file_path}")
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
    except Exception as e:
        print(f"Erro no KML: {e}")
        return None, None


# -----------------------------------------------------------------
# 3. CONFIGURAÇÃO E CARREGAMENTO
# -----------------------------------------------------------------
print("\n--- Iniciando Análise de Bloqueio (Portfólio Completo) ---")

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, ".")

kml_file = os.path.join(base_dir, "doc.kml")
wrg_file = os.path.join(base_dir, "vortex.105.0.wrg")

print(f"Procurando arquivos em: {base_dir}")

x, y = get_turbine_layout(kml_file)
f_norm, A_list, k_list = get_wind_data_from_wrg(wrg_file)

if x is None or f_norm is None:
    print("ABORTANDO: Falha ao carregar arquivos.")
    exit()

site = UniformWeibullSite(p_wd=f_norm, a=A_list, k=k_list, ti=0.10)

my_turbine_type = WindTurbine(
    name='V150-4.2MW (Genérico)',
    diameter=150.0,
    hub_height=105.0,
    powerCtFunction=CubePowerSimpleCt(power_rated=4.2, power_unit='MW', ct=0.8)
)

# -----------------------------------------------------------------
# 4. DEFINIÇÃO DOS MODELOS (USANDO All2AllIterative)
# -----------------------------------------------------------------
print("Configurando modelos de simulação (All2AllIterative)...")

# MODELO A: APENAS ESTEIRA
# Usamos All2AllIterative para ser consistente, mas sem bloqueio
model_wake_only = All2AllIterative(
    site=site,
    windTurbines=my_turbine_type,
    wake_deficitModel=BastankhahGaussianDeficit(k=0.075),
    superpositionModel=LinearSum(),
    blockage_deficitModel=None  # Sem bloqueio
)

# MODELO B: ESTEIRA + BLOQUEIO
# Agora sim o blockage_deficitModel será aceito e usado!
model_full = All2AllIterative(
    site=site,
    windTurbines=my_turbine_type,
    wake_deficitModel=BastankhahGaussianDeficit(k=0.075),
    superpositionModel=LinearSum(),
    blockage_deficitModel=SelfSimilarityDeficit()  # Com bloqueio
)

# -----------------------------------------------------------------
# 5. CÁLCULO QUANTITATIVO (AEP)
# -----------------------------------------------------------------
print("\nCalculando AEP comparativo (isso pode levar alguns segundos)...")

# AEP Total
aep_wake = model_wake_only(x, y).aep().sum()
aep_full = model_full(x, y).aep().sum()

perda_bloqueio_gwh = aep_wake - aep_full
perda_percentual = (perda_bloqueio_gwh / aep_wake) * 100

print("\n==================================================")
print("       RESULTADOS QUANTITATIVOS DE BLOQUEIO       ")
print("==================================================")
print(f"AEP (Modelo Padrão - Só Esteira):  {aep_wake:.2f} GWh/ano")
print(f"AEP (Modelo Completo - Com Bloqueio): {aep_full:.2f} GWh/ano")
print(f"--------------------------------------------------")
print(f"PERDA LÍQUIDA POR BLOQUEIO:        {perda_bloqueio_gwh:.2f} GWh/ano")
print(f"IMPACTO PERCENTUAL NO PARQUE:      {perda_percentual:.3f}%")
print("==================================================")

# -----------------------------------------------------------------
# 6. ANÁLISE VISUAL (MAPAS 2D)
# -----------------------------------------------------------------
print("\nGerando mapas de fluxo para visualização...")

wd_sim = [120]
ws_sim = [10]

grid = HorizontalGrid(
    x=np.linspace(min(x)-2000, max(x)+2000, 200),
    y=np.linspace(min(y)-2000, max(y)+2000, 200)
)

sim_wake = model_wake_only(x, y, wd=wd_sim, ws=ws_sim)
sim_full = model_full(x, y, wd=wd_sim, ws=ws_sim)

fm_wake = sim_wake.flow_map(grid)
fm_full = sim_full.flow_map(grid)

delta_map = fm_wake.WS_eff - fm_full.WS_eff

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

fm_wake.plot_wake_map(ax=axes[0])
axes[0].set_title("A) Modelo Padrão (Apenas Esteira)")

fm_full.plot_wake_map(ax=axes[1])
axes[1].set_title("B) Modelo Completo (Esteira + Bloqueio)")
axes[1].set_yticks([])

im = axes[2].contourf(grid.x, grid.y, delta_map.squeeze(),
                      levels=20, cmap='Reds')
plt.colorbar(im, ax=axes[2], label="Déficit de Velocidade (m/s)")
axes[2].scatter(x, y, c='black', s=5, label='Turbinas')
axes[2].set_title(
    f"C) Impacto Isolado do Bloqueio (A menos B)\nVento {ws_sim[0]} m/s a {wd_sim[0]}°")
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig("analise_bloqueio_mapas.png")
print("Gráfico salvo: analise_bloqueio_mapas.png")
print("\n--- Análise Concluída ---")
