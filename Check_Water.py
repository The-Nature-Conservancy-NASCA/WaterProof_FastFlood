import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.windows
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from pathlib import Path
import re
import shutil
import subprocess

_RC = {
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.which': 'both',
    'grid.alpha': 0.22,
    'grid.linestyle': '--',
    'grid.linewidth': 0.55,
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#f8f9fa',
    'axes.labelpad': 8,
}

def CommandFastFlood(NameCommand,
                     FastFloodPath, key=None, customurl=None,
                     BasinBox=None, DemResolution=None,
                     DEM_Path=None, Manning_Path=None, Inf_Path=None, IDF_Path=None, Fac_CC_Path=None,
                     D_DS=None, D_DS_CC=None, D=None, P=None, Q=None, SSP=None, TR=None,
                     H_Path=None, Q_Path=None, V_Path=None, nOut=None, InfOut=None, PathShp=None, Verbose=True,
                     ChW_Path=None,ChD_Path=None,TS_Path=None,Channel=None,LULC_Path=None, ocean=None, Rain=None,
                     FactorCal=False):
    """
    Construye el comando para ejecutar FastFlood desde línea de comandos.

    Esta función arma dinámicamente una lista de argumentos para ejecutar tareas específicas del modelo FastFlood,
    como descarga de insumos, ejecución de simulaciones y configuración de escenarios climáticos e hidráulicos.

    Parámetros
    ----------
    NameCommand : str
       Nombre del comando a ejecutar. Opciones principales: 'Run', 'Download'
    FastFloodPath : str
       Ruta al ejecutable de FastFlood.
    customurl : str
       URL personalizada para descarga de insumos (por ejemplo, AWS o servidor propio).
    BasinBox : list[float]
       Bounding box de la cuenca en coordenadas planas: [minx, maxy, maxx, miny].
    DemResolution : int
       Resolución del DEM en metros (20, 40, 150, 300 o 600).
    DEM_Path : str
       Ruta al archivo DEM de entrada o salida, según el comando (entrada o salida).
    Manning_Path : str
       Ruta al archivo raster de coeficientes de Manning (entrada o salida).
    Inf_Path : str
       Ruta al archivo de infiltración (entrada o salida).
    IDF_Path : str
       Ruta al archivo CSV de curvas IDF (salida).
    Fac_CC_Path : str
       Ruta al archivo CSV con factores de cambio climático (salida).
    D_DS : int
       Clima histórico - Duración del evento para el diseño de tormenta en horas (3, 6, 12, 24, 48, 72, 120, 240).
    D_DS_CC : int
       Cambio climático - Duración del evento para el diseño de tormenta en días (1, 3, 7).
    D : float
       Duración de la tormenta (horas). Valor mayor que cero
    P : int
       Año del periodo proyectado para cambio climático (2020–2100).
    Q : int
       Cuantil de precipitación (15, 50, 85).
    SSP : str
       Escenario climático (ssp124, ssp245, ssp460, ssp585).
    TR : int
       Periodo de retorno en años (2, 5, 10, 20, 40, 50, 100, 200, 500, 1000).
    H_Path, Q_Path, V_Path : str
       Rutas de salida para altura de agua (whout), caudal pico (qout) y velocidad pico (vout).
    nOut, InfOut : str
       Rutas de salida para coeficientes de Manning e infiltración.
    PathShp : str
       Ruta a shapefile o GeoJSON con polígonos de adaptación o modificación de parámetros.
    Verbose : bool, default=True
       Si es True, imprime información detallada.
    ChW_Path, ChD_Path : str
       Rutas de salida para ancho y profundidad de canal.
    TS_Path : str
       Ruta de salida del csv con el hidrograma (hydrograph).
    Channel : list[float]
       Parámetros del modelo 1D-2D de canales: [WidthMul, WidthExp, DepthMul, DepthExp, CrossSection, ChannelManning].
    LULC_Path : str
       Ruta al raster de cobertura del suelo (salida).
    ocean : float
       Altura de condición de frontera oceánica (para simulaciones costeras).
    FactorCal : Booleano
       Activar el uso factor de calibración

    Retorna
    -------
    list[str]
       Lista con los argumentos listos para pasar a `subprocess.run()` o comando shell.

    Notas
    -----
    - Si el comando es 'Run', se asume que todos los insumos ya están listos y solo se arma el `-sim`.
    - La función **no ejecuta** el modelo, solo construye el comando.
    """

    # iniciar comando
    Comando = [FastFloodPath]

    if key is not None:
        Comando += ['-key',key]
    if customurl is not None:
        Comando += ['-customurl',customurl]
    if Verbose:
        Comando += ['-verbose']
    if (DEM_Path is not None) and (NameCommand == 'Run'):
        Comando += ['-sim','-dem', DEM_Path]
    if (DEM_Path is not None) and (NameCommand == 'Download') and (BasinBox is not None):
        Comando += ['-d_dem', 'cop30', f'{DemResolution}m',
                    f'{BasinBox[0]}', f'{BasinBox[1]}', f'{BasinBox[2]}',f'{BasinBox[3]}',
                    '-dout', DEM_Path]
    if (NameCommand =='Download') and (BasinBox is None):
        Comando += ['-dem', DEM_Path]
    if ((Manning_Path is not None) or (LULC_Path is not None)) and (NameCommand =='Download'):
        Comando += ['-d_lu']
    if (Manning_Path is not None) and (NameCommand =='Download') and (DEM_Path is not None):
        Comando += ['-manout', Manning_Path]
    if (LULC_Path is not None) and (NameCommand == 'Download') and (DEM_Path is not None):
        Comando += ['-luout', LULC_Path]
    if (Inf_Path is not None) and (NameCommand == 'Download') and (DEM_Path is not None):
        Comando += ['-d_inf','-ksatout', Inf_Path]
    if (IDF_Path is not None) and (NameCommand == 'Download') and (DEM_Path is not None):
        Comando += ['-idfout', IDF_Path]
    if (Fac_CC_Path is not None) and (NameCommand == 'Download') and (DEM_Path is not None):
        Comando += ['-climout', Fac_CC_Path]
    if (Manning_Path is not None) and (NameCommand == 'Run'):
        Comando += ['-man', Manning_Path]
    if (Inf_Path is not None) and (NameCommand == 'Run'):
        Comando += ['-inf', Inf_Path]
    if (TR is not None) and (D_DS is not None) and (Rain is None):
        Comando += ['-designstorm', f'{TR}', f'{D_DS}']
    if (Rain is not None):
        Comando += ['-rain', f'{Rain}']
    if (D is not None):
        Comando += ['-dur', f'{D}']
    if (P is not None) and (Q is not None) and (TR is not None) and (D_DS_CC is not None) and (SSP != "Historic"):
        Comando += ['-climate', SSP, f'{P}', f'{Q}', f'{TR}', f'{D_DS_CC}']
    if PathShp is not None:
        Comando += ['-adaptation', PathShp]
    if Channel is not None:
        Comando += ['-channel', f'{Channel[0]}', f'{Channel[1]}', f'{Channel[2]}',f'{Channel[3]}',f'{Channel[4]}']
    if ocean is not None:
        Comando += ['-ocean',f'{ocean}']
    if H_Path is not None:
        Comando += ['-whout', H_Path]
    if Q_Path is not None:
        Comando += ['-qout', Q_Path]
    if V_Path is not None:
        Comando += ['-vout', V_Path]
    if nOut is not None:
        Comando += ['-manout', nOut]
    if InfOut is not None:
        Comando += ['-ksatout', InfOut]
    if ChW_Path is not None:
        Comando += ['-chwout', ChW_Path]
    if ChD_Path is not None:
        Comando += ['-chhout', ChD_Path]
    if TS_Path is not None:
        Comando += ['-hydrograph', TS_Path]
    if FactorCal:
        Comando += ['-d_cal']

    return Comando


def sherman(t, TR, K, m, a, n):
    """
    Fórmula de Sherman: i = K · T^m / (t + a)^n

    Parameters
    ----------
    t        : duración (h), escalar o array numpy
    TR       : período de retorno (años), escalar o array numpy
    K, m, a, n : parámetros calibrados

    Returns
    -------
    i : intensidad (mm/h)
    """
    return K * TR**m / (t + a)**n


def sherman_fit(data, output_dir='IDF', tabla=None, tabla_path=None):
    """
    Ajusta la fórmula de Sherman a datos IDF y exporta resultados.

    Fórmula: i = K * T^m / (t + a)^n

    Parameters
    ----------
    data : str, Path, o DataFrame
        Ruta al CSV IDF, o DataFrame con índice = duraciones (h) y
        columnas = TRs (años), ambos numéricos.
    output_dir : str or Path
        Carpeta de salida. Se crea si no existe. Default: 'IDF'.
    tabla : dict, optional
        {'D': [duraciones_h], 'TR': [periodos_años]} para generar tabla IDF nueva.
    tabla_path : str or Path, optional
        Ruta CSV para guardar la tabla generada. Requiere tabla.

    Returns
    -------
    dict
        Parametros {'K', 'm', 'a', 'n'} y metricas {'NSE', 'RMSE', 'R2'}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.index = df.index.astype(float)
        df.columns = df.columns.astype(float)
        csv_path = None
    else:
        csv_path = Path(data)
        df = pd.read_csv(csv_path, index_col=0)
        df.index = df.index.astype(float)
        df.columns = df.columns.astype(float)
        shutil.copy(csv_path, output_dir / f'backup_{csv_path.name}')

    durations = df.index.values
    trs = df.columns.values

    T_grid, t_grid = np.meshgrid(trs, durations)
    i_obs = df.values.flatten()
    t_arr = t_grid.flatten()
    T_arr = T_grid.flatten()

    def _model(X, K, m, a, n):
        t, T = X
        return sherman(t, T, K, m, a, n)

    popt, _ = curve_fit(
        _model, (t_arr, T_arr), i_obs,
        p0=[500, 0.2, 10, 0.7],
        bounds=([0, 0, 0, 0], [1e6, 2, 200, 2]),
        maxfev=20000
    )
    K, m, a, n = popt

    i_sim = _model((t_arr, T_arr), *popt)
    ss_res = np.sum((i_obs - i_sim) ** 2)
    ss_tot = np.sum((i_obs - i_obs.mean()) ** 2)
    nse = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((i_obs - i_sim) ** 2)))
    r2 = float(np.corrcoef(i_obs, i_sim)[0, 1] ** 2)

    pd.DataFrame({
        'parametro': ['K', 'm', 'a', 'n'],
        'valor': [K, m, a, n]
    }).to_csv(output_dir / 'parametros.csv', index=False)

    pd.DataFrame({
        'metrica': ['NSE', 'RMSE_mm_h', 'R2'],
        'valor': [nse, rmse, r2],
        'descripcion': [
            'Nash-Sutcliffe Efficiency',
            'Root Mean Square Error (mm/h)',
            'Pearson R² (obs vs sim)'
        ]
    }).to_csv(output_dir / 'metricas.csv', index=False)

    _plot_obs_vs_sim(df, durations, trs, _model, popt, output_dir)
    _plot_obs_vs_sim_linear(df, durations, trs, _model, popt, output_dir)

    if tabla is not None:
        new_D = np.array(tabla['D'], dtype=float)
        new_TR = np.array(tabla['TR'], dtype=float)
        TR_g, D_g = np.meshgrid(new_TR, new_D)
        i_new = _model((D_g.flatten(), TR_g.flatten()), *popt).reshape(len(new_D), len(new_TR))
        new_df = pd.DataFrame(i_new, index=new_D, columns=new_TR)
        new_df.index.name = 'duracion'

        if tabla_path is not None:
            p = Path(tabla_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            new_df.to_csv(p)

        _plot_new_idf(new_D, new_TR, K, m, a, n, output_dir)
        _plot_new_idf_linear(new_D, new_TR, K, m, a, n, output_dir)

    return {'K': K, 'm': m, 'a': a, 'n': n, 'NSE': nse, 'RMSE': rmse, 'R2': r2}


def _plot_obs_vs_sim(df, durations, trs, model, popt, output_dir):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(trs)))
        t_dense = np.geomspace(durations.min(), durations.max(), 400)

        for TR, color in zip(trs, colors):
            ax.scatter(durations, df[TR].values, color=color,
                       s=55, zorder=5, edgecolors='white', linewidths=0.6)
            i_line = model((t_dense, np.full_like(t_dense, TR)), *popt)
            ax.plot(t_dense, i_line, color=color, linewidth=2, label=f'{int(TR)}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel('Duración (h)', fontsize=13)
        ax.set_ylabel('Intensidad (mm/h)', fontsize=13)
        ax.set_title('Curvas IDF — Observado vs Simulado (Sherman)',
                     fontsize=15, fontweight='bold', pad=14)

        handles, labels = ax.get_legend_handles_labels()
        phantom = [
            Line2D([0], [0], marker='o', color='#555', linestyle='None',
                   markersize=7, markeredgecolor='white', markeredgewidth=0.6),
            Line2D([0], [0], color='#555', linewidth=2),
        ]
        leg = ax.legend(
            handles=phantom + handles,
            labels=['Observado', 'Simulado'] + labels,
            bbox_to_anchor=(1.02, 1), loc='upper left',
            framealpha=0.93, fontsize=9,
            title='  Símbolo / TR (años)', title_fontsize=10,
        )
        leg.get_frame().set_edgecolor('#d0d0d0')

        fig.tight_layout()
        fig.savefig(output_dir / 'obs_vs_sim.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def _plot_obs_vs_sim_linear(df, durations, trs, model, popt, output_dir):
    K, m, a, n = popt
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(trs)))
        t_dense = np.linspace(durations.min(), durations.max(), 400)

        for TR, color in zip(trs, colors):
            ax.scatter(durations, df[TR].values, color=color,
                       s=55, zorder=5, edgecolors='white', linewidths=0.6)
            i_line = model((t_dense, np.full_like(t_dense, TR)), *popt)
            ax.plot(t_dense, i_line, color=color, linewidth=2, label=f'{int(TR)}')

        ax.set_xlabel('Duración (h)', fontsize=13)
        ax.set_ylabel('Intensidad (mm/h)', fontsize=13)
        ax.set_title('Curvas IDF — Observado vs Simulado (Sherman) — Escala Lineal',
                     fontsize=15, fontweight='bold', pad=14)

        formula = (
            r'$i = \frac{' + f'{K:.2f}' + r'\cdot T^{' + f'{m:.4f}' + r'}}'
            r'{(t + ' + f'{a:.2f}' + r')^{' + f'{n:.4f}' + r'}}$'
        )
        ax.text(0.97, 0.97, formula, transform=ax.transAxes,
                fontsize=13, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.88, edgecolor='#cccccc', linewidth=1.2))

        handles, labels = ax.get_legend_handles_labels()
        phantom = [
            Line2D([0], [0], marker='o', color='#555', linestyle='None',
                   markersize=7, markeredgecolor='white', markeredgewidth=0.6),
            Line2D([0], [0], color='#555', linewidth=2),
        ]
        leg = ax.legend(
            handles=phantom + handles,
            labels=['Observado', 'Simulado'] + labels,
            bbox_to_anchor=(1.02, 1), loc='upper left',
            framealpha=0.93, fontsize=9,
            title='  Símbolo / TR (años)', title_fontsize=10,
        )
        leg.get_frame().set_edgecolor('#d0d0d0')

        fig.tight_layout()
        fig.savefig(output_dir / 'obs_vs_sim_lineal.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def _plot_new_idf(new_D, new_TR, K, m, a, n, output_dir):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(new_TR)))
        t_dense = np.geomspace(new_D.min(), new_D.max(), 400)

        for TR, color in zip(new_TR, colors):
            i_vals = sherman(t_dense, TR, K, m, a, n)
            ax.plot(t_dense, i_vals, color=color, linewidth=2.2, label=f'{int(TR)}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel('Duración (h)', fontsize=13)
        ax.set_ylabel('Intensidad (mm/h)', fontsize=13)
        ax.set_title('Curvas IDF Generadas — Fórmula de Sherman',
                     fontsize=15, fontweight='bold', pad=14)

        formula = (
            r'$i = \frac{' + f'{K:.2f}' + r'\cdot T^{' + f'{m:.4f}' + r'}}'
            r'{(t + ' + f'{a:.2f}' + r')^{' + f'{n:.4f}' + r'}}$'
        )
        ax.text(0.97, 0.97, formula, transform=ax.transAxes,
                fontsize=13, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.88, edgecolor='#cccccc', linewidth=1.2))

        leg = ax.legend(
            bbox_to_anchor=(1.02, 1), loc='upper left',
            framealpha=0.93, fontsize=9,
            title='TR (años)', title_fontsize=10,
        )
        leg.get_frame().set_edgecolor('#d0d0d0')

        fig.tight_layout()
        fig.savefig(output_dir / 'idf_nuevas.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def _plot_new_idf_linear(new_D, new_TR, K, m, a, n, output_dir):
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(new_TR)))
        t_dense = np.linspace(new_D.min(), new_D.max(), 400)

        for TR, color in zip(new_TR, colors):
            i_vals = sherman(t_dense, TR, K, m, a, n)
            ax.plot(t_dense, i_vals, color=color, linewidth=2.2, label=f'{int(TR)}')

        ax.set_xlabel('Duración (h)', fontsize=13)
        ax.set_ylabel('Intensidad (mm/h)', fontsize=13)
        ax.set_title('Curvas IDF Generadas — Fórmula de Sherman — Escala Lineal',
                     fontsize=15, fontweight='bold', pad=14)

        formula = (
            r'$i = \frac{' + f'{K:.2f}' + r'\cdot T^{' + f'{m:.4f}' + r'}}'
            r'{(t + ' + f'{a:.2f}' + r')^{' + f'{n:.4f}' + r'}}$'
        )
        ax.text(0.97, 0.97, formula, transform=ax.transAxes,
                fontsize=13, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.88, edgecolor='#cccccc', linewidth=1.2))

        leg = ax.legend(
            bbox_to_anchor=(1.02, 1), loc='upper left',
            framealpha=0.93, fontsize=9,
            title='TR (años)', title_fontsize=10,
        )
        leg.get_frame().set_edgecolor('#d0d0d0')

        fig.tight_layout()
        fig.savefig(output_dir / 'idf_nuevas_lineal.png', dpi=150, bbox_inches='tight')
        plt.close(fig)


def calcular_estadisticas_idf_cuenca(carpeta_rasters, shapefile_cuenca, estadistica='mean'):
    """
    Calcula estadísticas de rasters IDF para una cuenca y organiza en formato tabla IDF.

    Parámetros:
    -----------
    carpeta_rasters : str
        Ruta a la carpeta con rasters con nombres tipo: IDF_TR_2_D_240
    shapefile_cuenca : str or dict or list
        Ruta al shapefile de la cuenca, o geometría en formato GeoJSON.
        Acepta: FeatureCollection, Feature, geometría directa, o lista de features
    estadistica : str
        Estadística a calcular: 'mean', 'min', 'max', 'std', 'median', 'sum'

    Retorna:
    --------
    pandas.DataFrame
        DataFrame con duraciones como filas y períodos de retorno como columnas
    """

    # Cargar cuenca
    print("Cargando cuenca...")
    if isinstance(shapefile_cuenca, dict):
        # Si es GeoJSON
        if shapefile_cuenca.get('type') == 'FeatureCollection':
            # FeatureCollection: extraer features
            cuenca = gpd.GeoDataFrame.from_features(shapefile_cuenca['features'], crs='EPSG:4326')
        elif shapefile_cuenca.get('type') == 'Feature':
            # Feature individual
            cuenca = gpd.GeoDataFrame.from_features([shapefile_cuenca], crs='EPSG:4326')
        else:
            # Geometría directa (Polygon, MultiPolygon, etc.)
            cuenca = gpd.GeoDataFrame([{'geometry': shape(shapefile_cuenca)}], crs='EPSG:4326')
    elif isinstance(shapefile_cuenca, list):
        # Lista de features
        cuenca = gpd.GeoDataFrame.from_features(shapefile_cuenca, crs='EPSG:4326')
    else:
        # Si es ruta de archivo
        cuenca = gpd.read_file(shapefile_cuenca)

    if cuenca.crs != 'EPSG:4326':
        cuenca = cuenca.to_crs('EPSG:4326')

    # Obtener archivos raster
    raster_files = glob.glob(os.path.join(carpeta_rasters, "IDF_TR_*_D_*.tif"))
    if not raster_files:
        raster_files = glob.glob(os.path.join(carpeta_rasters, "IDF_TR_*_D_*.tiff"))

    if not raster_files:
        raise ValueError("No se encontraron archivos IDF con el formato esperado")

    print(f"Encontrados {len(raster_files)} archivos IDF")

    # Definir estadísticas
    stats_functions = {
        'mean': np.nanmean,
        'min': np.nanmin,
        'max': np.nanmax,
        'std': np.nanstd,
        'median': np.nanmedian,
        'sum': np.nansum
    }

    if estadistica not in stats_functions:
        raise ValueError(f"Estadística '{estadistica}' no disponible. Opciones: {list(stats_functions.keys())}")

    stat_func = stats_functions[estadistica]

    # Procesar cada raster
    def procesar_raster_idf(raster_path):
        try:
            nombre_archivo = os.path.basename(raster_path)

            # Extraer TR y D del nombre del archivo
            # Formato: IDF_TR_2_D_240.tif
            partes = nombre_archivo.replace('.tif', '').replace('.tiff', '').split('_')
            tr_idx = partes.index('TR') + 1
            d_idx = partes.index('D') + 1

            periodo_retorno = int(partes[tr_idx])
            duracion = int(partes[d_idx])

            with rasterio.open(raster_path) as src:
                # Extraer píxeles usando mask
                masked_data, _ = rasterio.mask.mask(
                    src, cuenca.geometry, crop=True, all_touched=True, filled=False
                )

                # Tomar primera banda si es multiband
                if masked_data.ndim > 2:
                    masked_data = masked_data[0]

                # Obtener píxeles válidos
                if hasattr(masked_data, 'mask'):
                    valid_pixels = masked_data[~masked_data.mask]
                else:
                    valid_pixels = masked_data.flatten()

                valid_pixels = valid_pixels[~np.isnan(valid_pixels)]

                if len(valid_pixels) == 0:
                    valor_estadistica = np.nan
                else:
                    valor_estadistica = stat_func(valid_pixels)

                return {
                    'periodo_retorno': periodo_retorno,
                    'duracion': duracion,
                    'valor': valor_estadistica,
                    'archivo': nombre_archivo
                }

        except Exception as e:
            print(f"Error procesando {raster_path}: {e}")
            return None

    # Procesamiento paralelo
    print("Procesando rasters...")
    resultados = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(procesar_raster_idf, raster_path): raster_path
                   for raster_path in raster_files}

        for i, future in enumerate(as_completed(futures), 1):
            resultado = future.result()
            if resultado is not None:
                resultados.append(resultado)
            print(f"Procesado {i}/{len(raster_files)}")

    if not resultados:
        raise ValueError("No se pudieron procesar archivos")

    # Crear DataFrame temporal
    df_temp = pd.DataFrame(resultados)

    # Crear tabla pivote: duraciones como filas, períodos de retorno como columnas
    df_idf = df_temp.pivot(index='duracion', columns='periodo_retorno', values='valor')

    # Ordenar filas (duraciones) y columnas (períodos de retorno)
    duraciones_ordenadas = [3, 6, 12, 24, 48, 72, 120, 240]
    periodos_ordenados = [2, 5, 10, 20, 40, 50, 100, 200, 500, 1000]

    # Filtrar solo duraciones y períodos que existen en los datos
    duraciones_existentes = [d for d in duraciones_ordenadas if d in df_idf.index]
    periodos_existentes = [p for p in periodos_ordenados if p in df_idf.columns]

    df_idf = df_idf.loc[duraciones_existentes, periodos_existentes]
    
    # Del raster original se tienen las precipitaciones totales por cada duración por TR. 
    # Para generar la intensidades dividimos por la duración
    D = df_idf.index.values.reshape(-1, 1)
    df_idf = df_idf / D
    
    print(f"\nCompletado! Tabla IDF con estadística '{estadistica}'")
    print(f"Duraciones: {duraciones_existentes}")
    print(f"Períodos de retorno: {periodos_existentes}")

    return df_idf

def statistical_zonal_raster(
    raster_path: str,
    stat: str = "mean",
    polygon_path: str = None,
    mask_raster: str = None,
    threshold: float = None,
) -> float | None:
    """
    Calcula una estadística zonal sobre un raster, opcionalmente dentro de una máscara.

    Parámetros
    ----------
    raster_path : str
        Ruta al raster de entrada (.tif).
    stat : str
        Estadística: 'mean', 'sum', 'min', 'max'.
    polygon_path : str | dict | list | GeoDataFrame, opcional
        Máscara por polígono. Acepta: ruta a shapefile/GeoJSON, GeoJSON dict
        (FeatureCollection, Feature o geometría directa), lista de features,
        o GeoDataFrame. Exclusivo con mask_raster.
    mask_raster : str, opcional
        Máscara por raster (píxeles > 0 y finitos). Exclusivo con polygon_path.
    threshold : float, opcional
        Excluye píxeles >= threshold.

    Retorna
    -------
    float | None
        Resultado de la estadística. None si no hay píxeles válidos.
    """
    stat = stat.lower()
    if stat not in ("mean", "sum", "min", "max"):
        raise ValueError(f"stat debe ser 'mean', 'sum', 'min' o 'max'. Recibido: {stat!r}")

    total = 0.0
    count = 0
    running = np.inf if stat == "min" else -np.inf

    def _update(values):
        nonlocal total, count, running
        if values.size == 0:
            return
        if stat == "mean":
            total += float(values.sum())
            count += values.size
        elif stat == "sum":
            total += float(values.sum())
        elif stat == "min":
            v = float(values.min())
            if v < running:
                running = v
        else:
            v = float(values.max())
            if v > running:
                running = v

    if mask_raster is not None:
        with rasterio.open(mask_raster) as msk:
            bounds   = msk.bounds
            msk_data = msk.read(1).astype(np.float32)
            nd_mask  = msk.nodata
            msk_tf   = msk.transform
            msk_crs  = msk.crs

        with rasterio.open(raster_path) as src:
            win    = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                                 transform=src.transform)
            data    = src.read(1, window=win).astype(np.float32)
            inf_tf  = src.window_transform(win)
            inf_nd  = src.nodata
            inf_crs = src.crs

        rows, cols = data.shape
        valid_src = np.ones_like(msk_data, dtype=np.int32)
        if nd_mask is not None:
            valid_src[msk_data == nd_mask] = 0
        valid_src[~np.isfinite(msk_data) | (msk_data <= 0)] = 0
        del msk_data

        ws = np.zeros((rows, cols), dtype=np.int32)
        reproject(source=valid_src, destination=ws,
                  src_transform=msk_tf, src_crs=msk_crs,
                  dst_transform=inf_tf, dst_crs=inf_crs,
                  resampling=Resampling.nearest)
        mask = ws.astype(bool)
        del ws, valid_src

        if inf_nd is not None:
            data[data == np.float32(inf_nd)] = np.nan
        data[~mask] = np.nan
        del mask
        if threshold is not None:
            data[data >= np.float32(threshold)] = np.nan

        _update(data[np.isfinite(data)])
        del data

    elif polygon_path is not None:
        if isinstance(polygon_path, gpd.GeoDataFrame):
            gdf = polygon_path
        elif isinstance(polygon_path, dict):
            if polygon_path.get("type") == "FeatureCollection":
                gdf = gpd.GeoDataFrame.from_features(polygon_path["features"], crs="EPSG:4326")
            elif polygon_path.get("type") == "Feature":
                gdf = gpd.GeoDataFrame.from_features([polygon_path], crs="EPSG:4326")
            else:
                gdf = gpd.GeoDataFrame([{"geometry": shape(polygon_path)}], crs="EPSG:4326")
        elif isinstance(polygon_path, list):
            gdf = gpd.GeoDataFrame.from_features(polygon_path, crs="EPSG:4326")
        else:
            gdf = gpd.read_file(polygon_path)

        with rasterio.open(raster_path) as rst:
            if not isinstance(polygon_path, str) and gdf.crs is not None:
                from pyproj import CRS as ProjCRS
                src = ProjCRS.from_user_input(gdf.crs)
                dst = ProjCRS.from_wkt(rst.crs.to_wkt())
                if not src.equals(dst):
                    gdf = gdf.to_crs(dst)
            polygon = unary_union(gdf.geometry.values)
            geom_list = [polygon.__geo_interface__]
            nodata = rst.nodata
            poly_win_raw = from_bounds(*polygon.bounds, rst.transform)
            raster_win = Window(0, 0, rst.width, rst.height)
            try:
                poly_window = rasterio.windows.intersection(poly_win_raw, raster_win)
            except rasterio.errors.WindowError:
                return None

            for _, blk_window in rst.block_windows(1):
                try:
                    win = rasterio.windows.intersection(blk_window, poly_window)
                except rasterio.errors.WindowError:
                    continue
                if win.width <= 0 or win.height <= 0:
                    continue

                data = rst.read(1, window=win).astype(np.float32)
                win_transform = rst.window_transform(win)
                pmask = geometry_mask(geom_list, out_shape=data.shape,
                                      transform=win_transform, invert=True)
                valid = data[pmask & (data != nodata)] if nodata is not None else data[pmask]
                if threshold is not None:
                    valid = valid[valid < np.float32(threshold)]
                _update(valid)

    else:
        with rasterio.open(raster_path) as rst:
            nodata = rst.nodata
            for _, win in rst.block_windows(1):
                data = rst.read(1, window=win).astype(np.float32)
                valid = data[data != nodata] if nodata is not None else data.ravel()
                valid = valid[np.isfinite(valid)]
                if threshold is not None:
                    valid = valid[valid < np.float32(threshold)]
                _update(valid)

    if stat == "mean":
        return float(total / count) if count > 0 else None
    if stat == "sum":
        return float(total)
    return float(running) if np.isfinite(running) else None


def Check_Available_water(PathDataBase: str, PathWatershed: str, Inf_Path: str, 
                          DEM_Path: str, Manning_Path: str, FastFloodPath: str, customurl: str):
    """
    Determina duraciones de tormenta donde la intensidad supera la infiltración mínima en la cuenca.

    Parámetros
    ----------
    PathDataBase : str
        Ruta a la base de datos. Debe contener subcarpeta 03-IDF con rasters IDF_TR_*_D_*.tif.
    PathWatershed : str
        Ruta al shapefile de la cuenca.
    Inf_Path : str
        Ruta al raster de tasa de infiltración (mismas unidades que intensidad IDF: mm/h).

    Retorna
    -------
    numpy.ndarray
        Duraciones (horas) donde intensidad TR=10 > infiltración mínima. Array vacío si ninguna.
    """
    # Estimar IDF de los rasters de base de datos
    carpeta_tiles = os.path.join(PathDataBase, "03-IDF")
    df_idf = calcular_estadisticas_idf_cuenca(carpeta_tiles, PathWatershed, 'mean')    

    # Estimación de parámetros de la ecuación de sherman para IDF
    # Guarda la IDF nueva en la carpeta de descarga de los datos
    PathNewIDF = os.path.join(os.path.dirname(Inf_Path),'IDF.csv')
    result = sherman_fit(
        data=df_idf,
        output_dir=os.path.join(os.path.dirname(Inf_Path),'IDF'),
        tabla={
            'D': [3, 4, 5, 6, 8, 10, 12, 18, 24, 48],
            'TR': [2, 5, 10, 20, 40, 50, 100, 200, 500, 1000]
        },
        tabla_path=PathNewIDF
    )
    for k, v in result.items():
        print(f'{k}: {v:.6f}')
    
    IDF_Table = pd.read_csv(PathNewIDF, index_col=0)
    
    # 2. Convertimos el índice a numérico
    IDF_Table.index = pd.to_numeric(IDF_Table.index)

    # 3. Convertimos los nombres de las columnas a numérico
    IDF_Table.columns = pd.to_numeric(IDF_Table.columns)
    
    # 4. ejecutar 
    ValueStorm = 150 # mm/hr
    Comando = CommandFastFlood("Run",FastFloodPath, customurl=customurl,
                                     DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                     D=3, Rain=ValueStorm, FactorCal=True)   

    env = os.environ.copy()
    for key in ('PROJ_LIB', 'PROJ_DATA', 'PROJ_NETWORK'):
        env.pop(key, None)

    proceso = subprocess.Popen(
        Comando,
        stdout=subprocess.PIPE,  # Captura la salida estándar
        stderr=subprocess.PIPE,  # Captura los errores estándar
        universal_newlines=True,  # Forzar la salida como texto
        env=env,
    )

    # Usar communicate() sin timeout
    salida, error = proceso.communicate()
    print(salida.strip(), flush=True)
    print("STDERR:", error.strip())

    match = re.search(
        r"rain total:\s*([\d.]+).*?runoff total:\s*([\d.]+).*?infil total:\s*([\d.]+)",
        salida, re.DOTALL | re.IGNORECASE
    )
    if match:
        rain   = float(match.group(1))
        runoff = float(match.group(2))
        infil  = float(match.group(3))

    # Se estima el mínimo pero se multiplica por FactorCorrect considerando el factor multiplicador
    # de calibración.
    infil_Raw = statistical_zonal_raster(Inf_Path, stat="mean",threshold=ValueStorm)
    FactorCorrect = infil/infil_Raw
    print(f'Correct Factor Infiltration: {FactorCorrect}')
    PathFactorCorrect = os.path.join(os.path.dirname(Inf_Path), 'FactorCorrect.csv')
    pd.DataFrame({'FactorCorrect': [FactorCorrect]}).to_csv(PathFactorCorrect, index=False)
    Min_Inf = statistical_zonal_raster(Inf_Path, stat="min", polygon_path=PathWatershed)*FactorCorrect

    # .values convierte a bool numpy — indexar D (numpy) con Series pandas produce resultados incorrectos
    # TR para umbral, por defecto 10 Años
    TR_i = 10
    mask = (IDF_Table.loc[:, TR_i] > Min_Inf).values
    
    # Duraciones para filtrar
    D = IDF_Table.index.values.reshape(-1, 1)
    
    return D.flatten()[mask]
