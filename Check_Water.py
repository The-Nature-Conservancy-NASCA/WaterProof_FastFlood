import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import rasterio.windows
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import shape
from shapely.ops import unary_union


def calcular_estadisticas_idf_cuenca(carpeta_rasters, shapefile_cuenca, estadistica='mean'):
    """
    Calcula estadísticas de rasters IDF para una cuenca y organiza en formato tabla IDF.

    Parámetros:
    -----------
    carpeta_rasters : str
        Ruta a la carpeta con rasters con nombres tipo: IDF_TR_2_D_240
    shapefile_cuenca : str
        Ruta al shapefile de la cuenca
    estadistica : str
        Estadística a calcular: 'mean', 'min', 'max', 'std', 'median', 'sum'

    Retorna:
    --------
    pandas.DataFrame
        DataFrame con duraciones como filas y períodos de retorno como columnas
    """

    # Cargar cuenca
    print("Cargando cuenca...")
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

    print(f"\nCompletado! Tabla IDF con estadística '{estadistica}'")
    print(f"Duraciones: {duraciones_existentes}")
    print(f"Períodos de retorno: {periodos_existentes}")

    return df_idf


def min_raster_en_poligono(raster_path: str, polygon_path: str) -> float | None:
    """
    Calcula el valor mínimo del raster dentro de un polígono sin cargar el raster completo.

    Itera sobre bloques nativos del raster alineados al bbox del polígono,
    aplicando máscara pixel-level por bloque. O(área_polígono), no O(raster_completo).

    Parámetros
    ----------
    raster_path : str
        Ruta al raster (.tif). Debe estar en el mismo CRS que el polígono.
    polygon_path : str
        Ruta al shapefile o GeoJSON del polígono.

    Retorna
    -------
    float | None
        Mínimo de píxeles válidos dentro del polígono. None si no hay píxeles válidos.
    """
    with fiona.open(polygon_path) as poly_src:
        polygon = unary_union([shape(f["geometry"]) for f in poly_src])

    geom_list = [polygon.__geo_interface__]

    with rasterio.open(raster_path) as rst:
        nodata = rst.nodata

        # Recortar al bbox del polígono — evita iterar bloques fuera de la zona de interés
        poly_win_raw = from_bounds(*polygon.bounds, rst.transform)
        raster_win = Window(0, 0, rst.width, rst.height)
        try:
            poly_window = rasterio.windows.intersection(poly_win_raw, raster_win)
        except rasterio.errors.WindowError:
            return None  # polígono completamente fuera del raster

        global_min = np.inf

        for _, blk_window in rst.block_windows(1):
            try:
                win = rasterio.windows.intersection(blk_window, poly_window)
            except rasterio.errors.WindowError:
                continue

            if win.width <= 0 or win.height <= 0:
                continue

            data = rst.read(1, window=win)
            win_transform = rst.window_transform(win)

            pmask = geometry_mask(
                geom_list,
                out_shape=data.shape,
                transform=win_transform,
                invert=True,
            )

            valid = data[pmask & (data != nodata)] if nodata is not None else data[pmask]

            if valid.size > 0:
                block_min = float(valid.min())
                if block_min < global_min:
                    global_min = block_min

        return float(global_min) if np.isfinite(global_min) else None


def Check_Available_water(PathDataBase: str, PathWatershed: str, PathInfiltrationRaster: str):
    """
    Determina duraciones de tormenta donde la intensidad supera la infiltración mínima en la cuenca.

    Parámetros
    ----------
    PathDataBase : str
        Ruta a la base de datos. Debe contener subcarpeta 03-IDF con rasters IDF_TR_*_D_*.tif.
    PathWatershed : str
        Ruta al shapefile de la cuenca.
    PathInfiltrationRaster : str
        Ruta al raster de tasa de infiltración (mismas unidades que intensidad IDF: mm/h).

    Retorna
    -------
    numpy.ndarray
        Duraciones (horas) donde intensidad TR=2 > infiltración mínima. Array vacío si ninguna.
    """
    carpeta_tiles = os.path.join(PathDataBase, "03-IDF")
    df_idf = calcular_estadisticas_idf_cuenca(carpeta_tiles, PathWatershed, 'mean')

    # Derivar D del índice real — D hardcodeado fallaría si faltan duraciones en los datos
    D = df_idf.index.values.reshape(-1, 1)
    IDF_Table = df_idf / D

    # Se estima el mínimo pero se multiplica por 1.5 considerando el caso más critico de factor multiplicador
    # de calibración
    Min_Inf = min_raster_en_poligono(PathInfiltrationRaster, PathWatershed)*1.5

    # .values convierte a bool numpy — indexar D (numpy) con Series pandas produce resultados incorrectos
    mask = (IDF_Table.loc[:, 2] > Min_Inf).values
    return D.flatten()[mask]


# python Check_Water.py <PathDataBase> <PathWatershed> <PathInfiltrationRaster>
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Uso: python Check_Water.py <PathDataBase> <PathWatershed> <PathInfiltrationRaster>")
        sys.exit(1)

    PathDataBase         = sys.argv[1]
    PathWatershed        = sys.argv[2]
    PathInfiltrationRaster = sys.argv[3]

    duraciones = Check_Available_water(PathDataBase, PathWatershed, PathInfiltrationRaster)
    print(f"Duraciones donde intensidad TR=2 > infiltración mínima (h): {duraciones}")
