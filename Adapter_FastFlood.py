# -*- coding: utf-8 -*-
'''
  Nature For Water Facility - The Nature Conservancy
  -------------------------------------------------------------------------
  Python 3.11
  -------------------------------------------------------------------------
                            BASIC INFORMATION
  -------------------------------------------------------------------------
  Author        : Jonathan Nogales Pimentel
  Email         : jonathan.nogales@tnc.org
  Date          : January, 2025

  -------------------------------------------------------------------------
                            DESCRIPTION
  -------------------------------------------------------------------------

'''

# === Librerías estándar de Python ===
import datetime
import glob
import json
import logging
import os
import psutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional, Dict, Set, Tuple

# === Librerías científicas ===
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import interp1d

# === GIS y manejo de geometrías ===
import fiona
import geopandas as gpd
from osgeo import gdal, ogr
from pyproj import CRS, Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

# === Rasterio y operaciones ráster ===
import rasterio
import rasterio.merge
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import xy
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

import os
import time
import glob
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.errors import RasterioIOError
import rasterio.mask
from rasterio.merge import merge
import fiona
from shapely.geometry import shape, box, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer
from rasterio.features import geometry_mask

import os
import glob
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import rasterio
import rasterio.mask
import rasterio.windows
from rasterio.features import geometry_mask
import geopandas as gpd
import pandas as pd
import numpy as np
from osgeo import gdal
import os
import time

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LOGGER
# ============================================================================

def setup_logger(name='FastFlood', log_file=None, level=logging.INFO):
    """
    Configura un logger para tracking de operaciones de FastFlood.

    Parameters
    ----------
    name : str
        Nombre del logger (default: 'FastFlood')
    log_file : str, optional
        Ruta del archivo de log. Si es None, solo se muestra en consola.
    level : logging level
        Nivel de logging (default: logging.INFO)

    Returns
    -------
    logging.Logger
        Logger configurado
    """
    logger = logging.getLogger(name)

    # Evitar duplicar handlers si ya existe
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formato detallado con timestamp, nivel, función y mensaje
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para archivo si se especificó
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_memory_usage():
    """
    Obtiene el uso de memoria del proceso actual.

    Returns
    -------
    dict
        Diccionario con memoria RSS, VMS y porcentaje de uso
    """
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()

        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size (RAM real)
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': mem_percent
        }
    except Exception as e:
        return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'error': str(e)}

def log_memory(logger, message="", level=logging.INFO):
    """
    Log con información de uso de memoria actual.

    Parameters
    ----------
    logger : logging.Logger
        Logger a utilizar
    message : str
        Mensaje adicional a incluir
    level : logging level
        Nivel de log
    """
    mem = get_memory_usage()
    if 'error' in mem:
        logger.log(level, f"{message} [Memory tracking error: {mem['error']}]")
    else:
        logger.log(
            level,
            f"{message} [RAM: {mem['rss_mb']:.1f} MB | {mem['percent']:.1f}%]"
        )

def log_array_info(logger, array, name="Array", level=logging.DEBUG):
    """
    Log con información detallada de un numpy array.

    Parameters
    ----------
    logger : logging.Logger
        Logger a utilizar
    array : numpy.ndarray
        Array a analizar
    name : str
        Nombre descriptivo del array
    level : logging level
        Nivel de log
    """
    if array is None:
        logger.log(level, f"{name}: None")
        return

    size_mb = array.nbytes / 1024 / 1024
    logger.log(
        level,
        f"{name}: shape={array.shape}, dtype={array.dtype}, size={size_mb:.2f} MB"
    )

# Crear logger global del módulo
_module_logger = None

def get_module_logger(log_file=None):
    """
    Obtiene o crea el logger global del módulo.

    Parameters
    ----------
    log_file : str, optional
        Ruta del archivo de log

    Returns
    -------
    logging.Logger
        Logger del módulo
    """
    global _module_logger
    if _module_logger is None:
        _module_logger = setup_logger('FastFlood', log_file=log_file)
    return _module_logger

# === Otros (no utilizados explícitamente o dudosos) ===
# from numpy.core._multiarray_umath import ndarray  # innecesario, ya viene con numpy
# from fontTools.misc.cython import returns  # no parece ser usado en tu código

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

def CreateFolders(base_path):
    '''
    Crea las carpetas requeridas para el análisis utilizando una ruta base especificada.

    :param base_path: Ruta base donde se crearán las carpetas.
    '''
    try:
        # Rutas de las carpetas
        folders = [
            os.path.join(base_path, 'in', '06-FLOOD', 'Raster'),
            os.path.join(base_path, 'in', '06-FLOOD', 'Damages'),
            os.path.join(base_path, 'in', '06-FLOOD', 'Shp'),
            os.path.join(base_path, 'out', '06-FLOOD', 'Damages'),
            os.path.join(base_path, 'out', '06-FLOOD', 'Discharge'),
            os.path.join(base_path, 'out', '06-FLOOD', 'Flood'),
            os.path.join(base_path, 'out', '06-FLOOD', 'Velocity'),
            os.path.join(base_path, 'out', '06-FLOOD', 'Other'),
            os.path.join(base_path, 'out', '06-FLOOD', 'Tmp')
        ]

        # Crear cada carpeta si no existe
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f'Carpeta creada: {folder}')

        print('Todas las carpetas han sido creadas con éxito.')
    except Exception as e:
        print(f'Error al crear las carpetas: {e}')

def ExeFastFlood(Comando, log=None):

    # Ejecutar el comando
    proceso = subprocess.Popen(
        Comando,
        stdout=subprocess.PIPE,  # Captura la salida estándar
        stderr=subprocess.PIPE,  # Captura los errores estándar
        universal_newlines=True,  # Forzar la salida como texto
    )

    # Usar communicate() sin timeout
    salida, error = proceso.communicate()

    # Escribir la salida en el log
    if salida:
        print(salida.strip(), flush=True)  # Mostrar la salida en la consola
        if log is not None:
            log.write(f'[OUTPUT] {salida}')  # Guardar en el log

    # Escribir los errores en el log
    if error:
        print(error.strip(), file=sys.stderr, flush=True)  # Mostrar el error en la consola
        if log is not None:
            log.write(f'[ERROR] {error}\n\n')  # Guardar en el log

def ClicRasterWithBasin(ruta_raster, ruta_shapefile, salida='raster_recortado.tif'):
    """
    Recorta un ráster usando un shapefile como máscara.

    Parámetros:
        ruta_raster (str): Ruta al archivo ráster.
        ruta_shapefile (str): Ruta al shapefile con la geometría de recorte.
        salida (str): Ruta al archivo de salida recortado.
    """
    # Cargar geometría del shapefile
    gdf = gpd.read_file(ruta_shapefile)

    with rasterio.open(ruta_raster) as src:
        # Reproyectar el shapefile si el CRS es distinto
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        geoms = gdf.geometry.values
        geoms = [geom.__geo_interface__ for geom in geoms]

        # Recorte con máscara
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

        # Actualizar metadatos
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "LZW",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "predictor": 2 if 'float' in out_image.dtype.name else 1,
            "BIGTIFF": "IF_SAFER"
        })

        # Guardar salida
        with rasterio.open(salida, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"✔️ Ráster recortado guardado en: {salida}")

def CheckPixelDepth(ruta_raster1, ruta_raster2, salida='raster_recortado.tif', Con=True):
    """
    Compara dos rásteres píxel a píxel. Si raster2 > raster1, reemplaza ese valor por el de raster1.
    Guarda el resultado con compresión LZW, tipo de dato ajustado automáticamente y bloques optimizados.

    Parámetros:
        ruta_raster1 (str): Ruta al primer ráster (valor límite superior).
        ruta_raster2 (str): Ruta al segundo ráster (a corregir si excede).
        salida (str): Ruta del archivo de salida corregido.
    """

    with rasterio.open(ruta_raster1) as src1, rasterio.open(ruta_raster2) as src2:
        # Validación de dimensiones y CRS
        assert src1.shape == src2.shape, "Los rásteres deben tener la misma dimensión"
        assert src1.crs == src2.crs, "Los CRS deben coincidir"

        # Leer directamente (sin VRT ya que están alineados)
        data1 = src1.read(1)
        data2 = src2.read(1)

        # Aplicar corrección
        if Con:
            resultado = np.where(data2 > data1, data1, data2)
        else:
            resultado = np.where(data2 < data1, data1, data2)

        # Determinar tipo de dato mínimo necesario
        dtype_resultado = resultado.dtype.name

        # Metadatos de salida
        perfil = src2.profile.copy()
        perfil.update({
            'dtype': dtype_resultado,
            'compress': 'LZW',
            'predictor': 2 if np.issubdtype(resultado.dtype, np.floating) else 1,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'BIGTIFF': 'IF_SAFER'
        })

        # Guardar ráster corregido
        with rasterio.open(salida, 'w', **perfil) as dst:
            dst.write(resultado, 1)

    print(f"✔️ Raster corregido guardado como: {salida} ({dtype_resultado}, LZW)")


def CheckVelocity(ruta_raster1, salida='V_out.tif', max_velocity=15.0):
    """
    Limita valores de velocidad a un máximo especificado.

    Valores mayores al límite se reemplazan por el valor máximo permitido.
    Útil para eliminar velocidades físicamente improbables en modelos hidráulicos.

    Parámetros:
        ruta_raster1 (str): Ruta al raster de velocidad de entrada.
        salida (str): Ruta del archivo de salida corregido.
        max_velocity (float): Velocidad máxima permitida en m/s (default: 15.0).
    """
    with rasterio.open(ruta_raster1) as src1:
        # Leer directamente
        data1 = src1.read(1)

        # Contar píxeles que exceden el límite
        nodata = src1.nodata
        if nodata is not None:
            valid_mask = data1 != nodata
            excede = np.sum((data1 > max_velocity) & valid_mask)
        else:
            excede = np.sum(data1 > max_velocity)

        # Aplicar límite (preservando nodata)
        resultado = np.where(data1 > max_velocity, max_velocity, data1)

        # Determinar tipo de dato
        dtype_resultado = resultado.dtype.name

        # Metadatos de salida
        perfil = src1.profile.copy()
        perfil.update({
            'dtype': dtype_resultado,
            'compress': 'LZW',
            'predictor': 2 if np.issubdtype(resultado.dtype, np.floating) else 1,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'BIGTIFF': 'IF_SAFER'
        })

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(salida), exist_ok=True)

        # Guardar ráster corregido
        with rasterio.open(salida, 'w', **perfil) as dst:
            dst.write(resultado, 1)

    print(f"✔️ Raster de velocidad corregido: {salida}")
    print(f"   - Tipo de dato: {dtype_resultado}, Compresión: LZW")
    print(f"   - Píxeles limitados a {max_velocity} m/s: {excede}")


def CheckPixelDepth_V2(ruta_raster1, ruta_raster2, ruta_raster3,
                       salida_current, salida_bau, salida_nbs):
    """
    Corrige tres rásteres de profundidad (Current, BaU, NbS) aplicando reglas
    de coherencia para asegurar que Current ≤ BaU ≤ NbS donde corresponda.

    Reglas aplicadas:
    - Regla 3: Si Current > BaU > NbS → Current = BaU
    - Regla 4: Si Current > NbS > BaU → BaU = Current
    - Regla 5: Si NbS > Current > BaU → BaU = Current, NbS = Current
    - Regla 6: Si NbS > BaU > Current → NbS = Current

    Parámetros:
        ruta_raster1 (str): Ruta al raster Current.
        ruta_raster2 (str): Ruta al raster BaU.
        ruta_raster3 (str): Ruta al raster NbS.
        salida_current (str): Ruta de salida para Current corregido.
        salida_bau (str): Ruta de salida para BaU corregido.
        salida_nbs (str): Ruta de salida para NbS corregido.
    """

    with rasterio.open(ruta_raster1) as src1, \
            rasterio.open(ruta_raster2) as src2, \
            rasterio.open(ruta_raster3) as src3:
        # Validación
        assert src1.shape == src2.shape == src3.shape, "Los rásteres deben tener la misma dimensión"
        assert src1.crs == src2.crs == src3.crs, "Los CRS deben coincidir"

        # Leer directamente
        Current = src1.read(1).astype(np.float32)
        BaU     = src2.read(1).astype(np.float32)
        NbS     = src3.read(1).astype(np.float32)

        # FILTRADO DE VALORES ATÍPICOS (percentil 99)
        # Calcular percentil 99 solo con valores > 0 (excluir zonas secas)
        p99_current = np.percentile(Current[Current > 0], 99) if np.any(Current > 0) else np.inf
        p99_bau     = np.percentile(BaU[BaU > 0], 99) if np.any(BaU > 0) else np.inf
        p99_nbs     = np.percentile(NbS[NbS > 0], 99) if np.any(NbS > 0) else np.inf

        print(f"Percentiles 99 calculados:")
        print(f"  - Current: {p99_current:.4f}")
        print(f"  - BaU:     {p99_bau:.4f}")
        print(f"  - NbS:     {p99_nbs:.4f}")

        # Reemplazar valores atípicos con el percentil 99
        outliers_current    = np.sum(Current > p99_current)
        outliers_bau        = np.sum(BaU > p99_bau)
        outliers_nbs        = np.sum(NbS > p99_nbs)

        Current[Current > p99_current] = p99_current
        BaU[BaU > p99_bau] = p99_bau
        NbS[NbS > p99_nbs] = p99_nbs

        print(f"\nValores atípicos filtrados:")
        print(f"  - Current: {outliers_current} píxeles")
        print(f"  - BaU:     {outliers_bau} píxeles")
        print(f"  - NbS:     {outliers_nbs} píxeles")

        # APLICAR REGLAS DE COHERENCIA
        # Regla 3 - Current > BaU > NbS
        id = (Current > BaU) & (BaU > NbS)
        Current[id] = BaU[id]

        # Regla 4 - Current > NbS > BaU
        id = (Current > NbS) & (NbS > BaU)
        BaU[id] = Current[id]
        NbS[id] = Current[id]

        # Regla 5 - NbS > Current > BaU
        id = (NbS > Current) & (Current > BaU)
        BaU[id] = Current[id]
        NbS[id] = Current[id]

        # Regla 6 - NbS > BaU > Current
        id = (NbS > BaU) & (BaU > Current)
        NbS[id] = Current[id]

        # Determinar tipo de dato
        dtype_resultado = Current.dtype.name

        # Perfil base
        perfil_base = {
            'dtype': dtype_resultado,
            'compress': 'LZW',
            'predictor': 2,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
            'BIGTIFF': 'IF_SAFER'
        }

        # Guardar Current corregido
        perfil_current = src1.profile.copy()
        perfil_current.update(perfil_base)
        with rasterio.open(salida_current, 'w', **perfil_current) as dst:
            dst.write(Current, 1)

        # Guardar BaU corregido
        perfil_bau = src2.profile.copy()
        perfil_bau.update(perfil_base)
        with rasterio.open(salida_bau, 'w', **perfil_bau) as dst:
            dst.write(BaU, 1)

        # Guardar NbS corregido
        perfil_nbs = src3.profile.copy()
        perfil_nbs.update(perfil_base)
        with rasterio.open(salida_nbs, 'w', **perfil_nbs) as dst:
            dst.write(NbS, 1)

    print(f"✔️ Rasters corregidos guardados:")
    print(f"   - Current: {salida_current}")
    print(f"   - BaU: {salida_bau}")
    print(f"   - NbS: {salida_nbs}")


def Get_Basin_bbox(shp_path):
    '''
    Toma un shapefile en coordenadas geográficas, extrae el bounding box,
    lo transforma a la proyección pseudo-Mercator (EPSG:3857) y devuelve
    las coordenadas transformadas como [minx, maxy, maxx, miny].

    :param shp_path: Ruta al archivo shapefile.
    :return: Lista con las coordenadas transformadas [minx, maxy, maxx, miny].
    '''
    # Abrir el shapefile con ogr
    datasource = ogr.Open(shp_path)
    if datasource is None:
        raise ValueError(f"No se pudo abrir el shapefile en la ruta: {shp_path}")

    # Obtener la capa (layer) del shapefile
    layer = datasource.GetLayer()

    # Obtener la extensión (bounding box) de la capa
    minx, maxx, miny, maxy = layer.GetExtent()

    # Obtener el CRS de la capa
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref is None:
        raise ValueError('El shapefile no tiene un sistema de coordenadas definido.')

    # Crear un transformador para convertir a EPSG:3857
    transformer = Transformer.from_crs(spatial_ref.ExportToWkt(), CRS.from_epsg(3857), always_xy=True)

    # Transformar las coordenadas a EPSG:3857
    minx, miny = transformer.transform(minx, miny)
    maxx, maxy = transformer.transform(maxx, maxy)

    # Devolver las coordenadas transformadas
    return [minx, miny, maxx, maxy]

def DownloadInputs(FastFloodPath, Basin_shp_BoundingBox, DemResolution, DEM_Path, key=None,
                   Manning_Path=None, LULC_Path=None, Inf_Path=None, IDF_Path=None, Fac_CC_Path=None,
                   log=None,Buffer_km=1, customurl=None):
    """
    Descarga y guarda los insumos necesarios para ejecutar el modelo FastFlood.

    Esta función permite automatizar la descarga del DEM, parámetros hidráulicos, cobertura del suelo, 
    infiltración y curvas IDF, utilizando el ejecutable de FastFlood.

    Parameters
    ----------
    FastFloodPath : str
        Ruta local del ejecutable de FastFlood.
    Basin_shp_BoundingBox : list or str
        Ruta del shapefile de la cuenca o
        Coordenadas del bounding box de la cuenca en proyección pseudo-Mercator (EPSG:3857) (e.g., [xmin, ymax, xmax, ymin]).
    DemResolution : int
        Resolución del DEM a descargar. Valores válidos: [20, 40, 150, 300, 600] (en metros).
    DEM_Path : str
        Ruta donde se guardará el DEM descargado.
    Manning_Path : str
        Ruta donde se guardará el raster de coeficientes de Manning.
    LULC_Path : str
        Ruta donde se guardará el raster de cobertura del suelo (LULC).
    Inf_Path : str
        Ruta donde se guardará el raster de infiltración.
    IDF_Path : str
        Ruta donde se guardará el archivo CSV de curvas Intensidad-Duración-Frecuencia (IDF).
    Fac_CC_Path : str
        Ruta donde se guardará el archivo CSV con factores de cambio climático.
    log : file object, optional
        Objeto de archivo abierto para registrar el log del proceso (modo escritura).
    Buffer_km : float, optional
        Tamaño del buffer (en kilómetros) que se añade al bounding box para la descarga de los insumos. 
        Por defecto es 1 km.
    Status_DEM : bool, optional
        Si `True`, se descargará el DEM. Si `False`, se omite. Por defecto es `True`.
    Status_Inf : bool, optional
        Si `True`, se descargará el raster de infiltración. Por defecto es `True`.
    Status_n : bool, optional
        Si `True`, se descargará el raster de n-Manning. Por defecto es `True`.
    Status_IDF : bool, optional
        Si `True`, se descargará el archivo de curvas IDF. Por defecto es `True`.
    customurl : str, optional
        URL personalizada (por ejemplo, una ruta AWS S3 o endpoint específico) para descargar los datos. 

    Returns
    -------
    None
        La función no retorna nada. Guarda los archivos directamente en las rutas indicadas.
    """

    if isinstance(Basin_shp_BoundingBox, list):
        BasinBox = Basin_shp_BoundingBox
        if len(BasinBox) != 4:
            raise ValueError(f"the list must contain four elements.")
    elif isinstance(Basin_shp_BoundingBox, str):
        # Extraer Boundary del shapefile de la cuenca
        BasinBox = Get_Basin_bbox(Basin_shp_BoundingBox)

    # ------------------------------------------------------------------------------------------------------------------
    # Descargar DEM
    # ------------------------------------------------------------------------------------------------------------------
    # Se estima el buffer para el dominio de descarga de la información
    Buffer_m    = Buffer_km*1000
    # xmin
    BasinBox[0] = BasinBox[0] - Buffer_m
    # ymin
    BasinBox[1] = BasinBox[1] - Buffer_m
    # xmax
    BasinBox[2] = BasinBox[2] + Buffer_m
    # ymax
    BasinBox[3] = BasinBox[3] + Buffer_m

    # Orden FastFlood Xmin Ymax Xmax Ymin
    BasinBox = [BasinBox[0], BasinBox[3], BasinBox[2], BasinBox[1]]

    if log is not None:
        # Agregar información al log
        log.write("# ---------------------------------------------------------------------------------------------------\n")
        log.write("# Boundary\n")
        log.write("# ---------------------------------------------------------------------------------------------------\n")
        log.write(f'ulx: {BasinBox[0]} uly: {BasinBox[1]} brx: {BasinBox[2]} bry: {BasinBox[3]}\n\n')

        log.write("# ---------------------------------------------------------------------------------------------------\n")
        log.write("# Download FastFlood Inputs\n")
        log.write("# ---------------------------------------------------------------------------------------------------\n")
        log.write("\n\n")

    if DEM_Path is not None:
        # Construcción del comando de descarga del DEM
        Comando = CommandFastFlood('Download',
                                   FastFloodPath, key, customurl,
                                   BasinBox=BasinBox,
                                   DemResolution=DemResolution,
                                   DEM_Path=DEM_Path)

        # Ejecutar comando
        if log is not None:
            ExeFastFlood(Comando, log)
        else:
            ExeFastFlood(Comando)

        # ------------------------------------------------------------------------------------------------------------------
        # Descargar n-Manning, Infiltración
        # ------------------------------------------------------------------------------------------------------------------
        if (Manning_Path is not None) or (LULC_Path is not None) or (Inf_Path is not None) or (IDF_Path is not None) or (Fac_CC_Path is not None):
            # Construcción del comando de descarga del Manning, Infiltración, IDF y Factor Cambio Climático
            Comando = CommandFastFlood('Download',
                                       FastFloodPath, key, customurl,
                                       DEM_Path=DEM_Path,
                                       Manning_Path=Manning_Path,
                                       LULC_Path=LULC_Path,
                                       Inf_Path=Inf_Path,
                                       IDF_Path=IDF_Path,
                                       Fac_CC_Path=Fac_CC_Path)

            # Ejecutar comando
            if log is not None:
                ExeFastFlood(Comando, log)
            else:
                ExeFastFlood(Comando)


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

def RunScenarios(ProjectPath, FastFloodPath, D, P, Q, SSP, TR,
                 DEM_Path, Manning_Path, Inf_Path, BasinPath, D_DS=None, D_DS_CC=None,
                 log=None, customurl=None, Channel=None, BoundaryCondition=None, IDF_Table=None, StatusExe=True):


    # Residensial
    Results = {'H': {'Currect':[],'BaU':[],'NbS':[]}, 'V': {'Currect':[],'BaU':[],'NbS':[]}, 'Q': {'Currect':[],'BaU':[],'NbS':[]}}

    # ------------------------------------------------------------------------------------------------------------------
    # Ejecutar FastFlood - Current
    # ------------------------------------------------------------------------------------------------------------------
    H_Path_Tmp = ProjectPath + f'/out/06-FLOOD/Tmp/H.tif'
    V_Path_Tmp = ProjectPath + f'/out/06-FLOOD/Tmp/V.tif'
    Q_Path_Tmp = ProjectPath + f'/out/06-FLOOD/Tmp/Q.tif'

    raster_paths_H = {}
    raster_paths_V = {}
    raster_paths_Q = {}
    for TR_i in TR:
        H_Path      = ProjectPath + f'/out/06-FLOOD/Flood/Flood_Current_TR-{TR_i}.tif'
        V_Path      = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_Current_TR-{TR_i}.tif'
        Q_Path      = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_Current_TR-{TR_i}.tif'
        TS_Q_Path   = ProjectPath + f'/out/06-FLOOD/Discharge/TS_Q_Current_TR-{TR_i}.csv'

        # Crear lista de rasters
        raster_paths_H[TR_i] = H_Path
        raster_paths_V[TR_i] = V_Path
        raster_paths_Q[TR_i] = Q_Path

        if SSP == "Historic":
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: Current | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            if IDF_Table is not None:
                ValueStorm = IDF_Table.loc[D_DS, TR_i]
                Comando = CommandFastFlood("Run",
                                             FastFloodPath, customurl=customurl,
                                             DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                             SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Rain=ValueStorm,Channel=Channel, ocean=BoundaryCondition,
                                             H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path, FactorCal=True)
            else:
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Channel=Channel, ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path,
                                           FactorCal=True)


        else:
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: Current | Climate: Scenario: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            if IDF_Table is not None:
                ValueStorm = IDF_Table.loc[D_DS, TR_i]
                Comando = CommandFastFlood("Run",
                                             FastFloodPath, customurl=customurl,
                                             DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                             SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Rain=ValueStorm, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel, ocean=BoundaryCondition,
                                             H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path, FactorCal=True)
            else:
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel,
                                           ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path,
                                           FactorCal=True)

        #'''
        # Ejecutar comando
        if StatusExe:
            ExeFastFlood(Comando,log)

            # Recortar el raster al shapefile de la cuenca
            ClicRasterWithBasin(H_Path_Tmp, BasinPath, salida=H_Path)
            ClicRasterWithBasin(V_Path_Tmp, BasinPath, salida=V_Path)
            ClicRasterWithBasin(Q_Path_Tmp, BasinPath, salida=Q_Path)

            # Filtro de Velocidades
            CheckVelocity(V_Path, V_Path)
        #'''

    # Guardar Listado de Raster de Profundidad
    Results['H']['Current'] = raster_paths_H
    Results['V']['Current'] = raster_paths_V
    Results['Q']['Current'] = raster_paths_Q

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - BaU
    # ------------------------------------------------------------------------------------------------------------------
    # n-Manning modificado por la opción de adaptación - BaU
    Manning_Path    = ProjectPath + '/in/06-FLOOD/Raster/Manning_BaU.tif'
    # Infiltración modificada por la opción de adaptación - BaU
    Inf_Path        = ProjectPath + '/in/06-FLOOD/Raster/Infiltration_BaU.tif'

    raster_paths_H = {}
    raster_paths_V = {}
    raster_paths_Q = {}

    for TR_i in TR:
        H_Path = ProjectPath + f'/out/06-FLOOD/Flood/Flood_BaU_TR-{TR_i}.tif'
        V_Path = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_BaU_TR-{TR_i}.tif'
        Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_BaU_TR-{TR_i}.tif'
        TS_Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/TS_Q_BaU_TR-{TR_i}.csv'

        # Crear lista de rasters
        raster_paths_H[TR_i] = H_Path
        raster_paths_V[TR_i] = V_Path
        raster_paths_Q[TR_i] = Q_Path

        if SSP == "Historic":
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: BaU | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            if IDF_Table is not None:
                ValueStorm = IDF_Table.loc[D_DS, TR_i]
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Rain=ValueStorm, Channel=Channel, ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path, FactorCal=True)
            else:
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Channel=Channel, ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path,
                                           FactorCal=True)

        else:
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: Bau | Climate : {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")
            if IDF_Table is not None:
                ValueStorm = IDF_Table.loc[D_DS, TR_i]
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS,Rain=ValueStorm, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel, ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path, FactorCal=True)
            else:
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel,
                                           ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path,
                                           FactorCal=True)


        #'''
        # Ejecutar comando
        if StatusExe:
            ExeFastFlood(Comando,log)

            # Recortar el raster al shapefile de la cuenca
            ClicRasterWithBasin(H_Path_Tmp, BasinPath, salida=H_Path)
            ClicRasterWithBasin(V_Path_Tmp, BasinPath, salida=V_Path)
            ClicRasterWithBasin(Q_Path_Tmp, BasinPath, salida=Q_Path)

            # Filtro de Velocidades
            CheckVelocity(V_Path, V_Path)
        #'''

    # Guardar Listado de Raster de Profundidad
    Results['H']['BaU'] = raster_paths_H
    Results['V']['BaU'] = raster_paths_V
    Results['Q']['BaU'] = raster_paths_Q

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - NbS
    # ------------------------------------------------------------------------------------------------------------------
    # n-Manning modificado por la opción de adaptación - BaU
    Manning_Path    = ProjectPath + '/in/06-FLOOD/Raster/Manning_NbS.tif'
    # Infiltración modificada por la opción de adaptación - BaU
    Inf_Path        = ProjectPath + '/in/06-FLOOD/Raster/Infiltration_NbS.tif'

    raster_paths_H = {}
    raster_paths_V = {}
    raster_paths_Q = {}

    for TR_i in TR:
        H_Path = ProjectPath + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR_i}.tif'
        V_Path = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_NbS_TR-{TR_i}.tif'
        Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_NbS_TR-{TR_i}.tif'
        TS_Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/TS_Q_NbS_TR-{TR_i}.csv'

        # Crear lista de rasters
        raster_paths_H[TR_i] = H_Path
        raster_paths_V[TR_i] = V_Path
        raster_paths_Q[TR_i] = Q_Path

        if SSP == "Historic":
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: NbS | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            if IDF_Table is not None:
                ValueStorm = IDF_Table.loc[D_DS, TR_i]
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Rain=ValueStorm, Channel=Channel, ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path, FactorCal=True)
            else:
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Channel=Channel, ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path,
                                           FactorCal=True)

        else:
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: NbS | Climate: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            if IDF_Table is not None:
                ValueStorm = IDF_Table.loc[D_DS, TR_i]
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Rain=ValueStorm, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel, ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path, FactorCal=True)
            else:
                Comando = CommandFastFlood("Run",
                                           FastFloodPath, customurl=customurl,
                                           DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                           SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel,
                                           ocean=BoundaryCondition,
                                           H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path,
                                           FactorCal=True)

        #'''
        # Ejecutar comando
        if StatusExe:
            ExeFastFlood(Comando,log)

            # Recortar el raster al shapefile de la cuenca
            ClicRasterWithBasin(H_Path_Tmp, BasinPath, salida=H_Path)
            ClicRasterWithBasin(V_Path_Tmp, BasinPath, salida=V_Path)
            ClicRasterWithBasin(Q_Path_Tmp, BasinPath, salida=Q_Path)

            # Filtro de Velocidades
            CheckVelocity(V_Path, V_Path)
        #'''

    # Guardar Listado de Raster de Profundidad
    Results['H']['NbS'] = raster_paths_H
    Results['V']['NbS'] = raster_paths_V
    Results['Q']['NbS'] = raster_paths_Q
    log.flush()

    return Results

def ReadDamageCurve(ProjectPath):

    # Leer Curva de factores
    FacCurve    = pd.read_csv(os.path.join(ProjectPath,'in','06-FLOOD','Damages','01-Damage_Factor_Curves.csv'),index_col=0)

    # Leer costos máximos anuales
    MaxCost     = pd.read_csv(os.path.join(ProjectPath, 'in', '06-FLOOD', 'Damages', '02-Maximum_Damage_Cost.csv'))

    # Sacar daños
    DC          = FacCurve.mul(MaxCost.iloc[0])

    return DC

def TR_Damage(df_profundidad, df_costos, category='Residential'):
    """
    Interpola/Extrapola los costos para la categoría en función de las profundidades.

    Parámetros:
    -----------
    df_profundidad: DataFrame
        DataFrame con columnas de periodos de retorno (TR) y filas de ubicaciones/escenarios.
        Ejemplo:
                   TR_2      TR_5     TR_10  ...    TR_1000
            0     0.000031  0.000041  0.000048  ...  0.000133
            1     0.000088  0.000133  0.000155  ...  0.000417

    df_costos: DataFrame
        DataFrame con profundidades como índice y costos por categoría.
        Ejemplo:
                  Residential  Commercial  ...  Agriculture
            Flood depth [m]
            0.0            0.0         0.0  ...          0.0
            0.5            0.32        0.32 ...          0.32

    category: str, por defecto 'Residential'
        Nombre de la categoría a interpolar. Debe coincidir con una columna de df_costos.

    Retorna:
    --------
    DataFrame
        DataFrame con los mismos índices y columnas que df_profundidad, con valores interpolados
        (o extrapolados si es necesario) de costos para la categoría seleccionada.
    """

    # Extraer datos de profundidad (x) y costos (y) de la categoría
    x = df_costos.index.to_numpy(dtype=float)            # Profundidades conocidas
    y = df_costos[category].to_numpy(dtype=float)        # Costos conocidos

    # Ordenar los datos por profundidad (necesario para interp1d)
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    # Crear función de interpolación lineal con extrapolación
    interpolador = interp1d(x_sorted, y_sorted,
                            kind='linear',
                            fill_value='extrapolate')

    # Aplicar la interpolación vectorizada al array de profundidades
    valores_interpolados = interpolador(df_profundidad.values)

    # Seguro de maximo 1.0
    #valores_interpolados[valores_interpolados > 1.0] = 1.0
    valores_interpolados[valores_interpolados > np.max(y)] = np.max(y)

    # Reconstruir DataFrame con los mismos índices y columnas
    return pd.DataFrame(valores_interpolados,
                        index=df_profundidad.index,
                        columns=df_profundidad.columns)

def EAD(TR, Damage,NameCol='EAD'):
    # Crear un DataFrame vacío para almacenar resultados
    df_resultado = pd.DataFrame()

    # Probabilidad de cada periodo de retorno
    P = 1/np.array(TR)

    # Iterar sobre cada categoría detectada y calcular el valor esperado
    df_resultado[NameCol] = np.trapezoid(Damage.values, x=P, axis=1)

    # Agregar índice de píxel
    df_resultado.index.name = "Pixel"

    # Mostrar los primeros resultados
    return df_resultado

def DesaggregationData(Data, NameCol, NBS, Time):
    """
    # Esta función realiza la desagregación de los daños
    """
    """
    #Esta función realiza la desagregación de los daños
    Data.iloc[0,:]  = 4000000
    Data.iloc[1,:]  = 10000000
    Data.iloc[2, :] = 7000000
    #"""

    # Función Lambda
    Sigmoid_Desaggregation = lambda Wmax, Wo, r, t: Wmax / (1 + (((Wmax / Wo) - 1) * np.exp(-t * r)))

    # ------------------------------------------------------------------------------------------------------------------
    # BaU
    # ------------------------------------------------------------------------------------------------------------------
    # Número de items a desagregar
    nn  = np.shape(Data)[1]
    # Parámetro r de la función logística
    r   = -1 * np.log(0.000000001) / Time
    # Parámetro t de la función logística
    t   = np.arange(0, Time + 1)
    Results_BaU = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)
    # Desagregación del escenario BaU
    for i in range(0, len(NameCol)):
        Wmax = Data[NameCol[i]][1]
        Wo   = Data[NameCol[i]][0]
        Results_BaU[NameCol[i]] = Sigmoid_Desaggregation(Wmax, Wo, r, t)

    # ------------------------------------------------------------------------------------------------------------------
    # NBS
    # ------------------------------------------------------------------------------------------------------------------
    NBS_Total = np.sum(NBS[:, 2:], 1)

    # Estimación de los parámetros de la función logística, ponderando por la cantidad de área de implementación de cada
    # NbS
    t_NBS   = np.nansum(NBS[:, 0] * NBS_Total) / np.nansum(NBS_Total)
    p_NBS   = np.nansum(NBS[:, 1] * NBS_Total) / np.nansum(NBS_Total)

    # Calcular
    NBS_Year    = np.cumsum(np.sum(NBS[:, 2:], 0))
    NBS_Year    = np.concatenate([np.array([0]), NBS_Year, np.repeat(NBS_Year[-1], Time - len(NBS_Year))])
    Factor      = NBS_Year / np.sum(NBS_Total)

    # Parámetro r de la función logística
    r = -1 * np.log(0.000000001) / t_NBS
    # Parámetro t de la función logística
    t = np.arange(0, Time+1)
    # Vector vacio de resultados
    Results_NBS = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)
    # Desagregación del escenario NbS
    for i in range(0, len(NameCol)):
        Wmax    = Data[NameCol[i]].iloc[2]
        Diff    = (Data[NameCol[i]][1] - Data[NameCol[i]].iloc[2]  )
        Wo      = Results_BaU[NameCol[i]][1] - (p_NBS * Diff * 0.01)

        Results_NBS[NameCol[i]]     = Sigmoid_Desaggregation(Wmax, Wo, r, t)
        Results_NBS.loc[0, NameCol[i]] = Data[NameCol[i]].iloc[0]
        Results_NBS.loc[1, NameCol[i]] = Wo

        #Results_NBS[NameCol[i]][0]  = Data[NameCol[i]][0]
        #Results_NBS[NameCol[i]][1]  = Wo

        # Aplicación de factor
        Results_NBS[NameCol[i]]     = Results_BaU[NameCol[i]] - (Results_BaU[NameCol[i]] - Results_NBS[NameCol[i]])*Factor

    Results_BaU = Results_BaU.fillna(0).infer_objects(copy=False)
    Results_NBS = Results_NBS.fillna(0).infer_objects(copy=False)

    return Results_BaU, Results_NBS

def ROI_FastFlood(PathProject_ROI):

    # ------------------------------------------------------------------------------------------------------------------
    # Leer Archivos de entrada
    # ------------------------------------------------------------------------------------------------------------------
    INPUTS          = 'in'
    OUTPUTS         = 'out'
    CostNBS         = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '1-NBS_Cost.csv'))
    Porfolio        = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '2-Porfolio_NBS.csv'))
    TD              = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '3-Financial_Parmeters.csv'))
    TimeAnalisys    = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '4-Time.csv'))
    C_BaU           = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '5-CO2_BaU.csv'))
    C_NBS           = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '6-CO2_NBS.csv'))
    CostNBS_Damage  = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '8-Damages_NbS.csv'),index_col=0).drop(0)
    CostBaU_Damage  = pd.read_csv(os.path.join(PathProject_ROI, INPUTS, '7-Damages_BaU.csv'),index_col=0).drop(0)

    # ------------------------------------------------------------------------------------------------------------------
    # Calculo de Beneficio Total
    # ------------------------------------------------------------------------------------------------------------------
    Benefit_Damage = CostBaU_Damage.values - CostNBS_Damage.values

    # Control para que los beneficios no sean negativos [Bug - 24-03-2023]
    Benefit_Damage[Benefit_Damage < 0] = 0

    '''
    ####################################################################################################################
                                                    Calculo de inversiones NBS
    ####################################################################################################################
    '''
    del CostNBS['Parameters']
    # Portafolio de SbN
    Porfolio    = Porfolio.set_index('Time')
    # Nombre de las SbN
    NameNBS     = CostNBS.columns
    # Timepo del análisis ROI
    t_roi       = TimeAnalisys['Time_ROI'][0]
    # Tiempo de implementación de las SbN
    t_nbs       = TimeAnalisys['Time_Implementation_NBS'][0]
    # Número de SbN implementadas
    n_nbs       = len(NameNBS)
    # Costos de implementación
    Cost_I      = np.zeros((t_roi, n_nbs))
    # Costos de mantenimiento y operación
    Cost_M      = np.zeros((t_roi, n_nbs))
    # Costos de oportunidad
    Cost_O      = np.zeros((t_roi, n_nbs))
    # Costos de plataforma
    Cost_P      = np.ones((t_roi, len(TD.values) - 5))

    # ------------------------------------------------------------------------------------------------------------------
    # Implementation Cost
    # ------------------------------------------------------------------------------------------------------------------
    Cost_I[0:t_nbs, :] = Porfolio.values * CostNBS.values[0, :]

    # ------------------------------------------------------------------------------------------------------------------
    # Costos de Operación y Mantenimiento
    # ------------------------------------------------------------------------------------------------------------------
    for j in range(0, t_nbs):
        Tmp1 = np.zeros((t_roi, n_nbs))
        for i in range(0, n_nbs):
            Posi = np.arange(j, int(t_roi), int(CostNBS[3:].values[0][i]))
            Tmp1[Posi, i] = 1

        Cost_M = Cost_M + Tmp1 * Porfolio.values[j, :] * CostNBS.values[1, :]

    # ------------------------------------------------------------------------------------------------------------------
    # Costo de Oportunidad
    # ------------------------------------------------------------------------------------------------------------------
    Cost_O[0:t_nbs, :] = np.cumsum(Porfolio.values * CostNBS.values[2, :], 0)
    Cost_O[t_nbs:, :] = Cost_O[t_nbs - 1, :]

    # ------------------------------------------------------------------------------------------------------------------
    # Costos de Transacción
    # ------------------------------------------------------------------------------------------------------------------
    TD = TD.set_index('ID')
    Cost_T = (np.sum(Cost_I, 1) + np.sum(Cost_M, 1) + np.sum(Cost_O, 1)) * TD['Value'][1]

    # ------------------------------------------------------------------------------------------------------------------
    # Costos de Plataforma
    # ------------------------------------------------------------------------------------------------------------------
    Cost_P = Cost_P * TD['Value'].values[5:]

    # ------------------------------------------------------------------------------------------------------------------
    # Carbons
    # ------------------------------------------------------------------------------------------------------------------
    Factor      = 44 / 12  # 44 g/mol CO2 - 12 g/mol C
    DifCO2_1    = (C_NBS.sum(1) - C_BaU.sum(1))  # Se estima la diferencia de almacenamiento de CO2 entre los dos escenarios
    DifCO2_2    = DifCO2_1.diff()  # Se estima la diferencia de los diferenciales de almacenamiento de CO2 por año
    DifCO2_2[np.isnan(DifCO2_2)]    = 0  # Todos los valores NaN son iguales a cero
    DifCO2_2[DifCO2_2 < 0.001]      = 0  # todos los valores por debajo de 0.001 se consideran cero
    Carbons     = Factor * DifCO2_2 * TD['Value'][5]  # Se pasa de CO2 a dinero
    Carbons     = Carbons.values[1:]

    # ------------------------------------------------------------------------------------------------------------------
    # Beneficios Totales (Daños + Carbono)
    # ------------------------------------------------------------------------------------------------------------------
    TotalBenefit_1      = np.sum(Benefit_Damage, 1) + Carbons
    TotalBenefit_1_TD_1 = TotalBenefit_1 / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    TotalBenefit_1_TD_2 = TotalBenefit_1 / ((1 + TD['Value'][3]) ** np.arange(1, t_roi + 1))
    TotalBenefit_1_TD_3 = TotalBenefit_1 / ((1 + TD['Value'][4]) ** np.arange(1, t_roi + 1))

    # ------------------------------------------------------------------------------------------------------------------
    # Costos Totales (Mantenimiento + Transacción + Oportunidad + Implementación + Plataforma)
    # ------------------------------------------------------------------------------------------------------------------
    TotalBenefit_2      = Cost_M.sum(1) + Cost_T + Cost_O.sum(1) + Cost_I.sum(1) + Cost_P.sum(1)
    TotalBenefit_2_TD_1 = TotalBenefit_2 / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    TotalBenefit_2_TD_2 = TotalBenefit_2 / ((1 + TD['Value'][3]) ** np.arange(1, t_roi + 1))
    TotalBenefit_2_TD_3 = TotalBenefit_2 / ((1 + TD['Value'][4]) ** np.arange(1, t_roi + 1))

    # ------------------------------------------------------------------------------------------------------------------
    # Cálculo ROI
    # ------------------------------------------------------------------------------------------------------------------
    # ROI - Sin tasa de descuento
    ROI_0 = TotalBenefit_1.sum() / TotalBenefit_2.sum()
    # ROI - Con tasa de descuento promedio
    ROI_1 = TotalBenefit_1_TD_1.sum() / TotalBenefit_2_TD_1.sum()
    # ROI - Con tasa de descuento mínima
    ROI_2 = TotalBenefit_1_TD_2.sum() / TotalBenefit_2_TD_2.sum()
    # ROI - Con tasa de descuento máxima
    ROI_3 = TotalBenefit_1_TD_3.sum() / TotalBenefit_2_TD_3.sum()

    # ------------------------------------------------------------------------------------------------------------------
    # Cálculo de Valor Presente Neto - I, M, O, T, P
    # ------------------------------------------------------------------------------------------------------------------
    # Costos de implementación a valor presente neto
    NPV_I = Cost_I.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    # Costos de mantenimiento a valor presente neto
    NPV_M = Cost_M.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    # Costos de oportunidad a valor presente neto
    NPV_O = Cost_O.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    # Costos de transacción valor presente neto
    NPV_T = Cost_T / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))
    # Costos de plataforma a valor presente neto
    NPV_P = Cost_P.sum(1) / ((1 + TD['Value'][2]) ** np.arange(1, t_roi + 1))

    '''
    ####################################################################################################################
                                                    Guardar Resultados
    ####################################################################################################################
    '''
    NameIndex = np.arange(1, t_roi + 1)
    Total_1 = pd.DataFrame(data=np.zeros((t_roi, 7)), columns=['Damage', 'Carbons', 'I', 'M', 'O', 'T', 'P'], index=NameIndex)
    Total_1['Damage']   = np.sum(Benefit_Damage, 1)
    Total_1['Carbons']  = Carbons
    Total_1['I']        = Cost_I.sum(1)
    Total_1['M']        = Cost_M.sum(1)
    Total_1['O']        = Cost_O.sum(1)
    Total_1['T']        = Cost_T
    Total_1['P']        = Cost_P.sum(1)

    Total_2 = pd.DataFrame(data=np.zeros((t_roi, 4)), columns=['Total', 'TD_Min', 'TD_Mean', 'TD_Max'])
    Total_2['Total']    = TotalBenefit_1
    Total_2['TD_Min']   = TotalBenefit_1_TD_2
    Total_2['TD_Mean']  = TotalBenefit_1_TD_1
    Total_2['TD_Max']   = TotalBenefit_1_TD_3

    Total_3 = pd.DataFrame(data=np.zeros((t_roi, 4)), columns=['Total', 'TD_Min', 'TD_Mean', 'TD_Max'])
    Total_3['Total']    = TotalBenefit_2
    Total_3['TD_Min']   = TotalBenefit_2_TD_2
    Total_3['TD_Mean']  = TotalBenefit_2_TD_1
    Total_3['TD_Max']   = TotalBenefit_2_TD_3

    ROI = pd.DataFrame(data=np.zeros((1, 4)), columns=['Total', 'TD_Min', 'TD_Mean', 'TD_Max'])
    ROI['Total']        = ROI_0
    ROI['TD_Min']       = ROI_2
    ROI['TD_Mean']      = ROI_1
    ROI['TD_Max']       = ROI_3

    NPV = pd.DataFrame(data=np.zeros((1, 7)), columns=['Implementation', 'Maintenance', 'Opportunity', 'Transaction', 'Platform', 'Benefit', 'Total'])
    NPV['Implementation']   = -1 * NPV_I.sum()
    NPV['Maintenance']      = -1 * NPV_M.sum()
    NPV['Opportunity']       = -1 * NPV_O.sum()
    NPV['Transaction']      = -1 * NPV_T.sum()
    NPV['Platform']         = -1 * NPV_P.sum()
    NPV['Benefit']          = TotalBenefit_1_TD_1.sum()
    NPV['Total']            = NPV.sum(1)

    Total_2 = Total_2.set_index(np.arange(0, t_roi) + 1)
    Total_3 = Total_3.set_index(np.arange(0, t_roi) + 1)

    Total_4_0 = pd.DataFrame(data=Cost_I, columns=NameNBS, index=NameIndex)
    Total_5_0 = pd.DataFrame(data=Cost_M, columns=NameNBS, index=NameIndex)
    Total_6_0 = pd.DataFrame(data=Cost_O, columns=NameNBS, index=NameIndex)
    Total_7_0 = pd.DataFrame(data=Cost_T, columns=['Cost'], index=NameIndex)
    Total_8_0 = pd.DataFrame(data=Cost_P, columns=TD['Cost'][5:], index=NameIndex)

    Total_9_0 = CostBaU_Damage - CostNBS_Damage

    # Control para que los beneficios no sean negativos - [Bug - 24-03-2023]
    Total_9_0[Total_9_0 < 0] = 0

    Total_11_0 = pd.DataFrame(data=Carbons, columns=['Carbons'], index=NameIndex)

    # ------------------------------------------------------------------------------------------------------------------
    # Tasa de descuento mínima
    # ------------------------------------------------------------------------------------------------------------------
    Total_4_1 = Total_4_0 / ((1 + (np.ones(Total_4_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_4_0.shape[1], 1)).transpose())
    Total_5_1 = Total_5_0 / ((1 + (np.ones(Total_5_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_5_0.shape[1], 1)).transpose())
    Total_6_1 = Total_6_0 / ((1 + (np.ones(Total_6_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_6_0.shape[1], 1)).transpose())
    Total_7_1 = Total_7_0 / ((1 + (np.ones(Total_7_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_7_0.shape[1], 1)).transpose())
    Total_8_1 = Total_8_0 / ((1 + (np.ones(Total_8_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_8_0.shape[1], 1)).transpose())
    Total_9_1 = Total_9_0 / ((1 + (np.ones(Total_9_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1),(Total_9_0.shape[1], 1)).transpose())

    Total_11_1 = Total_11_0 / ((1 + (np.ones(Total_11_0.shape) * TD['Value'][3])) ** np.tile(np.arange(1, t_roi + 1), (Total_11_0.shape[1], 1)).transpose())

    # ------------------------------------------------------------------------------------------------------------------
    # Tasa de descuento promedio
    # ------------------------------------------------------------------------------------------------------------------
    Total_4_2 = Total_4_0 / ((1 + (np.ones(Total_4_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_4_0.shape[1], 1)).transpose())
    Total_5_2 = Total_5_0 / ((1 + (np.ones(Total_5_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_5_0.shape[1], 1)).transpose())
    Total_6_2 = Total_6_0 / ((1 + (np.ones(Total_6_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_6_0.shape[1], 1)).transpose())
    Total_7_2 = Total_7_0 / ((1 + (np.ones(Total_7_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_7_0.shape[1], 1)).transpose())
    Total_8_2 = Total_8_0 / ((1 + (np.ones(Total_8_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_8_0.shape[1], 1)).transpose())
    Total_9_2 = Total_9_0 / ((1 + (np.ones(Total_9_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1),(Total_9_0.shape[1], 1)).transpose())

    Total_11_2 = Total_11_0 / ((1 + (np.ones(Total_11_0.shape) * TD['Value'][2])) ** np.tile(np.arange(1, t_roi + 1), (Total_11_0.shape[1], 1)).transpose())

    # ------------------------------------------------------------------------------------------------------------------
    # Tasa de descuento máxima
    # ------------------------------------------------------------------------------------------------------------------
    Total_4_3 = Total_4_0 / ((1 + (np.ones(Total_4_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_4_0.shape[1], 1)).transpose())
    Total_5_3 = Total_5_0 / ((1 + (np.ones(Total_5_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_5_0.shape[1], 1)).transpose())
    Total_6_3 = Total_6_0 / ((1 + (np.ones(Total_6_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_6_0.shape[1], 1)).transpose())
    Total_7_3 = Total_7_0 / ((1 + (np.ones(Total_7_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_7_0.shape[1], 1)).transpose())
    Total_8_3 = Total_8_0 / ((1 + (np.ones(Total_8_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_8_0.shape[1], 1)).transpose())
    Total_9_3 = Total_9_0 / ((1 + (np.ones(Total_9_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1),(Total_9_0.shape[1], 1)).transpose())
    Total_11_3 = Total_11_0 / ((1 + (np.ones(Total_11_0.shape) * TD['Value'][4])) ** np.tile(np.arange(1, t_roi + 1), (Total_11_0.shape[1], 1)).transpose())

    # ------------------------------------------------------------------------------------------------------------------
    # Guardar datos
    # ------------------------------------------------------------------------------------------------------------------
    Total_4_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '1.0_Implementation_Costs.csv'), index_label='Time')
    Total_5_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '2.0_Maintenance_Costs.csv'), index_label='Time')
    Total_6_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '3.0_Opportunity_Costs.csv'), index_label='Time')
    Total_7_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '4.0_Transaction_Costs.csv'), index_label='Time')
    Total_8_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '5.0_Platform_Costs.csv'), index_label='Time')
    Total_9_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '6.0_Damage_Saves.csv'))
    Total_11_0.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '7.0_Carbons_Saves.csv'))

    Total_4_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '1.1_Implementation_Costs.csv'), index_label='Time')
    Total_5_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '2.1_Maintenance_Costs.csv'), index_label='Time')
    Total_6_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '3.1_Opportunity_Costs.csv'), index_label='Time')
    Total_7_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '4.1_Transaction_Costs.csv'), index_label='Time')
    Total_8_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '5.1_Platform_Costs.csv'), index_label='Time')
    Total_9_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '6.1_Damage_Saves.csv'))
    Total_11_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '7.1_Carbons_Saves.csv'))

    Total_4_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '1.2_Implementation_Costs.csv'), index_label='Time')
    Total_5_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '2.2_Maintenance_Costs.csv'), index_label='Time')
    Total_6_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '3.2_Opportunity_Costs.csv'), index_label='Time')
    Total_7_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '4.2_Transaction_Costs.csv'), index_label='Time')
    Total_8_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '5.2_Platform_Costs.csv'), index_label='Time')
    Total_9_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '6.2_Damage_Saves.csv'))
    Total_11_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '7.2_Carbons_Saves.csv'))

    Total_4_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '1.3_Implementation_Costs.csv'), index_label='Time')
    Total_5_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '2.3_Maintenance_Costs.csv'), index_label='Time')
    Total_6_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '3.3_Opportunity_Costs.csv'), index_label='Time')
    Total_7_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '4.3_Transaction_Costs.csv'), index_label='Time')
    Total_8_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '5.3_Platform_Costs.csv'), index_label='Time')
    Total_9_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '6.3_Damage_Saves.csv'))
    Total_11_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '7.3_Carbons_Saves.csv'))

    Total_1.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '8_GlobalTotals.csv'), index_label='Time')
    Total_2.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '9_Benefit_Sensitivity.csv'), index_label='Time')
    Total_3.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '10_Cost_Sensitivity.csv'), index_label='Time')
    ROI.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '11_ROI_Sensitivity.csv'))
    NPV.to_csv(os.path.join(PathProject_ROI, OUTPUTS, '12_NPV.csv'))

def Raster2Zonal(profundidades, shapefile_mascara=None, Threshold_H=0.01):
    """
    Extrae y resume estadísticas básicas de uno o varios rásteres de profundidad,
    limitando opcionalmente por un shapefile en WGS84.

    Esta función optimizada utiliza técnicas de procesamiento eficiente incluyendo:
    - Lectura por bloques para manejo óptimo de memoria
    - Operaciones vectorizadas de NumPy
    - Uso de rasterio.mask para aplicación directa de geometrías
    - Lectura por ventanas cuando se aplica máscara espacial

    Parámetros
    ----------
    profundidades : str o dict
        Ruta a un solo ráster o diccionario {nombre_columna: ruta_raster}.
        Los rásteres deben contener valores de profundidad de inundación.

    shapefile_mascara : str, opcional
        Ruta a shapefile en WGS84 que define el área de análisis.
        Si se proporciona, el análisis se limitará a esta geometría.
        Por defecto None (analiza todo el ráster).

    Threshold_H : float, opcional
        Valor mínimo de profundidad para conservar en metros.
        Valores menores se consideran como 0 (sin inundación).
        Por defecto 0.01 metros.

    Retorna
    -------
    pd.DataFrame
        DataFrame con estadísticas por capa que incluye:
        - 'TR': Nombre de la capa/período de retorno
        - 'Maximum Flood Depth [m]': Profundidad máxima de inundación
        - 'Peak Descharge [m3/s]': Caudal pico (siempre 0.0)
        - 'Flood Area [ha]': Área inundada en hectáreas

    Raises
    ------
    ValueError
        Si el shapefile no contiene geometría válida.
        Si los archivos de ráster no se pueden abrir.

    Ejemplos
    --------
    >>> # Análisis de un solo ráster
    >>> resultado = Raster2Zonal('profundidad_tr100.tif')

    >>> # Análisis de múltiples rásteres con máscara
    >>> rasteres = {
    ...     'TR_10': 'profundidad_tr10.tif',
    ...     'TR_100': 'profundidad_tr100.tif'
    ... }
    >>> resultado = Raster2Zonal(rasteres, 'zona_estudio.shp', Threshold_H=0.05)

    Notas
    -----
    - Los rásteres deben estar en el mismo sistema de coordenadas
    - La función reprojectará automáticamente el shapefile al CRS del ráster
    - Para rásteres muy grandes, la función utiliza procesamiento por bloques
    - El campo 'Peak Descharge' siempre retorna 0.0 (placeholder para compatibilidad)
    """

    print("🚀 Iniciando análisis Raster2Zonal optimizado...")
    inicio_total = time.time()

    # Normalizar entrada a diccionario
    if isinstance(profundidades, str):
        profundidades = {'Profundidad': profundidades}
        print(f"📁 Procesando un solo ráster: {os.path.basename(profundidades['Profundidad'])}")
    else:
        print(f"📁 Procesando {len(profundidades)} rásteres: {list(profundidades.keys())}")

    # Obtener información del primer ráster como referencia
    print("🔍 Analizando ráster de referencia...")
    primera_ruta = list(profundidades.values())[0]

    try:
        with rasterio.open(primera_ruta) as ref_raster:
            crs_objetivo = ref_raster.crs
            shape = ref_raster.shape
            transform = ref_raster.transform
            pixel_area_m2 = abs(transform.a * transform.e)
            print(f"   • CRS: {crs_objetivo}")
            print(f"   • Dimensiones: {shape[1]} x {shape[0]} píxeles")
            print(f"   • Área por píxel: {pixel_area_m2:.2f} m²")
    except Exception as e:
        raise ValueError(f"Error al leer ráster de referencia: {e}")

    # Procesar shapefile de máscara si se proporciona
    geometrias_mask = None
    usar_ventana = False

    if shapefile_mascara:
        print(f"🗺️  Procesando máscara espacial: {os.path.basename(shapefile_mascara)}")
        try:
            gdf = gpd.read_file(shapefile_mascara)
            if 'geometry' not in gdf or gdf.geometry.isna().all():
                raise ValueError("El shapefile no contiene geometría válida.")

            print(f"   • Geometrías originales: {len(gdf)} en CRS {gdf.crs}")

            # Reproyectar si es necesario
            if gdf.crs != crs_objetivo:
                print("   • Reproyectando geometrías al CRS del ráster...")
                gdf = gdf.to_crs(crs_objetivo)

            geometrias_mask = gdf.geometry.values
            usar_ventana = True
            print("   • Máscara espacial configurada correctamente")

        except Exception as e:
            raise ValueError(f"Error al procesar shapefile: {e}")
    else:
        print("📊 Sin máscara espacial - analizando ráster completo")

    # Procesar cada ráster
    registros = []

    for i, (nombre_columna, ruta) in enumerate(profundidades.items(), 1):
        print(f"\n🔄 Procesando ráster {i}/{len(profundidades)}: {nombre_columna}")
        inicio_raster = time.time()

        try:
            with rasterio.open(ruta) as src:
                # Verificar compatibilidad
                if src.crs != crs_objetivo or src.transform != transform or src.shape != shape:
                    print("   ⚠️  Ráster con diferente proyección/resolución - continuando con precaución")

                # Estrategia optimizada: usar rasterio.mask si hay geometrías
                if geometrias_mask is not None:
                    print("   🎯 Aplicando máscara geométrica...")

                    # Usar rasterio.mask para aplicar geometrías directamente
                    masked_data, masked_transform = rasterio.mask.mask(
                        src, geometrias_mask, crop=True, nodata=0
                    )
                    data = masked_data[0]  # Primera banda

                    print(f"   • Datos enmascarados: {data.shape} píxeles")

                else:
                    print("   📖 Leyendo ráster completo...")
                    data = src.read(1)  # Leer toda la primera banda

                print(f"   • Datos cargados: {data.size:,} píxeles")

        except Exception as e:
            print(f"   ❌ Error al leer ráster: {e}")
            # Crear registro con valores por defecto para mantener compatibilidad
            registros.append({
                'TR': nombre_columna,
                'Maximum Flood Depth [m]': np.nan,
                'Peak Descharge [m3/s]': 0.0,
                'Flood Area [ha]': 0.0,
            })
            continue

        # Aplicar threshold de manera vectorizada
        print(f"   🔢 Aplicando threshold ({Threshold_H} m)...")

        # Operación vectorizada ultrarrápida
        mask_validos = (data >= Threshold_H) & (~np.isnan(data)) & (data != 0)
        valores_validos = data[mask_validos]

        if len(valores_validos) == 0:
            print(f"   ⚠️  Sin valores válidos para: {nombre_columna}")
            estadisticas = {
                'TR': nombre_columna,
                'Maximum Flood Depth [m]': np.nan,
                'Peak Descharge [m3/s]': 0.0,
                'Flood Area [ha]': 0.0,
            }
        else:
            # Cálculos vectorizados
            area_total_ha = (len(valores_validos) * pixel_area_m2) / 10000.0
            max_depth = float(np.max(valores_validos))

            estadisticas = {
                'TR': nombre_columna,
                'Maximum Flood Depth [m]': max_depth,
                'Peak Descharge [m3/s]': 0.0,  # Placeholder para compatibilidad
                'Flood Area [ha]': area_total_ha,
            }

            print(f"   ✅ Profundidad máxima: {max_depth:.3f} m")
            print(f"   ✅ Área inundada: {area_total_ha:.2f} ha")
            print(f"   ✅ Píxeles válidos: {len(valores_validos):,}")

        registros.append(estadisticas)

        tiempo_raster = time.time() - inicio_raster
        print(f"   ⏱️  Tiempo de procesamiento: {tiempo_raster:.2f}s")

    # Crear DataFrame resultado
    resultado = pd.DataFrame(registros)

    tiempo_total = time.time() - inicio_total
    print(f"\n🎉 Análisis completado exitosamente!")
    print(f"⏱️  Tiempo total: {tiempo_total:.2f}s")
    print(f"📊 Registros generados: {len(resultado)}")

    resultado = resultado.fillna(0)
    return resultado

def Raster2DataFrame_Damages_Generator(ruta_cobertura, profundidades, codigos_cobertura,
                                        shapefile_mascara=None, Threshold_H=0.01,
                                        verbose=True, log_file=None, chunk_size=2048):
    """
    Generador que procesa rásteres por chunks, yielding cada chunk como DataFrame.

    Mismo algoritmo que Raster2DataFrame_Damages_Chunked pero sin acumular en memoria.
    Threshold y filtro de ceros se aplican por chunk antes del yield.

    Yields
    ------
    DataFrame
        Chunk con columnas [Code, TR_1, TR_2, ...] para píxeles válidos del chunk.
    """
    import rasterio.windows

    logger = get_module_logger(log_file=log_file)
    logger.info("="*80)
    logger.info(f"INICIANDO Raster2DataFrame_Damages_Generator: {ruta_cobertura}")
    log_memory(logger, "Memoria inicial")

    if isinstance(profundidades, str):
        profundidades = {'Profundidad': profundidades}

    with rasterio.open(ruta_cobertura) as src_cov:
        crs_objetivo = src_cov.crs
        transform = src_cov.transform
        height, width = src_cov.shape

        geometria_mask = None
        if shapefile_mascara:
            gdf = gpd.read_file(shapefile_mascara)
            if not isinstance(gdf, gpd.GeoDataFrame) or 'geometry' not in gdf:
                raise ValueError("El archivo proporcionado no es un shapefile válido con geometría.")
            gdf = gdf.to_crs(crs_objetivo)
            geometria_mask = gdf.geometry

        n_chunks_x = (width  + chunk_size - 1) // chunk_size
        n_chunks_y = (height + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_x * n_chunks_y
        chunk_num = 0

        logger.info(f"Dimensiones: {width}x{height} px | Chunks: {total_chunks} ({n_chunks_y}x{n_chunks_x})")

        for row_offset in range(0, height, chunk_size):
            for col_offset in range(0, width, chunk_size):
                chunk_num += 1
                win_height = min(chunk_size, height - row_offset)
                win_width  = min(chunk_size, width  - col_offset)
                window = rasterio.windows.Window(col_offset, row_offset, win_width, win_height)

                cov_data_chunk  = src_cov.read(1, window=window)
                chunk_transform = rasterio.windows.transform(window, transform)

                if geometria_mask is not None and geometria_mask.notna().any():
                    mascara_geo_chunk = geometry_mask(
                        geometries=geometria_mask,
                        transform=chunk_transform,
                        invert=True,
                        out_shape=(win_height, win_width)
                    )
                else:
                    mascara_geo_chunk = np.ones((win_height, win_width), dtype=bool)

                mascara_cod_chunk   = np.isin(cov_data_chunk, codigos_cobertura)
                mascara_total_chunk = mascara_cod_chunk & mascara_geo_chunk
                filas_chunk, columnas_chunk = np.where(mascara_total_chunk)

                if len(filas_chunk) == 0:
                    del cov_data_chunk, mascara_geo_chunk, mascara_cod_chunk, mascara_total_chunk
                    continue

                valores_cobertura_chunk = cov_data_chunk[filas_chunk, columnas_chunk]
                df_chunk = pd.DataFrame({'Code': valores_cobertura_chunk})

                for nombre_columna, ruta in profundidades.items():
                    with rasterio.open(ruta) as src_prof:
                        with WarpedVRT(
                            src_prof,
                            crs=crs_objetivo,
                            transform=chunk_transform,
                            width=win_width,
                            height=win_height,
                            resampling=Resampling.nearest
                        ) as vrt_prof:
                            prof_data_chunk = vrt_prof.read(1)
                            df_chunk[nombre_columna] = prof_data_chunk[filas_chunk, columnas_chunk]
                            del prof_data_chunk

                # Threshold y filtro de ceros por chunk
                columnas_prof = list(profundidades.keys())
                for col in columnas_prof:
                    df_chunk[col] = np.where(df_chunk[col] < Threshold_H, 0, df_chunk[col])
                df_chunk = df_chunk[df_chunk[columnas_prof].sum(axis=1) != 0].reset_index(drop=True)

                del cov_data_chunk, mascara_geo_chunk, mascara_cod_chunk, mascara_total_chunk
                del filas_chunk, columnas_chunk, valores_cobertura_chunk

                if df_chunk.empty:
                    continue

                logger.debug(f"Chunk {chunk_num}/{total_chunks}: {len(df_chunk)} píxeles válidos")
                log_memory(logger, f"chunk {chunk_num}/{total_chunks}")

                yield df_chunk

    logger.info("COMPLETADO Raster2DataFrame_Damages_Generator")
    logger.info("="*80)

def Clic_Mosaic_DataBase(carpeta_tiles, aoi_path, ruta_salida, crs_salida=None, simplificar_aoi=None,
                         recorte_exacto=True, verbose=True):
    """
       Genera un mosaico raster recortado a un Área de Interés (AOI) a partir de una carpeta de tiles raster.

       Parámetros:
       -----------
       carpeta_tiles : str
           Ruta a la carpeta que contiene los archivos raster .tif.

       aoi_path : str
           Ruta al archivo Threshold_H=(puede ser shapefile, geojson, geopackage o raster .tif).

       ruta_salida : str
           Ruta de salida donde se guardará el mosaico final en formato GeoTIFF.

       crs_salida : str, opcional
           CRS de salida (por defecto usa el CRS de los tiles).

       simplificar_Threshold_H: float, opcional
           Tolerancia para simplificar la geometría del Threshold_H(en unidades del CRS del AOI).

       recorte_exacto : bool, opcional
           Si True, recorta el mosaico con la forma exacta del shapefile.
           Si False, mantiene el recorte rectangular (comportamiento por defecto).

       verbose : bool
           Si True, imprime mensajes de avance y tiempos de ejecución.

       Retorno:
       --------
       No retorna nada. Guarda el archivo final en disco.
    """
    start_total = time.perf_counter()

    # === A. Leer Threshold_H===
    t0 = time.perf_counter()
    ext = os.path.splitext(aoi_path)[1].lower()
    if ext in [".shp", ".geojson", ".gpkg"]:
        with fiona.open(aoi_path, "r") as src:
            crs_Threshold_H= CRS.from_wkt(src.crs_wkt)
            geometries = [shape(f["geometry"]) for f in src]
        geom_union = unary_union(geometries)
        # Guardar la geometría original para el recorte exacto
        geom_original = geom_union
    elif ext in [".tif", ".tiff"]:
        with rasterio.open(aoi_path) as src:
            crs_Threshold_H= CRS(src.crs)
            bounds = src.bounds
            geom_union = box(*bounds)
            # Para rasters, la geometría original es la misma que la union
            geom_original = geom_union
    else:
        raise ValueError("Threshold_Hdebe ser shapefile o ráster")

    if simplificar_aoi:
        geom_union = geom_union.simplify(simplificar_aoi)
        # Si se simplifica, también actualizar la geometría original si se va a usar recorte exacto
        if recorte_exacto:
            geom_original = geom_union
    if verbose: print(f"A. Threshold_Hcargado en {time.perf_counter() - t0:.2f} s")

    # === B. Leer/crear índice CSV ===
    t0 = time.perf_counter()
    csv_index = os.path.join(carpeta_tiles, "bbox_tiles.csv")
    if os.path.exists(csv_index):
        bbox_df = pd.read_csv(csv_index)
    else:
        registros = []
        for tile in glob.glob(os.path.join(carpeta_tiles, "*.tif")):
            try:
                with rasterio.open(tile) as src:
                    bounds = src.bounds
                    crs = src.crs.to_string()
                    registros.append({
                        "tile": os.path.basename(tile),
                        "norte": bounds.top,
                        "sur": bounds.bottom,
                        "este": bounds.right,
                        "oeste": bounds.left,
                        "crs": crs
                    })
            except RasterioIOError:
                continue
        bbox_df = pd.DataFrame(registros)
        bbox_df.to_csv(csv_index, index=False)
    if verbose: print(f"B. Índice bbox en {time.perf_counter() - t0:.2f} s")

    # === C. Reproyectar Threshold_H===
    t0 = time.perf_counter()
    crs_tiles = CRS.from_user_input(bbox_df.iloc[0]["crs"])
    if crs_Threshold_H!= crs_tiles:
        transformer = Transformer.from_crs(crs_Threshold_H, crs_tiles, always_xy=True)
        coords = list(geom_union.exterior.coords)
        coords_proj = [transformer.transform(x, y) for x, y in coords]
        geom_proj = box(*shape({'type': 'Polygon', 'coordinates': [coords_proj]}).bounds)

        # Reproyectar también la geometría original si se va a usar recorte exacto
        if recorte_exacto:
            if geom_original.geom_type == 'Polygon':
                coords_orig = list(geom_original.exterior.coords)
                coords_orig_proj = [transformer.transform(x, y) for x, y in coords_orig]
                geom_original_proj = shape({'type': 'Polygon', 'coordinates': [coords_orig_proj]})
            elif geom_original.geom_type == 'MultiPolygon':
                polygons_proj = []
                for polygon in geom_original.geoms:
                    coords_orig = list(polygon.exterior.coords)
                    coords_orig_proj = [transformer.transform(x, y) for x, y in coords_orig]
                    polygons_proj.append(shape({'type': 'Polygon', 'coordinates': [coords_orig_proj]}))
                geom_original_proj = MultiPolygon(polygons_proj)
            else:
                geom_original_proj = geom_original
        else:
            geom_original_proj = None
    else:
        geom_proj = geom_union
        geom_original_proj = geom_original if recorte_exacto else None
    if verbose: print(f"C. Reproyección AOI en {time.perf_counter() - t0:.2f} s")

    # === D. Filtrar tiles ===
    t0 = time.perf_counter()
    minx, miny, maxx, maxy = geom_proj.bounds
    bbox_df_filt = bbox_df[
        (bbox_df["oeste"] < maxx) &
        (bbox_df["este"] > minx) &
        (bbox_df["sur"] < maxy) &
        (bbox_df["norte"] > miny)
        ]
    if bbox_df_filt.empty:
        raise RuntimeError("❌ Ningún tile intersecta el AOI")
    if verbose: print(f"D. Tiles filtrados: {len(bbox_df_filt)} en {time.perf_counter() - t0:.2f} s")

    # === E. Construir lista de tiles a procesar ===
    t0 = time.perf_counter()
    tiles_list = []
    for _, row in bbox_df_filt.iterrows():
        path_tile = os.path.join(carpeta_tiles, row["tile"])
        if os.path.exists(path_tile):
            tiles_list.append(path_tile)

    if not tiles_list:
        raise RuntimeError("❌ No se encontraron tiles válidos")
    if verbose: print(f"E. Tiles a procesar: {len(tiles_list)} en {time.perf_counter() - t0:.2f} s")

    # === F. Crear VRT (Virtual Raster) y fusionar con GDAL Warp ===
    t0 = time.perf_counter()

    # Crear VRT temporal en memoria
    vrt_path = tempfile.NamedTemporaryFile(suffix='.vrt', delete=False).name
    try:
        # BuildVRT crea un mosaico virtual sin cargar datos en RAM
        vrt_options = gdal.BuildVRTOptions(
            resolution='highest',
            resampleAlg='nearest'
        )
        vrt_ds = gdal.BuildVRT(vrt_path, tiles_list, options=vrt_options)
        if vrt_ds is None:
            raise RuntimeError("❌ Error creando VRT")
        vrt_ds = None  # Cerrar para liberar

        if verbose: print(f"F.1. VRT creado: {len(tiles_list)} tiles fusionados")

        # Preparar opciones de Warp
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

        warp_options = {
            'format': 'GTiff',
            'creationOptions': ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'],
            'dstNodata': 0,
            'resampleAlg': gdal.GRA_NearestNeighbour,
            'multithread': True,
            'warpMemoryLimit': 512  # Limitar memoria de warp a 512 MB
        }

        # Intentar recorte exacto si está habilitado
        cutline_path = None
        cutline_applied = False

        if recorte_exacto and geom_original_proj is not None:
            cutline_path = tempfile.NamedTemporaryFile(suffix='.shp', delete=False).name
            try:
                # Crear GeoDataFrame con la geometría y el CRS correcto
                gdf = gpd.GeoDataFrame(
                    {'geometry': [geom_original_proj]},
                    crs=crs_tiles
                )

                # Guardar como shapefile temporal (incluye .prj con CRS)
                gdf.to_file(cutline_path, driver='ESRI Shapefile')

                # Agregar cutline a las opciones
                # El CRS se infiere automáticamente del archivo .prj del shapefile
                warp_options['cutlineDSName'] = cutline_path
                warp_options['cropToCutline'] = True
                warp_options['cutlineBlend'] = 0
                cutline_applied = True

                if verbose: print(f"F.2. Aplicando recorte exacto con geometría...")
            except Exception as e:
                if verbose: print(f"⚠️ Advertencia: No se pudo crear cutline: {e}")
                if verbose: print(f"    Usando recorte rectangular como fallback...")
                cutline_path = None
                cutline_applied = False

        # Si no se aplicó recorte exacto, usar recorte rectangular
        if not cutline_applied:
            minx, miny, maxx, maxy = geom_proj.bounds
            warp_options['outputBounds'] = [minx, miny, maxx, maxy]
            if verbose and recorte_exacto: print(f"F.2. Usando recorte rectangular (bounds)")
            elif verbose: print(f"F.2. Recorte rectangular aplicado")

        # Aplicar CRS de salida si se especificó
        if crs_salida:
            warp_options['dstSRS'] = crs_salida

        # Ejecutar Warp (fusiona + recorta en streaming, sin cargar todo en RAM)
        warp_opts = gdal.WarpOptions(**warp_options)
        result_ds = gdal.Warp(ruta_salida, vrt_path, options=warp_opts)

        if result_ds is None:
            raise RuntimeError("❌ Error en gdal.Warp")

        result_ds = None  # Cerrar dataset

        # Limpiar archivos de cutline si se creó (shapefile genera múltiples archivos)
        if cutline_path:
            try:
                # Eliminar todos los archivos del shapefile (.shp, .shx, .dbf, .prj, .cpg)
                base_path = os.path.splitext(cutline_path)[0]
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    file_to_delete = base_path + ext
                    if os.path.exists(file_to_delete):
                        os.unlink(file_to_delete)
            except:
                pass

        if verbose: print(f"F. Fusión y recorte completados en {time.perf_counter() - t0:.2f} s")

    finally:
        # Limpiar VRT temporal
        if os.path.exists(vrt_path):
            try:
                os.unlink(vrt_path)
            except:
                pass

    # === G. Optimizar tipo de datos del archivo final ===
    t0 = time.perf_counter()
    try:
        with rasterio.open(ruta_salida, 'r+') as dst:
            # Leer estadísticas sin cargar todo el raster
            stats = dst.statistics(1)
            max_val = stats.max

            # Determinar dtype óptimo
            current_dtype = dst.dtypes[0]
            optimal_dtype = 'uint8' if max_val <= 255 else 'uint16' if max_val <= 65535 else 'int32'

            if verbose and current_dtype != optimal_dtype:
                print(f"G. Tipo de datos: {current_dtype} (rango 0-{max_val:.0f})")
    except Exception as e:
        if verbose: print(f"⚠️ No se pudo optimizar dtype: {e}")

    if verbose: print(f"G. Escritura verificada en {time.perf_counter() - t0:.2f} s")

    print(f"⏱️ Tiempo total: {time.perf_counter() - start_total:.2f} s")
    if recorte_exacto:
        print("✂️ Recorte exacto aplicado con la forma del shapefile")

def Join_GHSBUILT_Agricultural(raster1_path, raster2_path, salida_path, bloque_tamano=512):
    """
    Incorpora píxeles de cultivos (valor 4) del raster2 al raster1,
    solo donde raster1 tiene valor 0 o nodata.
    Usa WarpedVRT para garantizar compatibilidad espacial.
    Procesa en bloques para minimizar uso de memoria.

    Parámetros:
    ----------
    raster1_path : str
        Ruta al archivo raster GHS-BUILT-C (entrada base).
    raster2_path : str
        Ruta al archivo raster de coberturas (cultivos valor=4).
    salida_path : str
        Ruta al archivo raster de salida.
    bloque_tamano : int
        Tamaño del bloque de procesamiento (en píxeles).
    """

    with rasterio.open(raster1_path) as src1:
        perfil = src1.profile.copy()
        no_data1 = src1.nodata if src1.nodata is not None else 0

        # Configura WarpedVRT para alinear raster2 al raster1 (misma resolución, crs, extensión)
        with rasterio.open(raster2_path) as src2:
            vrt_options = {
                'crs': src1.crs,
                'transform': src1.transform,
                'height': src1.height,
                'width': src1.width,
                'resampling': Resampling.nearest
            }

            with WarpedVRT(src2, **vrt_options) as vrt2:
                # Actualiza perfil de salida
                perfil.update(
                    compress='LZW',
                    tiled=True,
                    blockxsize=bloque_tamano,
                    blockysize=bloque_tamano
                )

                with rasterio.open(salida_path, 'w', **perfil) as dst:
                    # Recorre por ventanas (bloques)
                    for ji, window in src1.block_windows(1):
                        data1 = src1.read(1, window=window)
                        data2 = vrt2.read(1, window=window)

                        # Máscara para reemplazo
                        mascara_remplazo = ((data1 == 0) | (data1 == no_data1)) & (data2 == 4)

                        # Aplica reemplazo
                        data_salida = np.where(mascara_remplazo, 40, data1)

                        # Escribe bloque procesado en disco
                        dst.write(data_salida, 1, window=window)

    print(f"✅ Raster de salida generado (alineado y procesado en bloques): {salida_path}")

#-----------------------------------------------------------------------------------------------------------------------
# Códigos Infiltración y Manning - NbS y BaU
#-----------------------------------------------------------------------------------------------------------------------
def resample_raster(input_raster_path, reference_raster_path, output_raster_path, method=Resampling.nearest):
    with rasterio.open(reference_raster_path) as ref:
        ref_meta = ref.meta.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

        with rasterio.open(input_raster_path) as src:
            data_type = src.dtypes[0]
            count = src.count
            src_nodata = src.nodata

            dst_meta = ref_meta.copy()
            dst_meta.update({
                "driver": "GTiff",
                "dtype": data_type,
                "count": count,
                "crs": ref_crs,
                "transform": ref_transform,
                "width": ref_width,
                "height": ref_height,
                "nodata": src_nodata
            })

            with rasterio.open(output_raster_path, 'w', **dst_meta) as dst:
                for band in range(1, count + 1):
                    reproject(
                        source=rasterio.band(src, band),
                        destination=rasterio.band(dst, band),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=method,
                        src_nodata=src_nodata,
                        dst_nodata=src_nodata
                    )

def get_pixel_size(src):
    """Devuelve (ancho, alto) absolutos del píxel."""
    return abs(src.transform.a), abs(src.transform.e)

def resample_all_to_highest_resolution(
        raster_dict: dict,
        output_dir: str,
        method: Resampling = Resampling.nearest,
        tol: float = 1e-6          # tolerancia para comparar tamaños de píxel
    ):
    """
    Sincroniza resoluciones usando la malla más fina como referencia.
    Solo remuestrea rasters que tengan un tamaño de píxel distinto.

    Returns
    -------
    updated_paths : dict  nombre lógico → ruta final
    ref_path      : str   raster que sirvió de referencia
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ calcular áreas de píxel
    pixel_areas, pixel_sizes = {}, {}
    for key, path in raster_dict.items():
        with rasterio.open(path) as src:
            w, h = get_pixel_size(src)
            pixel_sizes[key] = (w, h)
            pixel_areas[key] = w * h

    # 2️⃣ referencia = raster con área mínima
    best_key = 'lulc_fastflood' #min(pixel_areas, key=pixel_areas.get)
    best_path = raster_dict[best_key]
    ref_w, ref_h = pixel_sizes[best_key]

    print(f"[INFO] Usando '{best_key}' como referencia (resolución más alta).")

    # 3️⃣ remuestrear solo si la resolución difiere
    updated_paths = {}
    for key, path in raster_dict.items():
        w, h = pixel_sizes[key]
        same_res = abs(w - ref_w) < tol and abs(h - ref_h) < tol

        if key == best_key or same_res:
            # ✔️ misma resolución: conservar archivo original
            updated_paths[key] = path
        else:
            out_path = os.path.join(output_dir, f"{key}_resampled.tif")
            resample_raster(path, best_path, out_path, method)
            updated_paths[key] = out_path

    return updated_paths, best_path

def generate_manning_bau(
        lulc_raster_path: str,
        bau_lulc_path: str,
        output_path: str,
        lulc_to_manning: Dict[int, float],
        lulc_fastflood_to_lulc_end: Dict[int, int],
        locked_categories: Set[int],
        manning_base_path: Optional[str] = None,
        use_raster: bool = False
):
    """
    Genera el raster de Manning para el escenario BaU.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Leer rásters principales ────────────────────────────────────────────
    with rasterio.open(lulc_raster_path) as lulc_src, \
            rasterio.open(bau_lulc_path) as bau_src:

        lulc_ff = lulc_src.read(1)
        bau_lulc = bau_src.read(1)

        # 1) manning_inicial ← raster o diccionario
        if use_raster and manning_base_path:
            with rasterio.open(manning_base_path) as man_src:
                manning_inicial = man_src.read(1).astype(np.float32)
        else:
            manning_inicial = np.vectorize(lulc_to_manning.get)(
                lulc_ff).astype(np.float32)

        # 2️⃣ temp Manning: inverso desde BaU → FastFlood → Manning
        man_temp = np.full_like(manning_inicial, np.nan, dtype=np.float32)
        for code in np.unique(bau_lulc):
            mask = bau_lulc == code
            if code == 8:  # sparse vegetation rule
                man_temp[mask] = 0.02
            elif code in lulc_fastflood_to_lulc_end:
                mapped = lulc_fastflood_to_lulc_end[code]
                if mapped in lulc_to_manning:
                    man_temp[mask] = lulc_to_manning[mapped]

        # 3️⃣ final Manning — minimum unless category is locked
        locked_mask = np.isin(lulc_ff, list(locked_categories))
        manning_final = np.where(locked_mask, manning_inicial, np.minimum(manning_inicial, man_temp))

        # ✅ Versión simple y directa
        # Manejar NaNs primero
        no_data_mask = np.isnan(manning_final)
        manning_final[no_data_mask] = manning_inicial[no_data_mask]
        # Forzar el límite superior píxel por píxel
        mask_excede = (manning_final > manning_inicial) & ~np.isnan(manning_inicial)
        manning_final[mask_excede] = manning_inicial[mask_excede]

        # ── Guardar ─────────────────────────────────────────────────────────
        meta = lulc_src.meta.copy()
        meta.update(dtype='float32', count=1)

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(np.round(manning_final, 2).astype(np.float32), 1)

def generate_infiltracion_bau(
        lulc_raster_path: str,                 # LULC FastFlood (alineado)
        lulc_raster_r_path: str,                 # LULC waterproof (alineado)
        bau_lulc_path: str,                    # LULC BaU (alineado)
        infil_base_path: str,                  # raster de infiltración base
        output_path: str,                      # ruta de salida
        lulc_ff_to_wp: Dict[int, int],         # mapeo FastFlood → WaterProof
        infiltration_change: Dict[tuple, int]  # (c_ini, c_bau) → % cambio
    ) -> None:
    """
    Genera el raster de infiltración para el escenario BaU.
    Si (c_ini, c_bau) está en `infiltration_change`, aplica el % de variación
    sobre el valor base de infiltración.

    Todos los rásters deben estar previamente alineados y en la misma resolución.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(lulc_raster_path) as lulc_src, \
         rasterio.open(lulc_raster_r_path) as lulc_r_src,\
         rasterio.open(bau_lulc_path)   as bau_src, \
         rasterio.open(infil_base_path) as infil_src:

        lulc_ff    = lulc_src.read(1)
        lulc_wp    = lulc_r_src.read(1)
        bau_lulc   = bau_src.read(1)
        infil_base = infil_src.read(1).astype(np.float32)

        # 1️⃣  Cobertura actual mapeada (None → np.nan, otypes=FLOAT)
        mapped_lulc_current = np.vectorize(lambda v: lulc_ff_to_wp.get(v, np.nan),
                                           otypes=[float] # <-- fuerza salida float
        )(lulc_ff)
        has_current = ~np.isnan(mapped_lulc_current)

        # 2️⃣  Cobertura inicial: si hay mapeo usa ese valor; de lo contrario, el original
        cobertura_ini = np.where(has_current, mapped_lulc_current, lulc_ff ).astype(np.uint8)
        # np.where(~np.isnan(mapped_current),
        #                          mapped_current,
        #                          lulc_ff).astype(np.uint16)

        # 3️⃣  Aplicar cambios porcentuales
        infil_final = infil_base.copy()
        rows, cols  = infil_base.shape

        for i in range(rows):
            for j in range(cols):
                key = (int(lulc_wp[i, j]), int(bau_lulc[i, j]))
                if key in infiltration_change:
                    factor = abs(infiltration_change[key] / 100.0)
                    if factor < 1:
                        factor2 = (1-factor)
                    else:
                        factor2 = 0
                    infil_final[i, j] = infil_base[i, j]*factor2

        # for i in range(rows):
        #     for j in range(cols):
        #         key = (int(cobertura_ini[i, j]), int(bau_lulc[i, j]))
        #         if key in infiltration_change:
        #             factor = infiltration_change[key] / 100.0
        #             infil_final[i, j] = infil_base[i, j] * (1 + factor)

        for i in range(rows):
            for j in range(cols):
                key = (int(cobertura_ini[i, j]), int(bau_lulc[i, j]))
                if cobertura_ini[i, j] != bau_lulc[i, j] and key in infiltration_change:
                    factor = abs(infiltration_change[key] / 100.0)
                    if factor < 1:
                        factor2 = (1-factor)
                    else:
                        factor2 = 0
                    infil_final[i, j] = infil_base[i, j]*factor2

        # 4️⃣  Check - El BaU no puede ser mejor que el Current
        infil_final = np.minimum(infil_final, infil_base)

        # 4️⃣  Guardar raster resultante
        meta = infil_src.meta.copy()
        meta.update(dtype='float32', count=1)

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(infil_final.astype(np.float32), 1)


def generate_nbs_manning(
        lulc_raster: str,                     # Raster LULC base (FastFlood)
        lulc_raster_r_path: str,              # Raster LULC Waterproof (LULC "final")
        nbs_raster: str,                      # Raster con portafolio de NbS
        bau_raster: str,                      # Raster de Manning BaU (sin NbS)
        manning_current_path: str,            # Raster de Manning del escenario Current
        output_path: str,                     # Ruta de salida
        lulc_mapping: Dict[int, float],       # LULC (FastFlood) -> Manning base
        lulc_fastflood_to_lulc_end: Dict[int, int],  # LULC FastFlood -> LULC final
        locked_categories: Set[int],          # LULC bloqueados (no modificar)
        manning_dict: Dict[Tuple[int, int], float],  # (NbS, LULC_final) -> Manning
        manning_base_path: Optional[str] = None,     # Raster alterno de Manning base
        use_raster: bool = False,             # Usar manning_base_path en vez de lulc_mapping
        default_manning: float = np.nan,      # Valor por defecto si falta mapping de Manning
        default_lulc_final: int = -9999       # Marcador si falta mapping FastFlood->Final
):
    """
    Genera un raster de Manning a partir del escenario BaU y aplica cambios
    solo donde hay NbS válidas y no bloqueadas, usando manning_dict.

    Notas prácticas:
    - Evita np.vectorize; usa mapeos por np.unique (más seguro y claro).
    - Maneja claves faltantes con valores por defecto.
    - Respeta categorías bloqueadas y nodata de NbS.
    - Verifica que pixeles con NbS no empeoren vs escenario Current.
    """

    def apply_dict_map(arr: np.ndarray,
                       dct: Dict[int, float | int],
                       default_val,
                       out_dtype):
        """
        Aplica un diccionario a un array entero sin usar loops por pixel.
        Cualquier clave faltante se mapea a `default_val`.
        """
        u, inv = np.unique(arr, return_inverse=True)
        mapped_u = np.full(u.shape, default_val, dtype=out_dtype)
        # Rellenar solo donde la clave exista
        for i, key in enumerate(u):
            v = dct.get(int(key), None)
            if v is not None:
                mapped_u[i] = v
        return mapped_u[inv].reshape(arr.shape)

    with rasterio.open(lulc_raster) as lulc_src, \
         rasterio.open(lulc_raster_r_path) as lulc_src_r, \
         rasterio.open(nbs_raster) as nbs_src, \
         rasterio.open(bau_raster) as bau_src:

        # Lecturas
        lulc_ff   = lulc_src.read(1)       # LULC FastFlood
        lulc_wp   = lulc_src_r.read(1)     # LULC final (Waterproof)
        nbs_data  = nbs_src.read(1)
        bau_data  = bau_src.read(1).astype(np.float32)

        # Raster final parte de BaU (regla: celdas sin NbS quedan igual)
        manning_final = bau_data.copy()

        # Definir presencia de NbS de manera robusta
        nbs_nodata = nbs_src.nodata
        if nbs_nodata is None:
            # Si no hay nodata, considera NbS presentes si > 0 (ajusta a tu codificación)
            has_nbs = nbs_data > 0
        else:
            has_nbs = nbs_data != nbs_nodata

        # Categorías bloqueadas (no tocar)
        locked_mask = np.isin(lulc_ff, list(locked_categories))
        not_locked  = ~locked_mask

        # Manning inicial:
        if use_raster and manning_base_path:
            with rasterio.open(manning_base_path) as man_src:
                manning_inicial = man_src.read(1).astype(np.float32)
        else:
            # Desde lulc_mapping (LULC FastFlood -> manning base)
            manning_inicial = apply_dict_map(
                arr=lulc_ff,
                dct=lulc_mapping,
                default_val=np.float32(default_manning),
                out_dtype=np.float32
            )

        # TEMP: valores candidatos desde manning_dict[(NbS, LULC_final)]
        # Primero, mapea FastFlood -> LULC_final donde haga falta
        # (si ya tienes lulc_wp como "final", úsalo directo; si no, usa el mapping)
        # Aquí usamos el LULC "final" proveniente de lulc_wp; si necesitas forzar
        # el mapping de FastFlood->Final, aplica apply_dict_map(lulc_ff, ...)
        lulc_final = lulc_wp.astype(np.int32)

        # Construir temp_data solo donde haya NbS y no esté bloqueado
        temp_data = np.full(nbs_data.shape, np.nan, dtype=np.float32)
        candidate_mask = has_nbs & not_locked

        if np.any(candidate_mask):
            nbs_vals   = nbs_data[candidate_mask].astype(np.int32)
            lulc_vals  = lulc_final[candidate_mask].astype(np.int32)
            temp_flat  = temp_data[candidate_mask]

            # Rellenar temp_flat con los valores de manning_dict cuando exista la clave
            # (loop sobre candidatos; suele ser suficientemente rápido y claro)
            for i, (nbs_val, lulc_val) in enumerate(zip(nbs_vals, lulc_vals)):
                v = manning_dict.get((int(nbs_val), int(lulc_val)), None)
                if v is not None:
                    temp_flat[i] = np.float32(v)

            # Escribir de regreso
            temp_data[candidate_mask] = temp_flat

        # REGLA 1: usar temp si mejora y hay NbS (y no está bloqueado)
        # Nota: "mejorar" = temp > manning_inicial; ajusta el criterio si hace falta.
        valid_temp = ~np.isnan(temp_data)
        cond_temp_greater = valid_temp & candidate_mask & (temp_data > manning_inicial)
        manning_final[cond_temp_greater] = temp_data[cond_temp_greater]

        # REGLA 2: donde hay NbS y temp válido pero NO mejora, intenta tabla con (NbS, LULC_final)
        remaining_mask = candidate_mask & valid_temp & (~cond_temp_greater)
        if np.any(remaining_mask):
            nbs_flat   = nbs_data[remaining_mask].astype(np.int32)
            lulc_flat  = lulc_final[remaining_mask].astype(np.int32)
            base_flat  = manning_inicial[remaining_mask].astype(np.float32)

            final_values = base_flat.copy()
            for i, (nbs_val, lulc_val) in enumerate(zip(nbs_flat, lulc_flat)):
                v = manning_dict.get((int(nbs_val), int(lulc_val)), None)
                if v is not None:
                    final_values[i] = np.float32(v)

            manning_final[remaining_mask] = final_values

        # REGLA 3: Verificación vs escenario Current
        # En pixeles con NbS (no bloqueados), si manning_final < current, usar current
        with rasterio.open(manning_current_path) as curr_src:
            manning_current = curr_src.read(1).astype(np.float32)

        condition_worse = has_nbs & not_locked & (manning_final < manning_current)
        manning_final[condition_worse] = manning_current[condition_worse]

        # (Sin NbS) o (bloqueados) -> mantienen BaU ya copiado en manning_final

        # Guardar
        meta = lulc_src.meta.copy()
        meta.update(dtype="float32", count=1)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(manning_final.astype(np.float32), 1)

def generate_nbs_infiltration(
    lulc_raster: str,  # LULC FastFlood (alineado)
    lulc_raster_r_path: str,  # LULC waterproof (alineado)
    nbs_raster: str,  # Portafolio (alineado)
    infiltration_base_path: str,  # Raster base de infiltración (sin NbS)
    bau_raster: str,  # Raster BaU de infiltración
    output_path: str,
    lulc_fastflood_to_lulc_end: Dict[int, int],
    locked_categories: Set[int],
    infiltration_dict: Dict[Tuple[int, int], float]  # Valores entre 1 y 100 (porcentaje)
):
    """Genera un raster de infiltración final aplicando aumentos porcentuales
    definidos por NbS, manteniendo valores de BaU donde no hay NbS.
    """
    with rasterio.open(lulc_raster) as lulc_src, \
         rasterio.open(lulc_raster_r_path) as lulc_src_r, \
         rasterio.open(nbs_raster) as nbs_src, \
         rasterio.open(infiltration_base_path) as infil_src, \
         rasterio.open(bau_raster) as bau_src:

        lulc_data    = lulc_src.read(1)
        lulc_data_wp = lulc_src_r.read(1)
        nbs_data     = nbs_src.read(1)
        base_data    = infil_src.read(1).astype(np.float32)  # valor actual de infiltración
        bau_data     = bau_src.read(1).astype(np.float32)    # infiltración sin NbS (referencia)

        # Inicializar resultado final como copia exacta del BaU
        infiltration_final = bau_data.copy()

        # Crear raster temporal con mejoras basadas en porcentaje
        infiltration_temp = np.full(nbs_data.shape, np.nan, dtype=np.float32)
        flat_nbs          = nbs_data.ravel()
        flat_lulc         = lulc_data_wp.ravel()
        flat_base         = base_data.ravel()
        infiltration_out  = infiltration_temp.ravel()

        for idx in range(flat_nbs.size):
            key = (flat_nbs[idx], flat_lulc[idx])
            if key in infiltration_dict:
                percentage = infiltration_dict[key] / 100.0  # convertir de 1-100 a 0.01-1.0
                infiltration_out[idx] = flat_base[idx] * (1 + percentage)

        #'''
        temp_data = infiltration_out.reshape(nbs_data.shape)

        # Máscaras
        has_nbs     = ~np.isnan(nbs_data)
        locked_mask = np.isin(lulc_data, list(locked_categories))
        not_locked  = ~locked_mask
        valid_temp  = ~np.isnan(temp_data)

        # Regla 1: aplicar aumento si mejora (más infiltración)
        condition_temp_greater = (temp_data > base_data) & not_locked & valid_temp & has_nbs
        infiltration_final[condition_temp_greater] = temp_data[condition_temp_greater]

        # Regla 2: si no mejora, intentar con LULC final
        remaining_mask = not_locked & valid_temp & ~condition_temp_greater & has_nbs
        if np.any(remaining_mask):
            nbs_flat         = nbs_data[remaining_mask].astype(np.uint8)
            lulc_flat        = lulc_data[remaining_mask].astype(np.uint8)
            base_flat        = base_data[remaining_mask]
            mapped_lulc_flat = np.vectorize(lulc_fastflood_to_lulc_end.get)(lulc_flat)

            final_values = base_flat.copy()
            for idx, (nbs_val, lulc_val) in enumerate(zip(nbs_flat, mapped_lulc_flat)):
                key = (nbs_val, lulc_val)
                if key in infiltration_dict:
                    percentage = infiltration_dict[key] / 100.0
                    final_values[idx] = base_flat[idx] * (1 + percentage)

            infiltration_final[remaining_mask] = final_values
        #'''

        # Guardar raster final
        meta = lulc_src.meta.copy()
        meta.update(dtype="float32", count=1)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(infiltration_final.astype(np.float32), 1)

def red_params_from_json(json_path):
    # Extraer códigos de NbS desde claves como "5-Conservation"
    def extract_codes_from_keys(data_dict, separator="-"):
        return {k: int(k.split(separator)[0]) for k in data_dict.keys()}
    # Leer archivo
    with open(json_path, "r") as f:
        config = json.load(f)

    inputs_paths = {
        "lulc_fastflood"   : config["FastFloodLulcPath"],
        "waterproof_current": config["CurrentLulcPath"],
        "bau_lulc"         : config["BauLulcPath"],
        "infiltracion_base": config["InfiltrationPath"],
        "manning_base"     : config["ManningPath"],
        "portafolio"       : config["PortfolioPath"]

    }

    path_out = config["ProjectPath"]

    # === 1. lulc_to_manning (desde LulcParams) ===
    lulc_to_manning = {
        int(k.split("-")[0]): v
        for k, v in config["LulcParams"]["Manning"].items()
    }

    # === 2. manning_dict e infiltracion_dict (desde NbS) ===
    manning_dict = {}
    infiltracion_dict = {}

    nbs_codes = extract_codes_from_keys(config["NbS"])
    for nbs_key, covers in config["NbS"].items():
        nbs_code = nbs_codes[nbs_key]
        for lulc_label, values in covers.items():
            # lulc_code = lulc_label_to_code.get(lulc_label)
            lulc_code = int(lulc_label.split("-")[0])
            if lulc_code is not None:
                if "Manning" in values:
                    manning_dict[(nbs_code, lulc_code)] = values["Manning"]
                if "Infiltration" in values:
                    infiltracion_dict[(nbs_code, lulc_code)] = values["Infiltration"]

    # === 3. infiltration_change_dict (desde InfiltrationChange) ===
    infiltration_change_dict = {}
    for key, val in config["LulcParams"]["InfiltrationChange"].items():
        parts = key.split(",")
        ini_label = parts[0].split("-")[0]
        fin_label = parts[1].split("-")[0]
        ini_code = int(ini_label)
        fin_code = int(fin_label)
        if ini_code is not None and fin_code is not None:
            infiltration_change_dict[(ini_code, fin_code)] = val

    return lulc_to_manning,manning_dict, infiltracion_dict, infiltration_change_dict, inputs_paths,path_out

def Indicators_BaU_NBS(PathProject):
    # ------------------------------------------------------------------------------------------------------------------
    # Indicadores para carbono
    # ------------------------------------------------------------------------------------------------------------------
    # Name Columns
    NameCol = ['WC (Ton)']

    BaU = pd.read_csv(os.path.join(PathProject, 'in', 'INPUTS_CARBON_BaU.csv'), usecols=NameCol)
    NBS = pd.read_csv(os.path.join(PathProject, 'in', 'INPUTS_CARBON_NbS.csv'), usecols=NameCol)

    BaU = BaU.drop([0])
    NBS = NBS.drop([0])

    # Indicators
    Indicators = ((NBS - BaU)/BaU)*100

    Tmp = [[Indicators['WC (Ton)'].max()]]

    Final_In = pd.DataFrame(data=Tmp, columns=NameCol)

    Indicators = np.round(Indicators,2)
    Final_In   = np.round(Final_In,2)

    Indicators.to_csv(os.path.join(PathProject, 'out', 'OUTPUTS_Indicators_TimeSeries.csv'), index_label='Time')
    Final_In.to_csv(os.path.join(PathProject, 'out', 'OUTPUTS_Max_Indicators.csv'), index_label='Time')

    # Integrated Indicator
    Indicators = ((NBS - BaU)/BaU)*100

    # Indicador total - Curvas integradas
    Int_NBS     = np.trapezoid(NBS, dx=1.0, axis=0)
    Int_BaU     = np.trapezoid(BaU, dx=1.0, axis=0)
    Tmp         = ((Int_NBS - Int_BaU) / Int_BaU) * 100

    Final_In = pd.DataFrame(data=Tmp, columns=NameCol)

    Indicators = np.round(Indicators,2)
    Indicators = (Indicators*0 + 1)*Tmp
    Final_In   = np.round(Final_In,2)

    Indicators.to_csv(os.path.join(PathProject,'out','OUTPUTS-Indicators_TimeSeries_Total.csv'), index_label='Time')
    Final_In.to_csv(os.path.join(PathProject,'out','OUTPUTS-Max_Indicators_Total.csv'), index_label='Time')

    # ------------------------------------------------------------------------------------------------------------------
    # Indicadores para inundación
    # ------------------------------------------------------------------------------------------------------------------
    BaU = pd.read_csv(os.path.join(PathProject, 'in', 'INPUTS_FLOOD_BaU.csv'), index_col=0)
    NBS = pd.read_csv(os.path.join(PathProject, 'in', 'INPUTS_FLOOD_NbS.csv'), index_col=0)

    Indicators = [
                  ((NBS['Peak Descharge [m3/s]'][10] - BaU['Peak Descharge [m3/s]'][10]) / BaU['Peak Descharge [m3/s]'][10]) * 100,
                  ((NBS['Peak Descharge [m3/s]'][100] - BaU['Peak Descharge [m3/s]'][100]) / BaU['Peak Descharge [m3/s]'][100]) * 100,
                  ((NBS['Flood Area [ha]'].sum() - BaU['Flood Area [ha]'].sum()) / BaU['Flood Area [ha]'].sum()) * 100,
                  Tmp[0]]

    Results = pd.DataFrame(data=np.array([Indicators]),
                           columns=['Peak Descharge TR 10', 'Peak Descharge TR 100','Flooded Area','Carbon'],index=[0])

    Results.to_csv(os.path.join(PathProject, 'out', 'OUTPUTS-Indicators.csv'),index=False)

"""
# ------------------------------------------------------------------------------------------------------------------
Detecta y reconstruye hidrogramas SCS dañados por errores numéricos.

Uso:
    correct_hydrographs(folder, dem_path)

Parámetros:
    folder       : ruta a la carpeta con los CSV y rasters de descarga
    dem_path     : ruta al DEM de la cuenca (se procesa solo si no hay TRs válidos)
    min_valid_trs: TRs válidos mínimos para usar regresión (default 2)
    n_default    : parámetro n fijo cuando no hay TRs válidos (default 13.5)

Retorna None. Reemplaza in-place los TR invalidos y escribe diagnosticos en '_diagnostics/'.
# ------------------------------------------------------------------------------------------------------------------
"""

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds as _window_from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

SCENARIOS = ['BaU', 'Current', 'NbS']
TRS = [2, 5, 10, 20, 40, 50, 100, 200, 500, 1000]

COL_TIME = 'time (hour)'
COL_Q    = 'discharge (m3/s)'


# ---------------------------------------------------------------------------
# Fórmulas y estadísticas
# ---------------------------------------------------------------------------

def scs_q(t: np.ndarray, qp: float, tp: float, n: float) -> np.ndarray:
    """
    Hidrograma Unitario Sintético SCS normalizado (Williams & Hann 1973).
    Garantiza q = qp exactamente en t = tp.

    t  : array de tiempo (h)
    qp : caudal pico (m³/s)
    tp : tiempo al pico (h)
    n  : parámetro de forma (n > 1)
    Returns array de caudal (m³/s).
    """
    r = np.clip(t / tp, 1e-12, None)
    return qp * (r ** (n - 1.0)) * np.exp((n - 1.0) * (1.0 - r))


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coeficiente de determinación R²."""
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 0.0 if ss_tot == 0 else float(1.0 - np.sum((y - y_hat) ** 2) / ss_tot)


def hydrograph_stats(df: pd.DataFrame) -> tuple:
    """
    Extrae tp, Qp y duración de un hidrograma.

    df : DataFrame con columnas [time (hour), discharge (m3/s)]
    Returns (tp_h, qp_m3s, duration_h).
    """
    t = df[COL_TIME].values
    q = df[COL_Q].values
    idx = int(np.argmax(q))
    tp, qp = float(t[idx]), float(q[idx])
    above = np.where(q > 0.01 * qp)[0]
    duration = float(t[above[-1]]) if above.size else float(t[-1])
    return tp, qp, duration


def has_plateau(q: np.ndarray, tol_rel: float = 1e-4) -> bool:
    """True si el pico persiste más de 2 pasos de tiempo consecutivos."""
    qp = q.max()
    at_peak = np.abs(q - qp) < tol_rel * qp
    streak = 0
    for v in at_peak:
        streak = streak + 1 if v else 0
        if streak > 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Validación
# ---------------------------------------------------------------------------

def validate_triplet(stats: dict, raw_q: dict) -> tuple:
    """
    Valida reglas de orden y unicidad para un TR.

    stats : {escenario: (tp, qp, duration)}
    raw_q : {escenario: array de caudal}
    Returns (is_valid: bool, reason: str).
    """
    tp = {sc: stats[sc][0] for sc in SCENARIOS}
    qp = {sc: stats[sc][1] for sc in SCENARIOS}
    reasons = []

    if tp['BaU'] >= tp['Current']:
        reasons.append(
            f"tp_BaU({tp['BaU']:.2f}h) >= tp_Current({tp['Current']:.2f}h)"
        )
    if tp['BaU'] >= tp['NbS']:
        reasons.append(
            f"tp_BaU({tp['BaU']:.2f}h) >= tp_NbS({tp['NbS']:.2f}h)"
        )
    if tp['NbS'] >= tp['Current']:
        reasons.append(
            f"tp_NbS({tp['NbS']:.2f}h) >= tp_Current({tp['Current']:.2f}h)"
        )

    if len({round(v, 3) for v in qp.values()}) < 3:
        reasons.append(
            "Qp no únicos: " + ", ".join(f"{sc}={qp[sc]:.3f}" for sc in SCENARIOS)
        )

    for sc in SCENARIOS:
        if has_plateau(raw_q[sc]):
            reasons.append(f"Plateau en pico: {sc}")

    return len(reasons) == 0, "; ".join(reasons)


# ---------------------------------------------------------------------------
# Calibración de n
# ---------------------------------------------------------------------------
def calibrate_n(t: np.ndarray, q_obs: np.ndarray,
                qp: float, tp: float, n0: float = 3.0) -> tuple:
    """
    Calibra n del HUS SCS vía Nelder-Mead minimizando SSE.

    t, q_obs : serie temporal observada
    qp, tp   : caudal y tiempo al pico fijos
    n0       : valor inicial de n
    Returns (n_optimo, RMSE).
    """
    def objective(params):
        n = params[0]
        if n <= 1.001:
            return 1e12
        return float(np.sum((scs_q(t, qp, tp, n) - q_obs) ** 2))

    res = minimize(
        objective, [n0], method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000, 'disp': False}
    )
    n_opt = max(float(res.x[0]), 1.001)
    rmse = float(np.sqrt(np.mean((scs_q(t, qp, tp, n_opt) - q_obs) ** 2)))
    return n_opt, rmse


# ---------------------------------------------------------------------------
# Regresión
# ---------------------------------------------------------------------------

def fit_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Ajusta regresiones lineal, potencial y exponencial. Retorna la mejor por R².

    x, y : arrays 1D de datos
    Returns dict con: type, params, r2, predict (callable x -> y).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    best: dict = {'type': None, 'params': [], 'r2': -np.inf, 'predict': None}

    def _try(label, params, y_hat, fn):
        r2 = r2_score(y, y_hat)
        if r2 > best['r2']:
            best.update(type=label, params=list(params), r2=r2, predict=fn)

    # Linear: y = a*x + b
    a, b = np.polyfit(x, y, 1)
    _try('linear', [a, b], a * x + b,
         lambda xx, a=a, b=b: a * np.asarray(xx, float) + b)

    # Power: y = a*x^b  (x>0, y>0)
    if np.all(x > 0) and np.all(y > 0):
        bp, la = np.polyfit(np.log(x), np.log(y), 1)
        ap = np.exp(la)
        _try('power', [ap, bp], ap * x ** bp,
             lambda xx, a=ap, b=bp: a * np.asarray(xx, float) ** b)

    # Exponential: y = a*exp(b*x)  (y>0)
    if np.all(y > 0):
        be, la = np.polyfit(x, np.log(y), 1)
        ae = np.exp(la)
        _try('exponential', [ae, be], ae * np.exp(be * x),
             lambda xx, a=ae, b=be: a * np.exp(b * np.asarray(xx, float)))

    return best


# ---------------------------------------------------------------------------
# Lectura de rasters
# ---------------------------------------------------------------------------

def get_qpeak_raster(folder: str, scenario: str, tr: int) -> float:
    """
    Lee el maximo pixel valido del raster de caudal pico.

    Returns Qp (m3/s).
    """
    path = os.path.join(folder, f"Qpeak_{scenario}_TR-{tr}.tif")
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nd = src.nodata
        if nd is not None:
            data = data[data != nd]
        data = data[np.isfinite(data) & (data > 0)]
        return float(data.max())


# ---------------------------------------------------------------------------
# Tiempo de concentración (D8 disk-based + Kirpich) — mínima huella de RAM
# ---------------------------------------------------------------------------

def compute_tc_kirpich(dem_path: str, mask_raster: str) -> tuple:
    """
    Estima tiempo de concentracion (h) por Kirpich.
    L = distancia maxima entre pixeles del borde de la mascara de cuenca.
    S = (Hmax - Hmin) / L

    dem_path    : DEM (puede ser mas grande que mask_raster)
    mask_raster : raster Qpeak — define extension y mascara de cuenca
    Returns (tc, L_m, S).
    """
    from scipy.ndimage import binary_erosion
    from scipy.spatial import ConvexHull

    # --- Qpeak: bounds y mascara ---
    with rasterio.open(mask_raster) as msk:
        bounds   = msk.bounds
        msk_data = msk.read(1).astype(np.float32)
        nd_mask  = msk.nodata
        msk_tf   = msk.transform
        msk_crs  = msk.crs

    # --- DEM clip ventana sobre extent del Qpeak ---
    with rasterio.open(dem_path) as dem_src:
        win     = _window_from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            transform=dem_src.transform,
        )
        dem_raw = dem_src.read(1, window=win).astype(np.float32)
        dem_tf  = dem_src.window_transform(win)
        dem_nd  = dem_src.nodata
        dem_crs = dem_src.crs
        res     = float(abs(float(dem_tf.a)))

    rows, cols = dem_raw.shape

    # --- Reproyectar mascara Qpeak al grid del DEM ---
    valid_src = np.ones_like(msk_data, dtype=np.int32)
    if nd_mask is not None:
        valid_src[msk_data == nd_mask] = 0
    valid_src[~np.isfinite(msk_data) | (msk_data <= 0)] = 0
    del msk_data

    ws = np.zeros((rows, cols), dtype=np.int32)
    reproject(
        source=valid_src, destination=ws,
        src_transform=msk_tf, src_crs=msk_crs,
        dst_transform=dem_tf, dst_crs=dem_crs,
        resampling=Resampling.nearest,
    )
    mask = ws.astype(bool)
    del ws, valid_src

    # --- DEM enmascarado ---
    if dem_nd is not None:
        dem_raw[dem_raw == np.float32(dem_nd)] = np.nan
    dem_raw[~mask] = np.nan

    # --- Hmax, Hmin ---
    valid_elev = dem_raw[mask & np.isfinite(dem_raw)]
    H_max = float(valid_elev.max())
    H_min = float(valid_elev.min())
    del valid_elev, dem_raw

    # --- Borde de la mascara ---
    eroded = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    border = mask & ~eroded
    del eroded

    br, bc = np.where(border)
    del border

    # Coordenadas fisicas (metros) en columna=x, fila=y
    pts = np.column_stack([bc.astype(np.float64) * res,
                           br.astype(np.float64) * res])

    # Convex hull reduce a O(sqrt(n)) vertices; diametro siempre en hull
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        pts  = pts[hull.vertices]

    # Par mas distante entre vertices del hull
    max_d2 = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d2 = (pts[i, 0] - pts[j, 0])**2 + (pts[i, 1] - pts[j, 1])**2
            if d2 > max_d2:
                max_d2 = d2
    L = max(float(np.sqrt(max_d2)), res)

    S  = max((H_max - H_min) / L, 0.001)
    tc = 0.000325 * (L ** 0.77) * (S ** (-0.385))
    return tc, L, S


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_hydrographs(diag: str, all_data: dict, reconstructed: dict,
                      valid_trs: list, invalid_trs: list) -> None:
    plots_dir = os.path.join(diag, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    tr_colors = plt.cm.get_cmap('tab10')

    for sc in SCENARIOS:
        fig, ax = plt.subplots(figsize=(12, 5))

        for i, tr in enumerate(valid_trs):
            df = all_data[sc][tr]
            ax.plot(df[COL_TIME], df[COL_Q], color=tr_colors(i),
                    lw=1.5, label=f'TR-{tr} original')

        for i, tr in enumerate(invalid_trs):
            color = tr_colors(len(valid_trs) + i)
            df_dmg = all_data[sc][tr]
            ax.plot(df_dmg[COL_TIME], df_dmg[COL_Q],
                    color='grey', lw=0.8, ls=':', alpha=0.6,
                    label=f'TR-{tr} dañado')
            if (sc, tr) in reconstructed:
                t_r, q_r = reconstructed[(sc, tr)]
                ax.plot(t_r, q_r, color=color,
                        lw=1.5, ls='--', label=f'TR-{tr} reconstruido')

        ax.set_xlabel('Tiempo (h)')
        ax.set_ylabel('Caudal (m3/s)')
        ax.set_title(f'Hidrogramas — {sc}  (gris punteado = original dañado)')
        ax.legend(fontsize=6, ncol=3, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'hydrographs_{sc}.png'), dpi=150)
        plt.close(fig)


def _plot_regressions(diag: str, regressions: dict, reg_data: dict) -> None:
    plots_dir = os.path.join(diag, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fn_keys = ['f1_tp', 'f2_duration', 'f3_n']
    ylabels = ['tp (h)', 'Duracion (h)', 'n']
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))

    for j, sc in enumerate(SCENARIOS):
        qp_pts = np.array([d['qp'] for d in reg_data[sc]])
        y_pts = {
            'f1_tp':       np.array([d['tp']       for d in reg_data[sc]]),
            'f2_duration': np.array([d['duration']  for d in reg_data[sc]]),
            'f3_n':        np.array([d['n']         for d in reg_data[sc]]),
        }
        for i, (fk, yl) in enumerate(zip(fn_keys, ylabels)):
            ax = axes[i][j]
            reg = regressions[sc][fk]
            x_line = np.linspace(qp_pts.min() * 0.8, qp_pts.max() * 1.2, 200)
            ax.scatter(qp_pts, y_pts[fk], color='black', zorder=5, s=40)
            ax.plot(x_line, reg['predict'](x_line), 'b-', lw=1.5)
            ax.set_title(
                f'{sc} — {fk}\n({reg["type"]}, R²={reg["r2"]:.3f})', fontsize=8
            )
            ax.set_xlabel('Qp (m3/s)', fontsize=7)
            ax.set_ylabel(yl, fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'regressions.png'), dpi=150)
    plt.close(fig)


def correct_hydrographs(
    folder: str,
    dem_path: str,
    min_valid_trs: int = 2,
    n_default: float = 13.5,
) -> None:
    """
    Detecta y reconstruye hidrogramas SCS dañados por errores numéricos.

    folder       : ruta con TS_Q_*.csv y Qpeak_*.tif
    dem_path     : ruta al DEM de la cuenca (se procesa solo si no hay TRs válidos)
    min_valid_trs: TRs válidos mínimos para usar regresión (default 2)
    n_default    : parámetro n fijo cuando no hay TRs válidos (default 13.5)
    Escribe:
        TS_Q_{SC}_TR-{TR}.csv  (reemplaza originals de TR invalidos)
        _diagnostics/_originals_backup/
        _diagnostics/01_validation.csv
        _diagnostics/02_n_calibrated.csv   (casos 2 y normal)
        _diagnostics/03_regressions.csv    (solo caso normal)
        _diagnostics/04_reconstruction.csv
        _diagnostics/plots/
    """
    diag = os.path.join(folder, '_diagnostics')
    os.makedirs(diag, exist_ok=True)
    backup = os.path.join(diag, '_originals_backup')
    os.makedirs(backup, exist_ok=True)
    for sc in SCENARIOS:
        for tr in TRS:
            name = f'TS_Q_{sc}_TR-{tr}.csv'
            dst = os.path.join(backup, name)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(folder, name), dst)

    # ------------------------------------------------------------------
    # 1. Cargar todos los CSV
    # ------------------------------------------------------------------
    all_data: dict = {sc: {} for sc in SCENARIOS}
    for sc in SCENARIOS:
        for tr in TRS:
            path = os.path.join(folder, f'TS_Q_{sc}_TR-{tr}.csv')
            all_data[sc][tr] = pd.read_csv(path)

    n_points = max(len(all_data[sc][tr]) for sc in SCENARIOS for tr in TRS)

    # ------------------------------------------------------------------
    # 2. Calcular estadísticas y validar por TR
    # ------------------------------------------------------------------
    stats: dict = {}
    val_rows = []
    valid_trs, invalid_trs = [], []

    for tr in TRS:
        stats[tr] = {}
        raw_q = {}
        for sc in SCENARIOS:
            df = all_data[sc][tr]
            tp, qp, dur = hydrograph_stats(df)
            stats[tr][sc] = (tp, qp, dur)
            raw_q[sc] = df[COL_Q].values

        is_valid, reason = validate_triplet(stats[tr], raw_q)
        (valid_trs if is_valid else invalid_trs).append(tr)

        for sc in SCENARIOS:
            tp, qp, dur = stats[tr][sc]
            t_last = float(all_data[sc][tr][COL_TIME].iloc[-1])
            val_rows.append({
                'TR': tr, 'scenario': sc,
                'tp_h': round(tp, 4),
                'Qp_m3s': round(qp, 4),
                'duration_h': round(dur, 4),
                'time_end_h': round(t_last, 4),
                'valid': is_valid,
                'reason': reason,
            })

    pd.DataFrame(val_rows).to_csv(
        os.path.join(diag, '01_validation.csv'), index=False
    )
    print(f"TRs válidos   : {valid_trs}")
    print(f"TRs inválidos : {invalid_trs}")

    # ------------------------------------------------------------------
    # Case 1: todos válidos
    # ------------------------------------------------------------------
    if not invalid_trs:
        print("Todos los hidrogramas son válidos. Sin cambios.")
        return

    # ------------------------------------------------------------------
    # Determinar modo de reconstrucción
    # ------------------------------------------------------------------
    fixed_params = None   # {sc: (tp, dur, n)} — None = usar regresiones
    regressions = None
    reg_data = None

    if not valid_trs:
        # Case 3: ningún TR válido — parámetros sintéticos desde DEM
        mask_raster = os.path.join(folder, f'Qpeak_{SCENARIOS[0]}_TR-{TRS[0]}.tif')
        print("Sin TRs válidos. Calculando tc desde DEM...")
        tc, L_m, S_mm = compute_tc_kirpich(dem_path, mask_raster)
        tp_s  = 0.6 * tc
        dur_s = 5.0 * tp_s
        fixed_params = {sc: (tp_s, dur_s, n_default) for sc in SCENARIOS}
        print(f"  tc={tc:.2f}h  tp={tp_s:.2f}h  dur={dur_s:.2f}h  n={n_default}")
        pd.DataFrame([{
            'L_km':       round(L_m / 1000.0, 4),
            'slope_m_m':  round(float(S_mm), 6),
            'tc_h':       round(tc, 4),
            'tp_h':       round(tp_s, 4),
        }]).to_csv(os.path.join(diag, '05_basin_geometry.csv'), index=False)

    elif len(valid_trs) < min_valid_trs:
        # Case 2: pocos TRs válidos — calibrar y promediar parámetros por escenario
        fixed_params = {}
        calib_rows = []
        for sc in SCENARIOS:
            tps, durs, ns = [], [], []
            for tr in valid_trs:
                df = all_data[sc][tr]
                t = df[COL_TIME].values
                q_obs = df[COL_Q].values
                tp, qp, dur = stats[tr][sc]
                n_opt, rmse = calibrate_n(t, q_obs, qp, tp)
                tps.append(tp)
                durs.append(dur)
                ns.append(n_opt)
                calib_rows.append({
                    'TR': tr, 'scenario': sc,
                    'Qp_m3s': round(qp, 4),
                    'tp_h': round(tp, 4),
                    'n_calibrated': round(n_opt, 6),
                    'RMSE_m3s': round(rmse, 6),
                })
            fixed_params[sc] = (
                float(np.mean(tps)),
                float(np.mean(durs)),
                float(np.mean(ns)),
            )
        pd.DataFrame(calib_rows).to_csv(
            os.path.join(diag, '02_n_calibrated.csv'), index=False
        )
        print(
            f"TRs válidos ({len(valid_trs)}) < min_valid_trs ({min_valid_trs}). "
            "Usando parámetros fijos por escenario."
        )

    else:
        # Normal: calibrar n + regresiones
        calib_rows = []
        reg_data = {sc: [] for sc in SCENARIOS}

        for sc in SCENARIOS:
            for tr in valid_trs:
                df = all_data[sc][tr]
                t = df[COL_TIME].values
                q_obs = df[COL_Q].values
                tp, qp, dur = stats[tr][sc]
                n_opt, rmse = calibrate_n(t, q_obs, qp, tp)
                reg_data[sc].append({'qp': qp, 'tp': tp, 'duration': dur, 'n': n_opt})
                calib_rows.append({
                    'TR': tr, 'scenario': sc,
                    'Qp_m3s': round(qp, 4),
                    'tp_h': round(tp, 4),
                    'n_calibrated': round(n_opt, 6),
                    'RMSE_m3s': round(rmse, 6),
                })

        pd.DataFrame(calib_rows).to_csv(
            os.path.join(diag, '02_n_calibrated.csv'), index=False
        )

        reg_rows = []
        regressions = {}
        for sc in SCENARIOS:
            qp_arr  = np.array([d['qp']       for d in reg_data[sc]])
            tp_arr  = np.array([d['tp']        for d in reg_data[sc]])
            dur_arr = np.array([d['duration']  for d in reg_data[sc]])
            n_arr   = np.array([d['n']         for d in reg_data[sc]])
            regressions[sc] = {
                'f1_tp':       fit_regression(qp_arr, tp_arr),
                'f2_duration': fit_regression(qp_arr, dur_arr),
                'f3_n':        fit_regression(qp_arr, n_arr),
            }
            for fk, reg in regressions[sc].items():
                reg_rows.append({
                    'scenario': sc,
                    'function': fk,
                    'type': reg['type'],
                    'params': str([round(p, 6) for p in reg['params']]),
                    'r2': round(reg['r2'], 6),
                })

        pd.DataFrame(reg_rows).to_csv(
            os.path.join(diag, '03_regressions.csv'), index=False
        )
        _plot_regressions(diag, regressions, reg_data)

    # ------------------------------------------------------------------
    # Reconstruir TRs inválidos
    # ------------------------------------------------------------------
    recon_rows = []
    reconstructed: dict = {}

    for tr in invalid_trs:
        for sc in SCENARIOS:
            qp_raster = get_qpeak_raster(folder, sc, tr)

            if fixed_params is not None:
                tp_est, dur_est, n_est = fixed_params[sc]
                n_est = max(n_est, 1.001)
            else:
                tp_est  = float(regressions[sc]['f1_tp']['predict'](qp_raster))
                dur_est = float(regressions[sc]['f2_duration']['predict'](qp_raster))
                n_est   = max(float(regressions[sc]['f3_n']['predict'](qp_raster)), 1.001)

            t     = np.linspace(0.0, dur_est, n_points)
            q_new = scs_q(t, qp_raster, tp_est, n_est)

            df_new = pd.DataFrame({COL_TIME: t, COL_Q: q_new})
            df_new.to_csv(
                os.path.join(folder, f'TS_Q_{sc}_TR-{tr}.csv'),
                index=False,
            )

            reconstructed[(sc, tr)] = (t, q_new)
            recon_rows.append({
                'TR': tr, 'scenario': sc,
                'Qp_raster_m3s':        round(qp_raster, 4),
                'tp_estimated_h':       round(tp_est, 4),
                'duration_estimated_h': round(dur_est, 4),
                'n_estimated':          round(n_est, 6),
            })

    pd.DataFrame(recon_rows).to_csv(
        os.path.join(diag, '04_reconstruction.csv'), index=False
    )

    _plot_hydrographs(diag, all_data, reconstructed, valid_trs, invalid_trs)

    print(f"Diagnosticos  : {diag}")
    print(f"Hidrogramas reconstruidos: {len(invalid_trs) * len(SCENARIOS)}")


def BashFastFlood(JSONPath, SaveFullCSV=False):

    # ------------------------------------------------------------------------------------------------------------------
    # Leer JSON con parámetros de ejecución
    # ------------------------------------------------------------------------------------------------------------------
    with open(JSONPath, 'r') as json_data:
        UserData = json.load(json_data)

    # ------------------------------------------------------------------------------------------------------------------
    # Leer JSON con URL AWS
    # ------------------------------------------------------------------------------------------------------------------
    with open(UserData["AccessData"], 'r') as json_data:
        AccessData = json.load(json_data)

    # ------------------------------------------------------------------------------------------------------------------
    # Crear archivo log
    # ------------------------------------------------------------------------------------------------------------------
    # Nombre del archivo log
    log_file = os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"],'log_FastFlood.txt')

    # Abrir el archivo log en modo escritura
    log = open(log_file, 'a')

    # Configurar logger del módulo con el archivo de log
    logger = get_module_logger(log_file=log_file)
    logger.info("#" * 100)
    logger.info("INICIANDO BashFastFlood")
    logger.info(f"JSONPath: {JSONPath}")
    logger.info(f"ProjectPath: {UserData['ProjectPath']}")
    logger.info(f"NameBasinFolder: {UserData['NameBasinFolder']}")
    log_memory(logger, "Memoria inicial BashFastFlood")

    # ------------------------------------------------------------------------------------------------------------------
    # Crear las carpetas para la ejecución de FastFlood
    # ------------------------------------------------------------------------------------------------------------------
    ProjectPath = os.path.join(UserData['ProjectPath'], UserData["NameBasinFolder"])
    CreateFolders(ProjectPath)

    # ------------------------------------------------------------------------------------------------------------------
    # Creación de los rasters de infiltración y manning de NbS y BaU
    # ------------------------------------------------------------------------------------------------------------------
    # Equivalencias de codificación entre las coberturas de FastFlood y WaterProof
    lulc_fastflood_to_lulc_end = {10: 2,  # Forest
                                  20: 7,  # Shrublands
                                  30: 3,  # Grassland
                                  40: 4,  # Agricultural
                                  50: 5,  # Building
                                  60: 6,  # Bare area
                                  70: 0,  # Ice
                                  80: 1}  # Water

    lulc_bau_to_lulcfastflood = {2: 10,  # Forest
                                 7: 20,  # Shrublands
                                 3: 30,  # Grassland
                                 4: 40,  # Agricultural
                                 5: 50,  # Building
                                 6: 60,  # Bare area
                                 0: 70,  # Ice
                                 1: 80}  # Water

    # Categorias bloqueadas para el análisis
    locked_categories = {1, 50, 70, 80, 90, 100, 110}

    # ------------------------------------------------------------------------------------------------------------------
    # Configurar entradas para contruccińo de rasters de Infiltración y n-Manning en BaU y SbN
    # ------------------------------------------------------------------------------------------------------------------
    # Leer parámetros del JSON
    lulc_to_manning, manning_dict, infiltracion_dict, infiltration_change_dict, raster_inputs, path_out = red_params_from_json(
        JSONPath)

    # Ajustar raster a la misma extensión y alineación
    aligned, ref_raster = resample_all_to_highest_resolution(raster_inputs, os.path.join(ProjectPath, 'out', '06-FLOOD', 'Tmp','Resampled'))

    # ------------------------------------------------------------------------------------------------------------------
    # Crear rasters de n-Manning para BaU
    # ------------------------------------------------------------------------------------------------------------------
    # '''
    generate_manning_bau(
        aligned["lulc_fastflood"],
        aligned["bau_lulc"],
        os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning_BaU.tif"),
        lulc_to_manning,
        lulc_bau_to_lulcfastflood,
        locked_categories,
        manning_base_path=aligned["manning_base"],  # opcional
        use_raster=False  # ← cambia a False si quieres forzar el diccionario  # ahora aprovechamos tu raster base
    )

    # Check - Los n-Manning en el BaU nunca deben ser mayor que el Current
    n_H     = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning.tif")
    n_BaU   = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning_BaU.tif")
    CheckPixelDepth(n_H, n_BaU, n_BaU)
    #'''

    # ------------------------------------------------------------------------------------------------------------------
    # Crear rasters de infiltración para BaU
    # ------------------------------------------------------------------------------------------------------------------
    # '''
    generate_infiltracion_bau(
        aligned["lulc_fastflood"],
        aligned["waterproof_current"],
        aligned["bau_lulc"],
        aligned["infiltracion_base"],
        os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Infiltration_BaU.tif"),
        lulc_fastflood_to_lulc_end,
        infiltration_change_dict
    )

    # Check - La Infiltración en el BaU nunca debe ser mayor que el Current
    n_H     = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Infiltration.tif")
    n_BaU   = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Infiltration_BaU.tif")
    CheckPixelDepth(n_H, n_BaU, n_BaU)
    #'''

    # ------------------------------------------------------------------------------------------------------------------
    # Crear rasters de n-Manning para SbN
    # ------------------------------------------------------------------------------------------------------------------
    # '''
    generate_nbs_manning(
        aligned["lulc_fastflood"],  #    LULC FastFlood (alineado)
        aligned["waterproof_current"],  # LULC waterproof (alineado)
        aligned["portafolio"],  # Portafolio (alineado)
        os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning_BaU.tif"), # Manning Bau (alineado)
        os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning.tif"), # Current
        os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning_NbS.tif"),
        lulc_to_manning,
        lulc_fastflood_to_lulc_end,
        locked_categories,
        manning_dict,
    )
    # Check - Los n-Manning de NbS nunca deben ser menores que el BaU
    n_BaU   = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning_BaU.tif")
    n_NbS   = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Manning_NbS.tif")
    CheckPixelDepth(n_BaU, n_NbS, n_NbS, Con=False)
    #'''
    # ------------------------------------------------------------------------------------------------------------------
    # Crear rasters de infiltración para SbN
    # ------------------------------------------------------------------------------------------------------------------
    # '''
    generate_nbs_infiltration(aligned["lulc_fastflood"],  # LULC FastFlood (alineado)
                              aligned["waterproof_current"],  # LULC waterproof (alineado)
                              aligned["portafolio"],  # Portafolio (alineado)
                              aligned["infiltracion_base"],  # Infiltracion Base
                              os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Infiltration_BaU.tif"),
                              os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Infiltration_NbS.tif"),
                              lulc_fastflood_to_lulc_end,  # Mapeo de LULC FF a final
                              locked_categories,  # Categorías que no deben cambiar
                              infiltracion_dict,
                              )

    # Check - La Infiltración en el BaU nunca debe ser mayor que el Current
    n_NbS   = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Infiltration_NbS.tif")
    n_BaU   = os.path.join(ProjectPath, "in", "06-FLOOD", "Raster", "Infiltration_BaU.tif")
    CheckPixelDepth(n_BaU, n_NbS, n_NbS, Con=False)
    #'''
    # ------------------------------------------------------------------------------------------------------------------
    # Recortar raster de usos del suelo para daños
    # ------------------------------------------------------------------------------------------------------------------
    """
    01 : MSZ, open spaces, low vegetation surfaces NDVI <= 0.3
    02 : MSZ, open spaces, medium vegetation surfaces 0.3 < NDVI <=0.5
    03 : MSZ, open spaces, high vegetation surfaces NDVI > 0.5
    04 : MSZ, open spaces, water surfaces LAND < 0.5
    05 : MSZ, open spaces, road surfaces
    11 : MSZ, built spaces, residential, building height <= 3m
    12 : MSZ, built spaces, residential, 3m < building height <= 6m
    13 : MSZ, built spaces, residential, 6m < building height <= 15m
    14 : MSZ, built spaces, residential, 15m < building height <= 30m
    15 : MSZ, built spaces, residential, building height > 30m
    21 : MSZ, built spaces, non-residential, building height <= 3m
    22 : MSZ, built spaces, non-residential, 3m < building height <= 6m
    23 : MSZ, built spaces, non-residential, 6m < building height <= 15m
    24 : MSZ, built spaces, non-residential, 15m < building height <= 30m
    25 : MSZ, built spaces, non-residential, building height > 30m
    NoData [255]
    """
    # Estos corresponden a los valores de los pixeles en los rasters de coberturas de daño,
    # que se consideran para evaluar en cada categoría de daño.
    CodeLULC = {}
    CodeLULC['Residential'] = [11, 12, 13, 14, 15]
    CodeLULC['Commercial']  = [21, 22, 23, 24, 25]
    CodeLULC['Industrial']  = [21, 22, 23, 24, 25]
    CodeLULC['InfraRoads']  = [1]
    CodeLULC['Agriculture'] = [40]
    CodeLULC['GHS_BUILT']   = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 40]

    LULC_Damage = {}
    LULC_Damage['Residential'] = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C_Agri.tif'
    LULC_Damage['Commercial']  = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C_Agri.tif'
    LULC_Damage['Industrial']  = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C_Agri.tif'
    LULC_Damage['InfraRoads']  = f'{ProjectPath}/in/06-FLOOD/Raster/Road.tif'
    LULC_Damage['Agriculture'] = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C_Agri.tif'
    LULC_Damage['GHS_BUILT']   = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C_Agri.tif'

    # Recortar base de datos de GHS-BUILT-C
    carpeta_tiles   = os.path.join(UserData["DamagesDataBasePath"],"01-GHS-BUILT-C")
    aoi_path        = UserData['CatchmentPath']
    ruta_salida     = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C.tif'
    Clic_Mosaic_DataBase(carpeta_tiles, aoi_path, ruta_salida)

    # Unir los datos de agricultura con GHS-BUILT-C
    Join_GHSBUILT_Agricultural(ruta_salida , UserData["CurrentLulcPath"], LULC_Damage['Residential'], bloque_tamano=512)

    # Recortar raster de vías
    carpeta_tiles   = os.path.join(UserData["DamagesDataBasePath"],"02-Road")
    aoi_path        = UserData['CatchmentPath']
    ruta_salida     = LULC_Damage['InfraRoads']
    Clic_Mosaic_DataBase(carpeta_tiles, aoi_path, ruta_salida)

    # ------------------------------------------------------------------------------------------------------------------
    # Generar IDF y obtener tormenta de diseño
    # ------------------------------------------------------------------------------------------------------------------
    carpeta_tiles = os.path.join(UserData["DamagesDataBasePath"], "03-IDF")
    aoi_path    = UserData['CatchmentPath']
    ruta_salida = os.path.join(UserData['ProjectPath'], UserData["NameBasinFolder"], 'in', '06-FLOOD', 'Raster','IDF.csv')
    df_idf = calcular_estadisticas_idf_cuenca(
        carpeta_tiles,
        aoi_path,
        'mean')

    # Del raster original se tienen las precipitaciones totales por cada duración por TR. Para generar la intensidades
    # dividimos por la duración
    df_idf = df_idf / np.array([[3], [6], [12], [24], [48], [72], [120], [240]])
    df_idf.to_csv(ruta_salida)

    # ------------------------------------------------------------------------------------------------------------------
    # Ejecutar Escenarios (Historic, BaU, NbS)
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Scenario simulation \n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    if UserData["FastFloodParams"]["ChannelParams"]["Status"] == 0:
        Channel = None
    else:
        Channel = [UserData["FastFloodParams"]["ChannelParams"]["WidthMul"],
                   UserData["FastFloodParams"]["ChannelParams"]["WidthExp"],
                   UserData["FastFloodParams"]["ChannelParams"]["DepthMul"],
                   UserData["FastFloodParams"]["ChannelParams"]["DepthExp"],
                   UserData["FastFloodParams"]["ChannelParams"]["CrossSection"],
                   UserData["FastFloodParams"]["ChannelParams"]["ChannelManning"]]

    if UserData["FastFloodParams"]["BoundaryCondition"] == 0:
        BoundaryCondition = None
    else:
        BoundaryCondition = UserData["FastFloodParams"]["BoundaryCondition"]

    # Ejecutar Escenarios
    HPaths   = RunScenarios( ProjectPath=ProjectPath,
                             FastFloodPath=UserData['FastFloodPath'],
                             customurl=AccessData["customurl"],
                             D=UserData["ClimateParams"]["AnalysisStormDuration"],
                             D_DS=UserData["ClimateParams"]["DesignStormDuration_Historic"],
                             D_DS_CC=UserData["ClimateParams"]["DesignStormDuration_ClimateChange"],
                             P=UserData["ClimateParams"]["Period"],
                             Q=UserData["ClimateParams"]["StormQuantile"],
                             SSP=UserData["ClimateParams"]["Scenario"],
                             TR=UserData["ClimateParams"]["ReturnPeriod"],
                             DEM_Path=UserData["DEMPath"],
                             Manning_Path=UserData["ManningPath"],
                             Inf_Path=UserData["InfiltrationPath"],
                             BasinPath=UserData['CatchmentPath'],
                             Channel=Channel,
                             BoundaryCondition=BoundaryCondition,
                             log=log, IDF_Table=df_idf, StatusExe=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Check - Reconstrucción de hidrogramas dañados por inestabilidad numérica
    # ------------------------------------------------------------------------------------------------------------------
    DischargePath  = ProjectPath + f'/out/06-FLOOD/Discharge'
    correct_hydrographs(DischargePath, UserData["DEMPath"])

    # ------------------------------------------------------------------------------------------------------------------
    # Check - Se verifica que las profundidades del escenario BaU nunca sean menores que las del escenario Current,
    #         Asi como también que las del escenario NbS nunca sean mayores que las del escenario BaU.
    # Reglas aplicadas:
    #     - Regla 3: Si Current > BaU > NbS → Current = BaU
    #     - Regla 4: Si Current > NbS > BaU → BaU = Current
    #     - Regla 5: Si NbS > Current > BaU → BaU = Current, NbS = Current
    #     - Regla 6: Si NbS > BaU > Current → NbS = Current
    # ------------------------------------------------------------------------------------------------------------------
    for TR_i in UserData["ClimateParams"]["ReturnPeriod"]:
        H_Path_H = ProjectPath + f'/out/06-FLOOD/Flood/Flood_Current_TR-{TR_i}.tif'
        H_Path_BaU = ProjectPath + f'/out/06-FLOOD/Flood/Flood_BaU_TR-{TR_i}.tif'
        H_Path_NbS = ProjectPath + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR_i}.tif'

        CheckPixelDepth_V2(H_Path_H, H_Path_BaU, H_Path_NbS, H_Path_H, H_Path_BaU, H_Path_NbS)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 7 - Leer curva de factors de daño y el costo máximo
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Estimate of expected annual damage for each scenario \n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    # Leer curvas de daño
    DC = ReadDamageCurve(ProjectPath=os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"]))
    LogMessage = f"Read Dagame Curve - Ok \n"
    log.write(LogMessage)

    # Se aplica la tasa de cambio en la cual se entregan los costos máximos de las funciones de daño
    DC = DC*UserData["DamagesExchangeRate"]

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Read Flood Depth - Scenario Current\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    # Las funciones de costo de daño por inundación están en las siguientes unidades
    # Residential ($/m^2)
    # Commercial  ($/m^2)
    # Industrial  ($/m^2)
    # InfraRoads  ($/m^2)
    # Agriculture ($/m^2)
    # Para estimar el costo de daño, se multiplican estos valores por el área de cada pixel.
    # En línea con esto, la base de datos se encuentra en el CSR:
    # Mollweide(WKID 54009), Albers Equal - Area, Lambert Azimuthal Equal - Area.
    # Proyección que es equivalente. Es decir que no distorsiona la cantidad de superficie cubierta,
    # aunque puede distorsionar forma y distancia.
    # Dado que la fuente de datos que se está utilizando para identificar el uso comercial
    # e industrial es GHS_BUILT_C y esta solo mapea el uso no residencial, se aplica
    # el factor extra de distribución ingresado por el usuario
    FactorArea = {}
    FactorArea['Residential']   = 100 #m^2
    FactorArea['Commercial']    = 100*UserData["SplitArea"]["Commercial"] #m^2
    FactorArea['Industrial']    = 100*UserData["SplitArea"]["Industrial"]  #m^2
    FactorArea['InfraRoads']    = 60 #m^2 (Se considera un ancho de vía 6 metros, pixels de 10 metros)
    FactorArea['Agriculture']   = 100 #m^2

    # Categorias de daño
    Cat_Damage  = ['Residential', 'Commercial', 'Industrial', 'InfraRoads', 'Agriculture']
    # Nombre de los escenarios
    NameSce     = ['Current', 'BaU', 'NbS']
    # Crear un DataFrame vacío para almacenar resultados
    Total_EAD   = pd.DataFrame(columns=Cat_Damage,index=NameSce)

    for Sce in NameSce:
        # --------------------------------------------------------------------------------------------------------------
        # Calculo de las profundidades globales para cálculo de indicadores
        # --------------------------------------------------------------------------------------------------------------
        df = Raster2Zonal(HPaths['H'][Sce], UserData["CatchmentPath"])

        RR = []
        for TR_i in UserData["ClimateParams"]["ReturnPeriod"]:
            dfQ = pd.read_csv(ProjectPath + f'/out/06-FLOOD/Discharge/TS_Q_{Sce}_TR-{TR_i}.csv')
            RR.append(dfQ["discharge (m3/s)"].max())

        df['Peak Descharge [m3/s]'] = RR
        df.to_csv(f'{UserData["ProjectPath"]}/INDICATORS/in/INPUTS_FLOOD_{Sce}.csv',index=False)

        # ----------------------------------------------------------------------------------------------------------
        # Calculo de las profundidades por uso de suelo de daño
        # ----------------------------------------------------------------------------------------------------------
        logger.info(f"Procesando profundidades para escenario {Sce}")
        log_memory(logger, f"INICIO pipeline chunked - Escenario {Sce}")

        # Mapeo raster fuente → categorías que usan sus códigos
        _raster_cats = {
            'GHS_BUILT':  ['Residential', 'Commercial', 'Industrial', 'Agriculture'],
            'InfraRoads': ['InfraRoads'],
        }

        # Acumuladores de EAD por categoría
        ead_accum = {Cat: 0.0 for Cat in Cat_Damage}

        # Limpiar CSVs previos si SaveFullCSV (se escribirán por chunk en modo append)
        if SaveFullCSV:
            for Cat in Cat_Damage:
                for _p in [
                    f'{UserData["ProjectPath"]}/{UserData["NameBasinFolder"]}/out/06-FLOOD/Flood/H_{Cat}_{Sce}.csv',
                    f'{UserData["ProjectPath"]}/{UserData["NameBasinFolder"]}/out/06-FLOOD/Damages/01-Damage_{Cat}_{Sce}.csv',
                    os.path.join(ProjectPath, 'out', '06-FLOOD', 'Damages', f'02-Expected_Annual_Damage_{Cat}_{Sce}.csv'),
                ]:
                    if os.path.exists(_p):
                        os.remove(_p)

        # Pipeline chunked: chunk → filtrar por Cat → TR_Damage → EAD → acumular escalar
        for source_key, cats_source in _raster_cats.items():
            logger.info(f"Fuente: {source_key} | Categorías: {cats_source}")
            for chunk_df in Raster2DataFrame_Damages_Generator(
                LULC_Damage[source_key],
                HPaths['H'][Sce],
                CodeLULC[source_key],
                UserData["CatchmentPath"],
                log_file=log_file
            ):
                for Cat in cats_source:
                    df_cat = chunk_df[chunk_df['Code'].isin(CodeLULC[Cat])].copy()
                    if df_cat.empty:
                        continue

                    if SaveFullCSV:
                        _path_h = f'{UserData["ProjectPath"]}/{UserData["NameBasinFolder"]}/out/06-FLOOD/Flood/H_{Cat}_{Sce}.csv'
                        df_cat.to_csv(_path_h, mode='a', header=not os.path.exists(_path_h), index=False)

                    df_cat.drop('Code', axis=1, inplace=True)
                    damage_chunk = TR_Damage(df_cat, DC, category=Cat) * FactorArea[Cat]

                    if SaveFullCSV:
                        _path_dmg = f'{UserData["ProjectPath"]}/{UserData["NameBasinFolder"]}/out/06-FLOOD/Damages/01-Damage_{Cat}_{Sce}.csv'
                        damage_chunk.to_csv(_path_dmg, mode='a', header=not os.path.exists(_path_dmg), index=False)

                    ead_chunk = EAD(TR=UserData['ClimateParams']['ReturnPeriod'], Damage=damage_chunk, NameCol=Cat)

                    if SaveFullCSV:
                        _path_ead = os.path.join(ProjectPath, 'out', '06-FLOOD', 'Damages', f'02-Expected_Annual_Damage_{Cat}_{Sce}.csv')
                        ead_chunk.to_csv(_path_ead, mode='a', header=not os.path.exists(_path_ead))

                    ead_accum[Cat] += ead_chunk.sum().sum()

                del chunk_df

        log_memory(logger, f"FIN pipeline chunked - Escenario {Sce}")

        for Cat in Cat_Damage:
            Total_EAD.loc[Sce, Cat] = ead_accum[Cat]
            log.write(f"Read flood depth for the {Sce} scenario for {Cat} damages category- Ok \n")
            log.write(f"Damage estimation for each return period of the {Sce} scenario {Cat} - Ok \n")
            log.write(f"Estimated annual expected damages for the {Sce} scenario for {Cat} damages category- Ok- Ok \n")
            log.write(f"Estimated cumulative annual expected damages for the {Sce} scenario for {Cat} damages category- Ok- Ok \n")
            print(f"Estimated cumulative annual expected damages for the {Sce} scenario for {Cat} damages category -> Ok")

    # Guardar total
    Total_EAD.to_csv(os.path.join(ProjectPath, 'out', '06-FLOOD', 'Damages', f'03-Expected_Annual_Damage_Total.csv'),index_label='Scenarios')
    log.write("Correct reading - Scenario Current \n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 10 - Desagregar costos
    # ------------------------------------------------------------------------------------------------------------------
    # Guardar entradas
    Total_EAD.to_csv(os.path.join(ProjectPath,'in','05-DISAGGREGATION','01-INPUTS_Flood.csv'),index_label='Scenarios')
    #'''

    # Leer desagregación del BaU y de NbS de Carbono
    CO2_BaU = pd.read_csv(os.path.join(ProjectPath,'out','05-DISAGGREGATION', '02-OUTPUTS_BaU.csv'),index_col=0)
    CO2_NBS = pd.read_csv(os.path.join(ProjectPath, 'out', '05-DISAGGREGATION','02-OUTPUTS_NBS.csv'),index_col=0)

    # Leer datos de la NBS
    NBS     = pd.read_csv(os.path.join(ProjectPath,'in','05-DISAGGREGATION','01-INPUTS_NBS.csv')).values[:, 1:]

    # Leer datos de tiempo
    Time    = pd.read_csv(os.path.join(ProjectPath,'in','05-DISAGGREGATION','01-INPUTS_Time.csv')).values[0][0]

    # Desagregar datos
    Results_BaU, Results_NBS = DesaggregationData(Total_EAD, Cat_Damage, NBS, Time)

    # Unir los datos con los de Carbono
    Results_BaU['WC (Ton)'] = CO2_BaU.values
    Results_NBS['WC (Ton)'] = CO2_NBS.values

    # Guardar datos en el folder de Damages
    Results_BaU.to_csv(os.path.join(ProjectPath, 'out', '06-FLOOD', 'Damages', '04-Damage_Dissagregation_BaU.csv'), index_label='Time')
    Results_NBS.to_csv(os.path.join(ProjectPath, 'out', '06-FLOOD', 'Damages', '04-Damage_Dissagregation_NbS.csv'), index_label='Time')
    Results_BaU.to_csv(os.path.join(ProjectPath, 'out', '05-DISAGGREGATION', '02-OUTPUTS_BaU_Flood.csv'),index_label='Time')
    Results_NBS.to_csv(os.path.join(ProjectPath, 'out', '05-DISAGGREGATION', '02-OUTPUTS_NbS_Flood.csv'),index_label='Time')

    # Guardar datos en el folder de ROI
    Results_BaU.to_csv(os.path.join(UserData['ProjectPath'], 'ROI', 'in', '7-Damages_BaU.csv'), index_label='Time')
    Results_NBS.to_csv(os.path.join(UserData['ProjectPath'], 'ROI', 'in', '8-Damages_NbS.csv'), index_label='Time')

    log.write(f"Disaggregation of the scenarios carried out correctly.\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 11 - Análisis ROI
    # ------------------------------------------------------------------------------------------------------------------
    ROI_FastFlood(os.path.join(UserData['ProjectPath'],'ROI'))
    log.write(f"ROI.\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 12 - Indicadores
    # ------------------------------------------------------------------------------------------------------------------
    Indicators_BaU_NBS(os.path.join(UserData['ProjectPath'],'INDICATORS'))
    log.write(f"Indicators.\n")

    # Cerrar Log
    log.close()
    #"""
