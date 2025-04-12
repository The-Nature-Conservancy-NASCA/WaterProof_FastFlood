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

import os, sys
import subprocess
import numpy as np
import threading
import datetime
from numpy.core._multiarray_umath import ndarray
from fontTools.misc.cython import returns
from osgeo import ogr,gdal
import geopandas as gpd
from pyproj import Transformer, CRS
import pandas as pd
from pyproj import CRS, Transformer
import json
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.features import geometry_mask
from rasterio.mask import mask
import time
import scipy as sp
import tempfile
from scipy.interpolate import interp1d


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

def CheckPixelDepth_BAU_NBS(ruta_raster1, ruta_raster2, salida='raster_recortado.tif'):
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

        with WarpedVRT(src1, resampling=Resampling.nearest) as vrt1, \
             WarpedVRT(src2, resampling=Resampling.nearest) as vrt2:

            data1 = vrt1.read(1)
            data2 = vrt2.read(1)

            # Aplicar corrección
            resultado = np.where(data2 > data1, data1, data2)

            # Determinar tipo de dato mínimo necesario
            dtype_resultado = resultado.dtype.name

            # Metadatos de salida (usamos src2, no el VRT)
            perfil = src2.profile.copy()
            perfil.update({
                'dtype': dtype_resultado,
                'compress': 'LZW',
                'predictor': 2 if np.issubdtype(resultado.dtype, np.floating) else 1,
                'tiled': True,
                'blockxsize': 256,
                'blockysize': 256,
                'BIGTIFF': 'IF_SAFER',
                'nodata': src2.nodata
            })

            # Guardar ráster corregido
            with rasterio.open(salida, 'w', **perfil) as dst:
                dst.write(resultado, 1)

    print(f"✔️ Raster corregido guardado como: {salida} ({dtype_resultado}, LZW)")


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
    return [minx, maxy, maxx, miny]

def DownloadInputs(FastFloodPath, Basin_shp_BoundingBox, DemResolution, DEM_Path,
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
        Coordenadas del bounding box de la cuenca en proyección pseudo-Mercator (EPSG:3857) (e.g., [xmin, ymin, xmax, ymax]).
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
        if len(BasinBox) == 4:
            raise ValueError(f"the list must contain four elements.")
    elif isinstance(Basin_shp_BoundingBox, str):
        # Extraer Boundary del shapefile de la cuenca
        BasinBox = Get_Basin_bbox(Basin_shp_BoundingBox)

    # ------------------------------------------------------------------------------------------------------------------
    # Descargar DEM
    # ------------------------------------------------------------------------------------------------------------------
    # Se estima el buffer para el dominio de descarga de la información
    Buffer_m    = Buffer_km*1000
    BasinBox[0] = BasinBox[0] - Buffer_m
    BasinBox[1] = BasinBox[1] + Buffer_m
    BasinBox[2] = BasinBox[2] + Buffer_m
    BasinBox[3] = BasinBox[3] - Buffer_m

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
                                   FastFloodPath, customurl,
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
                                       FastFloodPath, customurl,
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
                     FastFloodPath, customurl=None,
                     BasinBox=None, DemResolution=None,
                     DEM_Path=None, Manning_Path=None, Inf_Path=None, IDF_Path=None, Fac_CC_Path=None,
                     D_DS=None, D_DS_CC=None, D=None, P=None, Q=None, SSP=None, TR=None,
                     H_Path=None, Q_Path=None, V_Path=None, nOut=None, InfOut=None, PathShp=None, Verbose=True,
                     ChW_Path=None,ChD_Path=None,TS_Path=None,Channel=None,LULC_Path=None, ocean=None):
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
    if (TR is not None) and (D_DS is not None):
        Comando += ['-designstorm', f'{TR}', f'{D_DS}']
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

    return Comando

def RunScenarios(ProjectPath, FastFloodPath, D, P, Q, SSP, TR,
                 DEM_Path, Manning_Path, Inf_Path, BasinPath, D_DS=None, D_DS_CC=None,
                 log=None, customurl=None, Channel=None):

    # Residensial
    Results           = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Ejecutar FastFlood - Current
    # ------------------------------------------------------------------------------------------------------------------
    H_Path_Tmp = ProjectPath + f'/out/06-FLOOD/Tmp/H.tif'
    V_Path_Tmp = ProjectPath + f'/out/06-FLOOD/Tmp/V.tif'
    Q_Path_Tmp = ProjectPath + f'/out/06-FLOOD/Tmp/Q.tif'

    raster_paths = {}
    for TR_i in TR:
        H_Path      = ProjectPath + f'/out/06-FLOOD/Flood/Flood_Current_TR-{TR_i}.tif'
        V_Path      = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_Current_TR-{TR_i}.tif'
        Q_Path      = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_Current_TR-{TR_i}.tif'
        TS_Q_Path   = ProjectPath + f'/out/06-FLOOD/Discharge/TS_Q_Current_TR-{TR_i}.csv'

        # Crear lista de rasters
        raster_paths[TR_i] = H_Path

        if SSP == "Historic":
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: Current | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                         FastFloodPath, customurl,
                                         DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                         SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Channel=Channel,
                                         H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path)

        else:
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: Current | Climate: Scenario: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                         FastFloodPath, customurl,
                                         DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                         SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel,
                                         H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path)

        #'''
        # Ejecutar comando
        ExeFastFlood(Comando,log)

        # Recortar el raster al shapefile de la cuenca
        ClicRasterWithBasin(H_Path_Tmp, BasinPath, salida=H_Path)
        ClicRasterWithBasin(V_Path_Tmp, BasinPath, salida=V_Path)
        ClicRasterWithBasin(Q_Path_Tmp, BasinPath, salida=Q_Path)
        #'''

    # Guardar Listado de Raster de Profundidad
    Results['Current'] = raster_paths

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - BaU
    # ------------------------------------------------------------------------------------------------------------------
    # n-Manning modificado por la opción de adaptación - BaU
    Manning_Path    = ProjectPath + '/in/06-FLOOD/Raster/Manning_BaU.tif'
    # Infiltración modificada por la opción de adaptación - BaU
    Inf_Path        = ProjectPath + '/in/06-FLOOD/Raster/Infiltration_BaU.tif'
    raster_paths    = {}

    for TR_i in TR:
        H_Path = ProjectPath + f'/out/06-FLOOD/Flood/Flood_BaU_TR-{TR_i}.tif'
        V_Path = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_BaU_TR-{TR_i}.tif'
        Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_BaU_TR-{TR_i}.tif'
        TS_Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/TS_Q_BaU_TR-{TR_i}.csv'

        # Crear lista de rasters
        raster_paths[TR_i] = H_Path

        if SSP == "Historic":
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: BaU | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Channel=Channel,
                                       H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path)

        else:
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: Bau | Climate : {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel,
                                       H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path)

        #'''
        # Ejecutar comando
        ExeFastFlood(Comando,log)

        # Recortar el raster al shapefile de la cuenca
        ClicRasterWithBasin(H_Path_Tmp, BasinPath, salida=H_Path)
        ClicRasterWithBasin(V_Path_Tmp, BasinPath, salida=V_Path)
        ClicRasterWithBasin(Q_Path_Tmp, BasinPath, salida=Q_Path)
        #'''

    # Guardar Listado de Raster de Profundidad
    Results['BaU'] = raster_paths

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - NbS
    # ------------------------------------------------------------------------------------------------------------------
    # n-Manning modificado por la opción de adaptación - BaU
    Manning_Path    = ProjectPath + '/in/06-FLOOD/Raster/Manning_NbS.tif'
    # Infiltración modificada por la opción de adaptación - BaU
    Inf_Path        = ProjectPath + '/in/06-FLOOD/Raster/Infiltration_NbS.tif'
    raster_paths    = {}

    for TR_i in TR:
        H_Path = ProjectPath + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR_i}_Tmp.tif'
        V_Path = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_NbS_TR-{TR_i}.tif'
        Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_NbS_TR-{TR_i}.tif'
        TS_Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/TS_Q_NbS_TR-{TR_i}.csv'

        # Crear lista de rasters
        raster_paths[TR_i] = ProjectPath + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR_i}.tif'

        if SSP == "Historic":
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: NbS | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, Channel=Channel,
                                       H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path)
        else:
            if log is not None:
                log.write("# ---------------------------------------------------------------------------------------------------\n")
                log.write(f"# Execution FastFlood | Scenario: NbS | Climate: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
                log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D, D_DS=D_DS, D_DS_CC=D_DS_CC, P=P, Q=Q, Channel=Channel,
                                       H_Path=H_Path_Tmp, Q_Path=Q_Path_Tmp, V_Path=V_Path_Tmp, TS_Path=TS_Q_Path)

        #'''
        # Ejecutar comando
        ExeFastFlood(Comando,log)

        # Recortar el raster al shapefile de la cuenca
        ClicRasterWithBasin(H_Path_Tmp, BasinPath, salida=H_Path)
        ClicRasterWithBasin(V_Path_Tmp, BasinPath, salida=V_Path)
        ClicRasterWithBasin(Q_Path_Tmp, BasinPath, salida=Q_Path)
        #'''

    # Guardar Listado de Raster de Profundidad
    Results['NbS'] = raster_paths

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

    Retorna:
    --------
    Dict[str, DataFrame]
        Diccionario con DataFrames de costos interpolados para cada categoría.
        Cada DataFrame tiene la misma estructura que df_profundidad.
    """

    # Extraer datos de profundidad (x) y costos (y) de la categoría
    x = df_costos.index.to_numpy().astype(float)  # Profundidades conocidas
    y = df_costos[category].to_numpy().astype(float)  # Costos conocidos

    # Ordenar los datos por profundidad (necesario para interp1d)
    sorted_idx  = np.argsort(x)
    x_sorted    = x[sorted_idx]
    y_sorted    = y[sorted_idx]

    # Crear función de interpolación lineal con extrapolación
    interpolador = interp1d(x_sorted, y_sorted,
                            kind='linear',
                            fill_value='extrapolate')

    # Aplicar interpolación a todo el DataFrame de profundidades
    return df_profundidad.applymap(lambda val: interpolador(val))

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

def DesaggregationData(Data, NameCol, NBS, Time, CO2_BaU, CO2_NBS):

    """
    Data.iloc[0,:]  = 4000000
    Data.iloc[1,:]  = 10000000
    Data.iloc[2, :] = 7000000
    #"""

    # Funtion Lambda
    Sigmoid_Desaggregation = lambda Wmax, Wo, r, t: Wmax / (1 + (((Wmax / Wo) - 1) * np.exp(-t * r)))

    # ------------------------------------------------------------------------------------------------------------------
    # Current-BaU
    # ------------------------------------------------------------------------------------------------------------------
    nn = np.shape(Data)[1]
    Results_BaU = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)
    r = -1 * np.log(0.000000001) / Time
    t = np.arange(0, Time + 1)
    for i in range(0, len(NameCol)):
        Results_BaU[NameCol[i]] = Sigmoid_Desaggregation(Data[NameCol[i]][1], Data[NameCol[i]][0], r, t)

    # ------------------------------------------------------------------------------------------------------------------
    # BaU-NBS
    # ------------------------------------------------------------------------------------------------------------------
    # Estimation Time NBS
    n = np.size(NBS[0, 2:])
    t_NBS = np.empty([n, 1])
    p_NBS = np.empty([n, 1])
    for i in range(0, n):
        t_NBS[i] = np.nansum(NBS[:, 0] * NBS[:, i + 2]) / np.nansum(NBS[:, i + 2])
        p_NBS[i] = np.nansum(NBS[:, 1] * NBS[:, i + 2]) / np.nansum(NBS[:, i + 2])

    # Desaggregation
    Results_NBS = pd.DataFrame(data=np.empty([Time + 1, nn]), columns=NameCol)

    # Estimation Diff
    [f, c] = Data.shape
    Data1 = Data[2:].values
    Data1[0, :] = Data1[0, :] - Data.loc['BaU'].values
    Tmp = np.nancumsum(Data1, 0)
    for i in range(1, f - 2):
        Data1[i, :] = Data1[i, :] - Data.loc['BaU'].values - Tmp[i - 1, :]
        Tmp = np.nancumsum(Data1, 0)

    for i in range(0, nn):
        Tmp = np.zeros((Time + 1, n))
        for j in range(0, n):
            t = np.arange(0, Time + 1 - (j + 1))
            Wmax = Data1[j, i]
            tmax = t_NBS[j][0]
            Wo = p_NBS[j][0] * Data1[j, i] * 0.01
            r = -1 * np.log(0.000000001) / tmax

            # print(Wmax,'|', Wo,'|', r)
            Tmp[(j + 1):, j] = Sigmoid_Desaggregation(Wmax, Wo, r, t)

        Tmp[np.isnan(Tmp)] = 0
        Results_NBS[NameCol[i]] = np.sum(Tmp, 1) + Results_BaU[NameCol[i]].values

    # Correct Negative Values
    Posi = Results_NBS.index.values
    for i in range(0, nn):
        print(Data.columns[i])
        if np.sum(Results_NBS.iloc[:, i] < 0) > 0:
            TmpBaU = Results_BaU.iloc[:, i]
            TmpNBS = Results_NBS.iloc[:, i]

            nN = Posi[TmpNBS < 0]
            V = (TmpBaU[Posi[TmpNBS > 0]] - TmpNBS[Posi[TmpNBS > 0]]) / TmpBaU[Posi[TmpNBS > 0]]
            V = np.median(V)
            for j in range(0, len(nN)):
                TmpNBS[nN[j]] = TmpBaU[nN[j]] * V
            Results_NBS.iloc[:, i] = TmpNBS
        else:
            print(f'The {NameCol[i]} data does not require correction for negative values.')

    Results_BaU = Results_BaU.fillna(0)
    Results_NBS = Results_NBS.fillna(0)
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
    NameIndex: ndarray = np.arange(1, t_roi + 1)
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

def Raster2DataFrame(ruta_cobertura, profundidades, codigos_cobertura, shapefile_mascara=None, Threshold_H=0.01):
    """
    Extrae profundidades de uno o varios rásteres en las coordenadas de cobertura seleccionadas,
    limitando opcionalmente con un shapefile de máscara en WGS84.

    Parámetros:
        ruta_cobertura (str): Ruta al ráster de cobertura.
        profundidades (str | dict): Ruta a un solo ráster o diccionario {nombre_columna: ruta}.
        codigos_cobertura (list): Lista de códigos de cobertura deseados.
        shapefile_mascara (str, opcional): Ruta a shapefile para limitar el análisis.

    Retorna:
        pd.DataFrame: Con columnas ['Cobertura', <una o varias columnas de profundidad>]
    """

    # Establecer el CRS objetivo a partir del primer ráster de profundidad
    if isinstance(profundidades, str):
        profundidades = {'Profundidad': profundidades}

    primera_ruta = list(profundidades.values())[0]
    with rasterio.open(primera_ruta) as ref_raster:
        crs_objetivo = ref_raster.crs

    # Leer shapefile si se proporciona
    geometria_mask = None
    if shapefile_mascara:
        gdf = gpd.read_file(shapefile_mascara)

        if not isinstance(gdf, gpd.GeoDataFrame) or 'geometry' not in gdf:
            raise ValueError("El archivo proporcionado no es un shapefile válido con geometría.")

        gdf = gdf.to_crs(crs_objetivo)  # reproyectar
        geometria_mask = gdf.geometry

    # Abrir ráster de cobertura
    with rasterio.open(ruta_cobertura) as src_cov:
        if src_cov.crs != crs_objetivo:
            vrt_cov = WarpedVRT(src_cov, crs=crs_objetivo, resampling=Resampling.nearest)
        else:
            vrt_cov = WarpedVRT(src_cov)

        with vrt_cov:
            cov_data = vrt_cov.read(1)
            transform = vrt_cov.transform
            shape = cov_data.shape

            # Máscara por geometría si se dio shapefile
            if geometria_mask is not None and geometria_mask.notna().any():
                mascara_geo = geometry_mask(
                    geometries=geometria_mask,
                    transform=transform,
                    invert=True,
                    out_shape=shape
                )
            else:
                mascara_geo = np.ones(shape, dtype=bool)  # sin restricción espacial

            # Máscara por códigos de cobertura
            mascara_cod = np.isin(cov_data, codigos_cobertura)

            # Combinar máscaras
            mascara_total = mascara_cod & mascara_geo

            filas, columnas = np.where(mascara_total)
            coords = list(zip(*xy(transform, filas, columnas)))
            valores_cobertura = cov_data[filas, columnas]

    # DataFrame base
    df = pd.DataFrame({'Code': valores_cobertura})

    # Agregar capas de profundidad
    for nombre_columna, ruta in profundidades.items():
        with rasterio.open(ruta) as src_prof:
            with WarpedVRT(src_prof, resampling=Resampling.nearest) as vrt_prof:
                valores_prof = list(vrt_prof.sample(coords))
                valores_prof = np.array(valores_prof).flatten()
                df[nombre_columna] = valores_prof

    # Aplicar Threshold de alturas si existe
    n = profundidades.keys().__len__()
    df[df < Threshold_H] = 0
    df = df[df.iloc[:, 1:n].sum(axis=1) != 0]

    return df

def BashFastFlood(JSONPath):

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
    log = open(log_file, 'w')

    # ------------------------------------------------------------------------------------------------------------------
    # Crear las carpetas para la ejecución de FastFlood
    # ------------------------------------------------------------------------------------------------------------------
    ProjectPath = os.path.join(UserData['ProjectPath'], UserData["NameBasinFolder"])
    CreateFolders(ProjectPath)

    '''
    # ------------------------------------------------------------------------------------------------------------------
    # Descargar: DEM, Infiltración, Manning, LULC, IDF y Factores Cambio Climático
    # ------------------------------------------------------------------------------------------------------------------
    ResultsPath = { 'DEM': os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"],'in','06-FLOOD','Raster','DEM.tif'),
                    'Manning': os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"],'in','06-FLOOD','Raster','Manning.tif'),
                    'LULC': os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"],'in','06-FLOOD','Raster','Lu.tif'),
                    'Infiltration': os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"],'in','06-FLOOD','Raster','Infiltration.tif'),
                    'IDF': os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"],'in','06-FLOOD','Raster','IDF.csv'),
                    'Fac_CC_Path': os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"],'in','06-FLOOD','Raster','Factor_ClimateChange.csv')}

    # Ojo, incluir el parámetro de la url del repositorio TNC
    # customurl=AccessData["customurl"],
    DownloadInputs(FastFloodPath=UserData['FastFloodPath'],
                   customurl=AccessData["customurl"],
                   Basin_shp_BoundingBox=UserData['CatchmentPath'],
                   DemResolution=UserData['DemResolution'],
                   log=log,
                   DEM_Path=ResultsPath['DEM'],
                   Manning_Path=ResultsPath['Manning'],
                   LULC_Path=ResultsPath['LULC'],
                   Inf_Path=ResultsPath['Infiltration'],
                   IDF_Path=ResultsPath['IDF'],
                   Fac_CC_Path=ResultsPath['Fac_CC_Path'])    
    #'''

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
                             log=log)

    # ------------------------------------------------------------------------------------------------------------------
    # Check - Se verifica que las profundidades
    # ------------------------------------------------------------------------------------------------------------------
    '''
    for TR_i in UserData["ClimateParams"]["ReturnPeriod"]:
        H_Path_BaU = ProjectPath + f'/out/06-FLOOD/Flood/Flood_BaU_TR-{TR_i}.tif'
        H_Path_NbS = ProjectPath + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR_i}_Tmp.tif'
        H_Path_New = ProjectPath + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR_i}.tif'
        # Check
        CheckPixelDepth_BAU_NBS(H_Path_BaU, H_Path_NbS, H_Path_New)
        # Remover el raster temporal
        os.remove(H_Path_NbS )
    # '''

    # ------------------------------------------------------------------------------------------------------------------
    # Step 7 - Leer curva de factors de daño y el costo máximo
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Estimate of expected annual damage for each scenario \n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    #'''

    # Leer curvas de daño
    DC = ReadDamageCurve(ProjectPath=os.path.join(UserData['ProjectPath'],UserData["NameBasinFolder"]))
    log.write(f"Read Dagame Curve - Ok \n")

    '''
    Ojo! mirar el tema de la tasa de cambio para estar acorde con la moneda de análisis de WaterProof
    '''
    # Se aplica la tasa de cambio en la cual se entregan los costos máximos de las funciones de daño
    DC = DC*UserData["DamagesExchangeRate"]


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
    # que se consideran para evaluar en cada categoria de daño.
    CodeLULC = {}
    CodeLULC['Residential'] = [11, 12, 13, 14, 15]
    CodeLULC['Commercial']  = [21, 22, 23, 24, 25]
    CodeLULC['Industrial']  = [21, 22, 23, 24, 25]
    CodeLULC['InfraRoads']  = [1]
    CodeLULC['Agriculture'] = [1]

    # Las funciones de costo están en las siguientes unidades
    # Residential ($/m^2)
    # Commercial ($/m^2)
    # Industrial ($/m^2)
    # InfraRoads ($/Km)
    # Agriculture ($/m^2)'
    # Dado que los rasters de las categorías de daño están con una resolución de 10 metros
    # Estos corresponde a los valores de área/km de cada pixel para tener $
    # Dado que la fuente de datos que se está utilizando para identificar el uso comercial
    # e industrial es GHS_BUILT_C y esta solo mapea el uso no residencial, se aplica
    # el factor extra de distribución ingresado por el usuario
    FactorArea = {}
    FactorArea['Residential']   = 10*10 #m^2
    FactorArea['Commercial']    = 10*10*UserData["SplitArea"]["Commercial"] #m^2
    FactorArea['Industrial']    = 10*10*UserData["SplitArea"]["Industrial"]  #m^2
    FactorArea['InfraRoads']    = 10 #km
    FactorArea['Agriculture']   = 10*10 #m^2

    LULC_Damage = {}
    LULC_Damage['Residential'] = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C.tif'
    LULC_Damage['Commercial']  = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C.tif'
    LULC_Damage['Industrial']  = f'{ProjectPath}/in/06-FLOOD/Raster/GHS_BUILT_C.tif'
    LULC_Damage['InfraRoads']  = f'{ProjectPath}/in/06-FLOOD/Raster/Road.tif'
    LULC_Damage['Agriculture'] = f'{ProjectPath}/in/06-FLOOD/Raster/Agricultural.tif'

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Read Flood Depth - Scenario Current\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    # Categorias de daño
    Cat_Damage = ['Residential', 'Commercial', 'Industrial', 'InfraRoads', 'Agriculture']
    # Nombre de los escenarios
    NameSce     = ['Current', 'BaU', 'NbS']
    # Crear un DataFrame vacío para almacenar resultados
    Total_EAD =  pd.DataFrame(columns=Cat_Damage,index=NameSce)
    for Sce in NameSce:
        for Cat in Cat_Damage:
            # ----------------------------------------------------------------------------------------------------------
            # Leer profundidades
            # ----------------------------------------------------------------------------------------------------------
            df = Raster2DataFrame(LULC_Damage[Cat], HPaths[Sce], CodeLULC[Cat], UserData["CatchmentPath"])
            df.to_csv( f'{UserData["ProjectPath"]}/{UserData["NameBasinFolder"]}/out/06-FLOOD/Flood/H_{Cat}_{Sce}.csv', index=False)
            log.write(f"Read flood depth for the {Sce} scenario for {Cat} damages category- Ok \n")

            # ----------------------------------------------------------------------------------------------------------
            # Estimar costos
            # ----------------------------------------------------------------------------------------------------------
            if 'Code' in df.columns:
                df.drop('Code', axis=1, inplace=True)

            df = TR_Damage(df, DC, category=Cat)*FactorArea[Cat]
            df.to_csv(f'{UserData["ProjectPath"]}/{UserData["NameBasinFolder"]}/out/06-FLOOD/Damages/01-Damage_{Cat}_{Sce}.csv', index=False)
            log.write(f"Damage estimation for each return period of the {Sce} scenario {Cat} - Ok \n")

            # ----------------------------------------------------------------------------------------------------------
            # Step 9 - Estimar el daño anual esperado
            # ----------------------------------------------------------------------------------------------------------
            # Este factor corresponde a la división de área
            OutEAD = EAD(TR=UserData['ClimateParams']['ReturnPeriod'], Damage=df, NameCol=Cat)
            OutEAD.to_csv(os.path.join(ProjectPath, 'out', '06-FLOOD', 'Damages', f'02-Expected_Annual_Damage_{Cat}_{Sce}.csv'))
            log.write(f"Estimated annual expected damages for the {Sce} scenario for {Cat} damages category- Ok- Ok \n")

            # Agregar datos
            Total_EAD.loc[Sce][Cat] = OutEAD.sum().sum()
            log.write(f"Estimated cumulative annual expected damages for the {Sce} scenario for {Cat} damages category- Ok- Ok \n")

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
    NBS[:, 2] = np.sum(NBS[:, 2:], 1)
    NBS     = NBS[:,0:3]

    # Leer datos de tiempo
    Time    = pd.read_csv(os.path.join(ProjectPath,'in','05-DISAGGREGATION','01-INPUTS_Time.csv')).values[0][0]

    # Desagregar datos
    Results_BaU, Results_NBS = DesaggregationData(Total_EAD, Cat_Damage, NBS, Time, CO2_BaU, CO2_NBS)

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

    # Cerrar Log
    log.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Step 12 - Indicadores
    # ------------------------------------------------------------------------------------------------------------------
    #"""
