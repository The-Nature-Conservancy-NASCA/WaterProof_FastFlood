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
from osgeo import ogr
import geopandas as gpd
from pyproj import Transformer, CRS
import pandas as pd
from pyproj import CRS, Transformer
import json
import rasterio
import time

def CreateFolders(base_path):
    '''
    Crea las carpetas requeridas para el análisis utilizando una ruta base especificada.

    :param base_path: Ruta base donde se crearán las carpetas.
    '''
    try:
        # Rutas de las carpetas
        folders = [
            os.path.join(base_path, 'Flood', 'in', 'Raster'),
            os.path.join(base_path, 'Flood', 'in', 'Damages'),
            os.path.join(base_path, 'Flood', 'in', 'Shp'),
            os.path.join(base_path, 'Flood', 'out', 'Damages'),
            os.path.join(base_path, 'Flood', 'out', 'Discharge'),
            os.path.join(base_path, 'Flood', 'out', 'Flood'),
            os.path.join(base_path, 'Flood', 'out', 'Velocity'),
            os.path.join(base_path, 'Flood', 'out', 'Other')
        ]

        # Crear cada carpeta si no existe
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f'Carpeta creada: {folder}')

        print('Todas las carpetas han sido creadas con éxito.')
    except Exception as e:
        print(f'Error al crear las carpetas: {e}')


def ExeFastFlood(Comando, log):

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
        log.write(f'[OUTPUT] {salida}')  # Guardar en el log

    # Escribir los errores en el log
    if error:
        print(error.strip(), file=sys.stderr, flush=True)  # Mostrar el error en la consola
        log.write(f'[ERROR] {error}\n\n')  # Guardar en el log


def Raster2DataFrame(raster_paths):
    """
    Lee una lista de archivos TIFF (raster) y crea un DataFrame con los valores de los píxeles.

    Parámetros:
        raster_paths (list): Lista de rutas a los archivos TIFF.

    Retorna:
        pd.DataFrame: Un DataFrame donde cada columna representa los valores de los píxeles de un raster.
    """
    # Verificar que la lista no esté vacía
    if not raster_paths:
        raise ValueError("La lista de rutas de rasters está vacía.")

    # Lista para almacenar los datos de cada raster
    raster_data = []

    # Leer cada raster y almacenar sus valores en la lista
    for path in raster_paths:
        with rasterio.open(path) as src:
            # Leer la banda 1 del raster (asumiendo que es un raster de una sola banda)
            band = src.read(1)
            # Aplanar la matriz de píxeles a un array 1D
            flattened_band = band.flatten()
            raster_data.append(flattened_band)

    # Crear un DataFrame con los datos de los rasters
    df = pd.DataFrame(raster_data).T  # Transponer para que cada columna sea un raster

    # Asignar nombres a las columnas (opcional)
    df.columns = [f"TR_{i}" for i in [2, 5, 10, 20, 40, 50, 100, 200, 500, 1000]]

    return df


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

def DownloadInputs(FastFloodPath,Key,BasinBox,DemResolution,log,DEM_Path,Manning_Path,Inf_Path,IDF_Path,Fac_CC_Path,Buffer_km=3,
                   Status_DEM=True, Status_Inf=True,Status_n=True,Status_IDF=True, customurl=None):

    # ------------------------------------------------------------------------------------------------------------------
    # Step 4 - Descargar DEM
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download DEM\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    # Buffer para el dominio
    Buffer_m    = Buffer_km*1000
    BasinBox[0] = BasinBox[0] - Buffer_m
    BasinBox[1] = BasinBox[1] + Buffer_m
    BasinBox[2] = BasinBox[2] + Buffer_m
    BasinBox[3] = BasinBox[3] - Buffer_m

    Comando = CommandFastFlood('Download_DEM',
                               FastFloodPath, Key, customurl,
                               BasinBox=BasinBox,
                               DemResolution=DemResolution,
                               DEM_Path=DEM_Path)
    if Status_DEM:
        ExeFastFlood(Comando,log)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 5 - Descargar n-Manning
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download n-Manning\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    Comando = CommandFastFlood('Download_Manning',
                               FastFloodPath, Key, customurl,
                               DEM_Path=DEM_Path,
                               Manning_Path=Manning_Path)

    if Status_n:
        ExeFastFlood(Comando,log)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 6 - Descargar Infiltración
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download Infiltration\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    Comando = CommandFastFlood('Download_Infiltration',
                               FastFloodPath, Key, customurl,
                               DEM_Path=DEM_Path,
                               Inf_Path=Inf_Path)

    if Status_Inf:
        ExeFastFlood(Comando,log)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 7 - Descargar IDF
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download IDF\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    Comando = CommandFastFlood('Download_IDF',
                                FastFloodPath, Key, customurl,
                                DEM_Path=DEM_Path,
                                IDF_Path=IDF_Path,
                                Fac_CC_Path=Fac_CC_Path)

    if Status_IDF:
        ExeFastFlood(Comando,log)

def CommandFastFlood(NameCommand,
                     FastFloodPath, Key, customurl=None,
                     BasinBox=None, DemResolution=None,
                     DEM_Path=None, Manning_Path=None, Inf_Path=None, IDF_Path=None, Fac_CC_Path=None,
                     D=None, P=None, Q=None, SSP=None, TR=None,
                     H_Path=None, Q_Path=None, V_Path=None, nOut=None, InfOut=None, PathShp=None):

    # iniciar comando
    Comando = [FastFloodPath, '-key', Key]

    if customurl is not None:
        Comando += ['-customurl',customurl]
    if NameCommand == 'Run':
        Comando += ['-sim']
    if NameCommand == 'Download_DEM':
        Comando += ['-d_dem', 'cop30', f'{DemResolution}m',
                    f'{BasinBox[0]}', f'{BasinBox[1]}', f'{BasinBox[2]}',f'{BasinBox[3]}',
                    '-dout', DEM_Path]
    if (DEM_Path is not None) and (NameCommand is not 'Download_DEM'):
        Comando += ['-dem', DEM_Path]
    if NameCommand == 'Download_Manning':
        Comando += ['-d_lu','-manout', Manning_Path]
    if NameCommand == 'Download_Infiltration':
        Comando += ['-d_inf','-ksatout', Inf_Path]
    if NameCommand == 'Download_IDF':
        Comando += ['-idfout', IDF_Path,'-climout', Fac_CC_Path]
    if (Manning_Path is not None) and (NameCommand is not 'Download_Manning'):
        Comando += ['-man', Manning_Path]
    if (Inf_Path is not None) and (NameCommand is not 'Download_Infiltration'):
        Comando += ['-inf', Inf_Path]
    if (TR is not None) and (D is not None) and (SSP == "Historic"):
        Comando += ['-designstorm', f'{TR}', f'{D}']
    if (D is not None) and (SSP == "Historic"):
        Comando += ['-dur', f'{D}']
    if (P is not None) and (Q is not None) and (TR is not None) and (D is not None) and (SSP == "ClimateChange"):
        Comando += ['-climate', SSP, f'{P}', f'{Q}', f'{TR}', f'{D}']
    if PathShp is not None:
        Comando += ['-adaptation', PathShp]
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

    return Comando

def RunScenarios(ProjectPath, FastFloodPath, Key, D, P, Q, SSP, TR, DEM_Path, Manning_Path, Inf_Path, log, customurl=None):

    # ------------------------------------------------------------------------------------------------------------------
    # Step 8 - Ejecutar FastFlood - Current
    # ------------------------------------------------------------------------------------------------------------------
    raster_paths = []
    for TR_i in TR:
        H_Path = ProjectPath + f'/out/06-FLOOD/Flood/Flood_Current_TR-{TR_i}.tif'
        V_Path = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_Current_TR-{TR_i}.tif'
        Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_Current_TR-{TR_i}.tif'

        # Crear lista de rasters
        raster_paths.append(H_Path)

        if SSP == "Historic":
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: Current | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                         FastFloodPath, Key, customurl,
                                         DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                         SSP=SSP, TR=TR_i, D=D,
                                         H_Path=H_Path, Q_Path=Q_Path, V_Path=V_Path)

        else:
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: Current | Climate: Scenario: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                         FastFloodPath, Key, customurl,
                                         DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                         SSP=SSP, TR=TR_i, D=D, P=P, Q=Q,
                                         H_Path=H_Path, Q_Path=Q_Path, V_Path=V_Path)

        # Ejecutar comando
        ExeFastFlood(Comando,log)

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Read Flood Depth - Scenario Current\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    H_TR_C = Raster2DataFrame(raster_paths)
    log.write(f"\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - BaU
    # ------------------------------------------------------------------------------------------------------------------
    # Infiltración modificada por la opción de adaptación - BaU
    InfOut  = ProjectPath + f'/in/06-FLOOD/Raster/Infiltration_BaU.tif'
    # n-Manning modificado por la opción de adaptación - BaU
    nOut    = ProjectPath + f'/in/06-FLOOD/Raster/n-Manning_BaU.tif'
    # Shp de cambios del escenario BaU
    PathShp = ProjectPath + f'/in/06-FLOOD/Shp/BaU.shp'

    raster_paths = []
    for TR_i in TR:
        H_Path = ProjectPath + f'/out/06-FLOOD/Flood/Flood_BaU_TR-{TR_i}.tif'
        V_Path = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_BaU_TR-{TR_i}.tif'
        Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_BaU_TR-{TR_i}.tif'

        # Crear lista de rasters
        raster_paths.append(H_Path)

        if SSP == "Historic":
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: BaU | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, Key, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D,
                                       PathShp=PathShp,
                                       H_Path=H_Path, Q_Path=Q_Path, V_Path=V_Path)

        else:
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: Bau | Climate : {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, Key, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D, P=P, Q=Q,
                                       PathShp=PathShp,
                                       H_Path=H_Path, Q_Path=Q_Path, V_Path=V_Path)
        if TR_i == 2:
            Comando += ['-manout', nOut,
                        '-ksatout', InfOut]

        # Ejecutar comando
        # ExeFastFlood(Comando,log)

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Read Flood Depth - Scenario BaU\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    H_TR_BaU = Raster2DataFrame(raster_paths)
    log.write(f"\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - NbS
    # ------------------------------------------------------------------------------------------------------------------
    # Infiltración modificada por la opción de adaptación - NbS
    InfOut = ProjectPath + f'/in/06-FLOOD/Raster/Infiltration_NbS.tif'
    # n-Manning modificado por la opción de adaptación - NbS
    nOut = ProjectPath + f'/in/06-FLOOD/Raster/n-Manning_NbS.tif'
    # Shp de cambios del escenario NbS
    PathShp = ProjectPath + f'/in/06-FLOOD/Shp/NbS.shp'
    raster_paths = []
    for TR_i in TR:
        H_Path = ProjectPath + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR_i}.tif'
        V_Path = ProjectPath + f'/out/06-FLOOD/Velocity/Velocity_NbS_TR-{TR_i}.tif'
        Q_Path = ProjectPath + f'/out/06-FLOOD/Discharge/Qpeak_NbS_TR-{TR_i}.tif'

        # Crear lista de rasters
        raster_paths.append(H_Path)

        if SSP == "Historic":
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: NbS | Climate: Historic | TR: {TR_i} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, Key, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D,
                                       PathShp=PathShp,
                                       H_Path=H_Path, Q_Path=Q_Path, V_Path=V_Path)
        else:
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: NbS | Climate: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR_i} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = CommandFastFlood("Run",
                                       FastFloodPath, Key, customurl,
                                       DEM_Path=DEM_Path, Manning_Path=Manning_Path, Inf_Path=Inf_Path,
                                       SSP=SSP, TR=TR_i, D=D, P=P, Q=Q,
                                       PathShp=PathShp,
                                       H_Path=H_Path, Q_Path=Q_Path, V_Path=V_Path)

        if TR_i == 2:
            Comando += ['-manout', nOut,
                        '-ksatout', InfOut]

        # Ejecutar comando
        # ExeFastFlood(Comando,log)

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Read Flood Depth - Scenario NbS\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    H_TR_NbS = Raster2DataFrame(raster_paths)
    log.write(f"\n")

    return H_TR_C, H_TR_BaU, H_TR_NbS

def DamageCurve():
    o = 1

def BashFastFlood(JSONPath):

    # ------------------------------------------------------------------------------------------------------------------
    # Step 1 - Leer JSON con parámetros de ejecución
    # ------------------------------------------------------------------------------------------------------------------
    with open(JSONPath, 'r') as json_data:
        UserData = json.load(json_data)

    with open(UserData["AccessData"], 'r') as json_data:
        AccessData = json.load(json_data)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 2 - Crear archivo log
    # ------------------------------------------------------------------------------------------------------------------
    # Nombre del archivo log
    log_file = UserData["ProjectPath"] + f'/log_FastFlood.txt'

    # Abrir el archivo log en modo escritura
    log = open(log_file, 'w')

    # ------------------------------------------------------------------------------------------------------------------
    # Step 3 - Crear carpeta para proyecto FastFlood
    # ------------------------------------------------------------------------------------------------------------------
    CreateFolders(UserData['ProjectPath'])

    # ------------------------------------------------------------------------------------------------------------------
    # Step 4 - Extraer Boundary del shapefile d la cuenca
    # ------------------------------------------------------------------------------------------------------------------
    BasinBox = Get_Basin_bbox(UserData['CatchmentPath'])

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Boundary\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f'ulx: {BasinBox[0]} uly: {BasinBox[1]} brx: {BasinBox[2]} bry: {BasinBox[3]}\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Step 5 - Descargar DEM, Infiltración, Manning y IDF
    # ------------------------------------------------------------------------------------------------------------------
    ResultsPath = { 'DEM': UserData['ProjectPath'] + '/in/06-FLOOD/Raster/DEM.tif',
                    'Manning': UserData['ProjectPath'] + '/in/06-FLOOD/Raster/Manning.tif',
                    'Infiltration': UserData['ProjectPath'] + '/in/06-FLOOD/Raster/Infiltration.tif',
                    'IDF': UserData['ProjectPath'] + '/in/06-FLOOD/IDF.csv',
                    'Fac_CC_Path': UserData['ProjectPath'] + '/in/06-FLOOD/Factor_ClimateChange.csv',}

    DownloadInputs(FastFloodPath=UserData['FastFloodPath'],
                   Key=AccessData["Key"],
                   customurl=AccessData["customurl"],
                   BasinBox=BasinBox,
                   DemResolution=UserData['DemResolution'],
                   log=log,
                   DEM_Path=ResultsPath['DEM'],
                   Manning_Path=ResultsPath['Manning'],
                   Inf_Path=ResultsPath['Infiltration'],
                   IDF_Path=ResultsPath['IDF'],
                   Fac_CC_Path=ResultsPath['Fac_CC_Path'])

    # ------------------------------------------------------------------------------------------------------------------
    # Step 6 - Descargar DEM, Infiltración, Manning y IDF
    # ------------------------------------------------------------------------------------------------------------------
    [H_TR_C, H_TR_BaU, H_TR_NbS]  = RunScenarios(ProjectPath=UserData['ProjectPath'],
                                                 FastFloodPath=UserData['FastFloodPath'],
                                                 Key=AccessData["Key"],
                                                 customurl=AccessData["customurl"],
                                                 D=UserData["ClimateParams"]["StormDuration"],
                                                 P=UserData["ClimateParams"]["Period"],
                                                 Q=UserData["ClimateParams"]["StormQuantile"],
                                                 SSP=UserData["ClimateParams"]["Scenario"],
                                                 TR=UserData["ClimateParams"]["ReturnPeriod"],
                                                 DEM_Path=ResultsPath['DEM'],
                                                 Manning_Path=ResultsPath['Manning'],
                                                 Inf_Path=ResultsPath['Infiltration'],
                                                 log=log)

    # Cerrar Log
    log.close()

    # ------------------------------------------------------------------------------------------------------------------
    # Step 7 - Estimación de daño probabale
    # ------------------------------------------------------------------------------------------------------------------


JSONPath = r'/home/nogales/00-TNC/02-Adapter/PROJECT/WI_1785/FastFlood.json'
#BashFastFlood(JSONPath)


