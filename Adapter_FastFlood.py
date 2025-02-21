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
    Crea las carpetas 'in/Raster', 'in/GeoJSON' y 'out' en la ruta base especificada.

    :param base_path: Ruta base donde se crearán las carpetas.
    '''
    try:
        # Definir las rutas de las carpetas
        folders = [
            os.path.join(base_path, 'Flood', 'in', 'Raster'),
            os.path.join(base_path, 'Flood', 'in', 'Damages'),
            os.path.join(base_path, 'Flood', 'in', 'GeoJSON'),
            os.path.join(base_path, 'Flood', 'out', 'Damages'),
            os.path.join(base_path, 'Flood', 'out', 'Discharge'),
            os.path.join(base_path, 'Flood', 'out', 'Flood'),
            os.path.join(base_path, 'Flood', 'out', 'Velocity')
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
    las coordenadas transformadas como [minx, miny, maxx, maxy].

    :param shp_path: Ruta al archivo shapefile.
    :return: Lista con las coordenadas transformadas [minx, miny, maxx, maxy].
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


def BashFastFlood(JSONPath):

    wsl = ''

    # ------------------------------------------------------------------------------------------------------------------
    # Step 1 - Leer JSON con parámetros de ejecución
    # ------------------------------------------------------------------------------------------------------------------
    with open(JSONPath, 'r') as json_data:
        UserData = json.load(json_data)

    Key = 'hmrSW5oM3gxuUtpWWzLugSxj5ekDTFKP'
    ResultsPath = { 'DEM': UserData['ProjectPath'] + '/in/06-FLOOD/Raster/DEM.tif',
                    'Manning': UserData['ProjectPath'] + '/in/06-FLOOD/Raster/Manning.tif',
                    'Infiltration': UserData['ProjectPath'] + '/in/06-FLOOD/Raster/Infiltration.tif',
                    'IDF': UserData['ProjectPath'] + '/in/06-FLOOD/Raster/IDF.csv' }

    # Nombre del archivo log
    log_file = UserData["ProjectPath"] + f'/log_FastFlood.txt'

    # Abrir el archivo log en modo escritura
    log = open(log_file, 'w')

    # ------------------------------------------------------------------------------------------------------------------
    # Step 2 - Crear carpeta para proyecto FastFlood
    # ------------------------------------------------------------------------------------------------------------------
    CreateFolders(UserData['ProjectPath'])

    # ------------------------------------------------------------------------------------------------------------------
    # Step 3 - Boundary
    # ------------------------------------------------------------------------------------------------------------------
    BasinBox = Get_Basin_bbox(UserData['CatchmentPath'])

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Boundary\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f'ulx: {BasinBox[0]} uly: {BasinBox[1]} brx: {BasinBox[2]} bry: {BasinBox[3]}\n\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Step 4 - Descargar DEM
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download DEM\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    Comando = [wsl, UserData['FastFloodPath'],
               '-key',Key,
               '-d_dem','cop30',f'{UserData['DemResolution']}m',
               f'{BasinBox[0]}', f'{BasinBox[1]}', f'{BasinBox[2]}', f'{BasinBox[3]}',
               '-dout',ResultsPath['DEM']]

    #ExeFastFlood(Comando,log)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 5 - Descargar n-Manning
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download n-Manning\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    Comando = [wsl, UserData['FastFloodPath'],
               '-key', Key,
               '-dem', ResultsPath['DEM'],
               '-d_lu',
               '-manout', ResultsPath['Manning']]

    #ExeFastFlood(Comando,log)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 6 - Descargar Infiltración
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download Infiltration\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    Comando = [wsl, UserData['FastFloodPath'],
               '-key', Key,
               '-dem', ResultsPath['DEM'],
               '-d_inf',
               '-ksatout', ResultsPath['Infiltration']]

    #ExeFastFlood(Comando,log)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 7 - Descargar IDF
    # ------------------------------------------------------------------------------------------------------------------
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write("# Download IDF\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")

    Comando = [wsl, UserData['FastFloodPath'],
               '-key', Key,
               '-dem', ResultsPath['DEM'],
               '-idfout', ResultsPath['IDF']]

    #ExeFastFlood(Comando,log)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 8 - Ejecutar FastFlood - Current
    # ------------------------------------------------------------------------------------------------------------------
    D   = UserData["ClimateParams"]["StormDuration"]
    P   = UserData["ClimateParams"]["Period"]
    Q   = UserData["ClimateParams"]["StormQuantile"]
    SSP = UserData["ClimateParams"]["Scenario"]

    raster_paths = []
    for TR in UserData["ClimateParams"]["ReturnPeriod"]:
        H_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Flood/Flood_Current_TR-{TR}.tif'
        V_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Velocity/Velocity_Current_TR-{TR}.tif'
        Q_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Discharge/Qpeak_Current_TR-{TR}.tif'

        # Crear lista de rasters
        raster_paths.append(H_Path)

        if UserData["ClimateParams"]["Scenario"] == "Historic":
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: Current | Climate: Historic | TR: {TR} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = [wsl, UserData['FastFloodPath'],
                       '-key', Key,
                       '-sim',
                       '-dem', ResultsPath['DEM'],
                       '-man', ResultsPath['Manning'],
                       '-inf', ResultsPath['Infiltration'],
                       '-designstorm',f'{TR}',f'{D}',
                       '-dur',f'{D}',
                       '-whout', H_Path,
                       '-qout', Q_Path,
                       '-vout',V_Path]
        else:
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: Current | Climate: Scenario: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = [wsl, UserData['FastFloodPath'],
                       '-key', Key,
                       '-sim',
                       '-dem', ResultsPath['DEM'],
                       '-man', ResultsPath['Manning'],
                       '-inf', ResultsPath['Infiltration'],
                       '-climate', SSP, f'{P}', f'{Q}', f'{TR}', f'{D}',
                       '-whout', H_Path,
                       '-qout', Q_Path,
                       '-vout', V_Path]

        #ExeFastFlood(Comando,log)

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Read Flood Depth - Scenario Current\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    H_TR_C = Raster2DataFrame(raster_paths)
    log.write(f"\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - BaU
    # ------------------------------------------------------------------------------------------------------------------
    PathGeoJSON = UserData['ProjectPath'] + f'/in/06-FLOOD/GeoJSON/BaU.geojson'
    raster_paths = []
    for TR in UserData["ClimateParams"]["ReturnPeriod"]:
        H_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Flood/Flood_BaU_TR-{TR}.tif'
        V_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Velocity/Velocity_BaU_TR-{TR}.tif'
        Q_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Discharge/Qpeak_BaU_TR-{TR}.tif'

        # Crear lista de rasters
        raster_paths.append(H_Path)

        if UserData["ClimateParams"]["Scenario"] == "Historic":
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: BaU | Climate: Historic | TR: {TR} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = [wsl, UserData['FastFloodPath'],
                       '-key', Key,
                       '-sim',
                       '-dem', ResultsPath['DEM'],
                       '-man', ResultsPath['Manning'],
                       '-inf', ResultsPath['Infiltration'],
                       '-designstorm', f'{TR}', f'{D}',
                       '-dur', f'{D}',
                       '-adaptation', PathGeoJSON,
                       '-whout', H_Path,
                       '-qout', Q_Path,
                       '-vout', V_Path]
        else:
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: Bau | Climate : {SSP} | Period: {P} | Quantile: {Q} | TR: {TR} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = [wsl, UserData['FastFloodPath'],
                       '-key', Key,
                       '-sim',
                       '-dem', ResultsPath['DEM'],
                       '-man', ResultsPath['Manning'],
                       '-inf', ResultsPath['Infiltration'],
                       '-climate', SSP, f'{P}', f'{Q}', f'{TR}', f'{D}',
                       '-adaptation', PathGeoJSON,
                       '-whout', H_Path,
                       '-qout', Q_Path,
                       '-vout', V_Path]

        #ExeFastFlood(Comando,log)

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Read Flood Depth - Scenario BaU\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    H_TR_BaU = Raster2DataFrame(raster_paths)
    log.write(f"\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Step 9 - Ejecutar FastFlood - NbS
    # ------------------------------------------------------------------------------------------------------------------
    PathGeoJSON = UserData['ProjectPath'] + f'/in/06-FLOOD/GeoJSON/NbS.geojson'
    raster_paths = []
    for TR in UserData["ClimateParams"]["ReturnPeriod"]:
        H_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Flood/Flood_NbS_TR-{TR}.tif'
        V_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Velocity/Velocity_NbS_TR-{TR}.tif'
        Q_Path = UserData['ProjectPath'] + f'/out/06-FLOOD/Discharge/Qpeak_NbS_TR-{TR}.tif'

        # Crear lista de rasters
        raster_paths.append(H_Path)

        if UserData["ClimateParams"]["Scenario"] == "Historic":
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: NbS | Climate: Historic | TR: {TR} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = [wsl, UserData['FastFloodPath'],
                       '-key', Key,
                       '-sim',
                       '-dem', ResultsPath['DEM'],
                       '-man', ResultsPath['Manning'],
                       '-inf', ResultsPath['Infiltration'],
                       '-designstorm', f'{TR}', f'{D}',
                       '-dur', f'{D}',
                       '-adaptation',PathGeoJSON,
                       '-whout', H_Path,
                       '-qout', Q_Path,
                       '-vout', V_Path]
        else:
            log.write("# ---------------------------------------------------------------------------------------------------\n")
            log.write(f"# Execution FastFlood | Scenario: NbS | Climate: {SSP} | Period: {P} | Quantile: {Q} | TR: {TR} | Duration: {D} \n")
            log.write("# ---------------------------------------------------------------------------------------------------\n")

            Comando = [wsl, UserData['FastFloodPath'],
                       '-key', Key,
                       '-sim',
                       '-dem', ResultsPath['DEM'],
                       '-man', ResultsPath['Manning'],
                       '-inf', ResultsPath['Infiltration'],
                       '-climate', SSP, f'{P}', f'{Q}', f'{TR}', f'{D}',
                       '-adaptation', PathGeoJSON,
                       '-whout', H_Path,
                       '-qout', Q_Path,
                       '-vout', V_Path]

        #ExeFastFlood(Comando,log)

    log.write("# ---------------------------------------------------------------------------------------------------\n")
    log.write(f"# Read Flood Depth - Scenario NbS\n")
    log.write("# ---------------------------------------------------------------------------------------------------\n")
    H_TR_NbS = Raster2DataFrame(raster_paths)
    log.write(f"\n")

    # Cerrar Log
    log.close()

JSONPath = r'/home/nogales/00-TNC/02-Adapter/PROJECT/WI_1785/FastFlood.json'

BashFastFlood(JSONPath)