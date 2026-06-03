# Documentación técnica del adaptador FastFlood para estimación de beneficios monetarios por mitigación de inundaciones en WaterProof

**Versión:** 0.2
**Componente:** Adaptador FastFlood integrado a WaterProof
**Función principal:** `BashFastFlood(JSONPath, SaveFullCSV=False)`
**Objetivo:** ejecutar el flujo completo de modelación de inundación, estimación de daños evitados, desagregación temporal, cálculo de ROI e indicadores asociados a portafolios de Soluciones basadas en la Naturaleza.

---

## 1. Propósito general del adaptador

El adaptador FastFlood es el componente de WaterProof encargado de traducir la configuración de un caso de estudio en una secuencia completa de modelación hidráulica y evaluación económica. Su función principal es tomar los insumos definidos por el usuario en WaterProof, construir los escenarios hidrológico-hidráulicos requeridos, ejecutar FastFlood para múltiples periodos de retorno, estimar daños por inundación para distintas categorías de exposición y transferir los resultados al módulo financiero de ROI.

El proceso está diseñado para evaluar el beneficio monetario de inversiones en Soluciones basadas en la Naturaleza en cuencas hidrográficas. Este beneficio se estima como la reducción del daño anual esperado entre un escenario sin inversión en SbN y un escenario con implementación de un portafolio de SbN. En términos operativos, las SbN se representan en FastFlood mediante cambios espaciales en el coeficiente de rugosidad de Manning y en la infiltración. Estos parámetros modifican la propagación del flujo superficial, la generación de escorrentía efectiva y la distribución espacial de profundidades, velocidades y caudales pico.

El adaptador no es únicamente una rutina de ejecución de FastFlood. Es un flujo integrado que incluye preparación de datos, homologación de coberturas, construcción de rásteres de escenario, ejecución hidráulica, control de coherencia física, corrección de hidrogramas, procesamiento espacial de exposición, integración daño-probabilidad, desagregación temporal, cálculo financiero y generación de indicadores.

---

## 2. Entradas principales del flujo

El flujo inicia con un archivo JSON de configuración. Este JSON actúa como contrato de entrada entre WaterProof y el adaptador. Allí se definen rutas, parámetros climáticos, parámetros hidráulicos, información de exposición, condiciones de borde y reglas de parametrización de SbN.

Las entradas se agrupan en siete bloques principales.

| Bloque                                 | Contenido técnico                                                                                                                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rutas del proyecto                     | Ruta raíz del proyecto, carpeta de cuenca, ejecutable FastFlood, archivo de acceso a datos, shapefile de cuenca y carpetas de entrada/salida.                                                     |
| Insumos ráster de FastFlood            | DEM, raster de Manning, raster de infiltración y raster de cobertura del suelo usado por FastFlood.                                                                                               |
| Insumos de escenarios WaterProof       | Cobertura actual, cobertura BaU y raster del portafolio de SbN.                                                                                                                                   |
| Parámetros de SbN                      | Tabla de valores de Manning e infiltración por combinación de tipo de SbN y cobertura intervenida.                                                                                                |
| Parámetros de degradación o cambio BaU | Tabla de cambios porcentuales de infiltración asociados a transiciones de cobertura entre condición actual y BaU.                                                                                 |
| Parámetros climáticos                  | Periodos de retorno, duración de tormenta de análisis, duración de tormenta de diseño histórica, duración de tormenta para cambio climático, escenario SSP, periodo futuro y percentil climático. |
| Parámetros de daño y ROI               | Curvas de daño, costos máximos, tasa de cambio, proporción comercial-industrial, costos de SbN, costos de operación, mantenimiento, oportunidad, transacción, plataforma y tasas de descuento.    |

El JSON también permite definir si existe interacción con el mar. Cuando el parámetro de condición de borde es distinto de cero, el adaptador incorpora ese nivel como condición de borde oceánica en FastFlood. Cuando el valor es cero, la simulación se ejecuta sin condición oceánica explícita.

---

## 3. Inicialización del proceso

La función `BashFastFlood` inicia leyendo el JSON principal y el archivo auxiliar de acceso a datos. Este último contiene la URL personalizada usada por el ejecutable de FastFlood para acceder a datos alojados en el entorno de WaterProof.

Después, el adaptador crea un archivo de log dentro de la carpeta de la cuenca. Este log registra el inicio del proceso, la ruta del JSON, la ruta del proyecto, el nombre de la cuenca, la memoria usada y los principales hitos de ejecución. El registro es relevante porque el flujo procesa rásteres y bases de exposición que pueden ser grandes; por tanto, se requiere trazabilidad en caso de fallos, interrupciones o inconsistencias.

Luego se crea la estructura estándar de carpetas del componente de inundación:

```text
in/06-FLOOD/Raster
in/06-FLOOD/Damages
in/06-FLOOD/Shp
out/06-FLOOD/Damages
out/06-FLOOD/Discharge
out/06-FLOOD/Flood
out/06-FLOOD/Velocity
out/06-FLOOD/Other
out/06-FLOOD/Tmp
```

Esta estructura separa insumos, resultados hidráulicos, resultados económicos, archivos temporales y productos intermedios. La separación es necesaria porque el adaptador produce múltiples salidas por escenario, periodo de retorno y categoría de daño.

---

## 4. Homologación de códigos de cobertura

El adaptador trabaja con dos sistemas de codificación de coberturas. El primero corresponde a las clases usadas por FastFlood, derivadas de su estructura de uso/cobertura del suelo. El segundo corresponde a las clases usadas por WaterProof para describir la cobertura actual, la cobertura BaU y las coberturas finales asociadas a SbN.

Para conectar ambos sistemas se definen dos diccionarios de equivalencia:

```text
FastFlood → WaterProof
10 → 2  Forest
20 → 7  Shrublands
30 → 3  Grassland
40 → 4  Agricultural
50 → 5  Building
60 → 6  Bare area
70 → 0  Ice
80 → 1  Water
```

```text
WaterProof/BaU → FastFlood
2 → 10  Forest
7 → 20  Shrublands
3 → 30  Grassland
4 → 40  Agricultural
5 → 50  Building
6 → 60  Bare area
0 → 70  Ice
1 → 80  Water
```

También se define un conjunto de categorías bloqueadas:

```text
{1, 50, 70, 80, 90, 100, 110}
```

Estas categorías no deben ser modificadas por la lógica de SbN. En términos prácticos, esto evita aplicar cambios de rugosidad o infiltración sobre píxeles que representan agua, hielo/nieve, áreas construidas u otras clases que no deben ser intervenidas por el portafolio.

---

## 5. Lectura y estructuración de parámetros desde el JSON

El adaptador convierte la información del JSON en cuatro estructuras operativas:

| Estructura                 | Uso                                                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `lulc_to_manning`          | Asocia cada código de cobertura FastFlood con un valor base de Manning.                                                           |
| `manning_dict`             | Asocia cada par `(tipo de SbN, cobertura WaterProof)` con un valor de Manning para el escenario NbS.                              |
| `infiltracion_dict`        | Asocia cada par `(tipo de SbN, cobertura WaterProof)` con un porcentaje de mejora de infiltración para el escenario NbS.          |
| `infiltration_change_dict` | Asocia cada transición `(cobertura inicial, cobertura BaU)` con un porcentaje de reducción de infiltración para el escenario BaU. |

La lógica es distinta para BaU y para NbS. En BaU se aplican cambios asociados a transición de cobertura, normalmente degradantes o conservadores. En NbS se aplican mejoras condicionadas por el raster de portafolio, es decir, solo donde WaterProof indicó intervención.

---

## 6. Alineación espacial de rásteres

Antes de construir los escenarios, el adaptador alinea todos los rásteres relevantes a una misma malla espacial. Los rásteres que se sincronizan son:

```text
lulc_fastflood
waterproof_current
bau_lulc
infiltracion_base
manning_base
portafolio
```

La malla de referencia es el raster de cobertura usado por FastFlood (`lulc_fastflood`). Cada raster se evalúa en términos de tamaño de píxel. Si un raster no coincide con la resolución de referencia, se remuestrea para que comparta resolución, extensión, número de filas, número de columnas y sistema de coordenadas.

Este paso es fundamental porque las reglas de escenario se aplican píxel a píxel. La condición `(NbS, cobertura)`, la transición `(cobertura actual, cobertura BaU)`, el valor base de infiltración y el valor base de Manning deben corresponder al mismo píxel físico. Si las capas no estuvieran alineadas, el adaptador podría aplicar una SbN o una transición de cobertura a una celda equivocada.

---

## 7. Construcción del raster de Manning para el escenario BaU

El escenario BaU representa la condición sin inversión en SbN. En este escenario, el Manning se recalcula a partir de la cobertura BaU y se compara contra la condición actual.

El proceso sigue esta lógica:

1. Se lee la cobertura FastFlood actual y la cobertura BaU alineada.
2. Se calcula un Manning inicial a partir de la cobertura FastFlood actual. Si se decide usar el raster base, puede leerse directamente el Manning de entrada; en la configuración actual se usa la tabla `lulc_to_manning`.
3. Se construye un raster temporal de Manning a partir de la cobertura BaU. Para cada código BaU se identifica su equivalencia con una clase FastFlood y luego se asigna el Manning correspondiente.
4. Para la clase de vegetación escasa se aplica una regla específica y se asigna un Manning de `0.02`.
5. En píxeles bloqueados se conserva el Manning inicial.
6. En píxeles no bloqueados se toma el mínimo entre el Manning inicial y el Manning temporal BaU.

La regla central es:

```text
Manning_BaU = Manning_Current, si la categoría está bloqueada

Manning_BaU = min(Manning_Current, Manning_BaU_temp), si la categoría no está bloqueada
```

La interpretación hidráulica es que el escenario BaU no debe generar una rugosidad artificialmente mayor que la condición actual cuando representa degradación o ausencia de intervención. Una reducción de Manning implica menor resistencia superficial, mayor velocidad potencial del flujo y menor capacidad de retención superficial.

Después de generar `Manning_BaU.tif`, el adaptador ejecuta un control adicional píxel a píxel: si el Manning BaU resulta mayor que el Manning actual, se reemplaza por el valor actual. Con esto se fuerza la condición:

```text
Manning_BaU ≤ Manning_Current
```

Esta regla evita que el escenario BaU sea hidráulicamente más favorable que el escenario actual en términos de rugosidad.

---

## 8. Construcción del raster de infiltración para el escenario BaU

El raster `Infiltration_BaU.tif` se construye a partir de la infiltración base y de las transiciones de cobertura entre la condición actual y el escenario BaU.

El proceso es el siguiente:

1. Se leen cuatro capas alineadas: cobertura FastFlood, cobertura actual WaterProof, cobertura BaU e infiltración base.
2. La cobertura FastFlood se traduce a códigos WaterProof cuando existe correspondencia.
3. Se identifica la cobertura inicial por píxel. Cuando existe mapeo FastFlood → WaterProof, se usa esa equivalencia; cuando no existe, se conserva el código original.
4. Para cada píxel se evalúa la transición `(cobertura actual, cobertura BaU)`.
5. Si esa transición existe en `infiltration_change_dict`, se aplica una reducción sobre la infiltración base.
6. La reducción se calcula usando el valor absoluto del cambio porcentual. Si el cambio es menor a 100 %, el valor final se calcula como:

```text
Infiltration_BaU = Infiltration_Base · (1 - |Cambio|/100)
```

7. Si el cambio es igual o mayor a 100 %, la infiltración se lleva a cero:

```text
Infiltration_BaU = 0
```

8. Luego se repite la evaluación usando la cobertura inicial homologada. Esta segunda pasada cubre casos en los que la equivalencia de códigos permite detectar transiciones no capturadas con la primera lectura.
9. Finalmente, el raster BaU se limita para que nunca supere la infiltración base:

```text
Infiltration_BaU = min(Infiltration_BaU, Infiltration_Base)
```

La interpretación es que BaU no puede mejorar la capacidad de infiltración respecto a la condición actual. Si la transición de cobertura implica degradación o urbanización, la infiltración se reduce. Si no existe regla de transición, el valor base se conserva.

Después de guardar el raster, el adaptador aplica un control adicional con la misma lógica: si la infiltración BaU es mayor que la infiltración actual, se reemplaza por la infiltración actual. Con esto se fuerza:

```text
Infiltration_BaU ≤ Infiltration_Current
```

---

## 9. Construcción del raster de Manning para el escenario NbS

El escenario NbS parte del escenario BaU y aplica cambios únicamente en los píxeles donde existe intervención del portafolio de SbN.

La construcción de `Manning_NbS.tif` sigue esta lógica:

1. Se lee la cobertura FastFlood, la cobertura WaterProof, el raster del portafolio de SbN, el Manning BaU y el Manning actual.
2. El raster final se inicializa como una copia del Manning BaU. Esto significa que, por defecto, todo píxel sin SbN mantiene la condición BaU.
3. Se identifica la presencia de SbN. Si el raster del portafolio tiene valor NoData, se usa esa marca para excluir píxeles; si no tiene NoData, se consideran intervenidos los píxeles con valor mayor que cero.
4. Se identifican las categorías bloqueadas usando la cobertura FastFlood.
5. Se construye una máscara candidata:

```text
candidate_mask = has_nbs AND not_locked
```

6. Para los píxeles candidatos se busca el valor de Manning asociado al par:

```text
(tipo de SbN, cobertura WaterProof)
```

7. Si existe un valor en `manning_dict`, se asigna como Manning temporal.
8. La primera regla aplica el valor temporal cuando representa una mejora respecto al Manning inicial:

```text
Si Manning_temp > Manning_Current_base:
    Manning_NbS = Manning_temp
```

9. La segunda regla cubre píxeles donde existe SbN y valor temporal válido, pero el valor no supera el Manning inicial. En ese caso, el adaptador vuelve a consultar la tabla `(tipo de SbN, cobertura)` y asigna el valor definido si existe. Esta regla evita perder intervenciones por diferencias entre el valor base y el valor parametrizado.
10. La tercera regla compara contra el escenario actual. Si el Manning NbS calculado queda por debajo del Manning actual, se reemplaza por el valor actual:

```text
Si Manning_NbS < Manning_Current:
    Manning_NbS = Manning_Current
```

11. Finalmente, después de guardar el raster, se aplica un control adicional contra BaU:

```text
Manning_NbS ≥ Manning_BaU
```

Esta condición se implementa revisando píxel a píxel: si el Manning NbS es menor que el Manning BaU, se reemplaza por el Manning BaU.

La lógica hidráulica es clara: una SbN que mejora la rugosidad superficial no debe reducir la resistencia al flujo respecto al escenario sin intervención. Por tanto, el escenario NbS debe mantener o aumentar la rugosidad frente a BaU, excepto en categorías bloqueadas o sin intervención.

---

## 10. Construcción del raster de infiltración para el escenario NbS

El raster `Infiltration_NbS.tif` también parte del escenario BaU. La infiltración final se inicializa como una copia de `Infiltration_BaU.tif`, de modo que los píxeles sin intervención mantienen la condición sin inversión.

El proceso es el siguiente:

1. Se leen la cobertura FastFlood, la cobertura WaterProof, el raster de portafolio, la infiltración base y la infiltración BaU.
2. Se inicializa:

```text
Infiltration_NbS = Infiltration_BaU
```

3. Se construye un raster temporal de infiltración. Para cada píxel se evalúa el par:

```text
(tipo de SbN, cobertura WaterProof)
```

4. Si el par existe en `infiltracion_dict`, el valor se interpreta como porcentaje de mejora y se aplica sobre la infiltración base:

```text
Infiltration_temp = Infiltration_Base · (1 + Mejora/100)
```

5. Se identifican píxeles con SbN, píxeles no bloqueados y píxeles con valor temporal válido.
6. La primera regla aplica la mejora solo si el valor temporal supera la infiltración base:

```text
Si Infiltration_temp > Infiltration_Base:
    Infiltration_NbS = Infiltration_temp
```

7. La segunda regla cubre los píxeles restantes donde hay SbN, el valor temporal es válido, pero no se activó la primera regla. Allí se intenta nuevamente la asignación usando la cobertura FastFlood mapeada a WaterProof.
8. El resultado se guarda como `Infiltration_NbS.tif`.
9. Finalmente, se aplica un control contra BaU:

```text
Infiltration_NbS ≥ Infiltration_BaU
```

La interpretación hidrológica es que una SbN no debe reducir la infiltración frente al escenario sin inversión. Si no hay una regla válida de mejora, o si el píxel pertenece a una categoría bloqueada, se conserva la infiltración BaU.

---

## 11. Preparación de capas de exposición para daños

Después de crear los parámetros hidráulicos de los escenarios, el adaptador prepara los rásteres de exposición usados para estimar daños económicos.

El proceso utiliza dos fuentes principales:

```text
01-GHS-BUILT-C
02-Road
```

La fuente GHS-BUILT-C se usa para representar exposición residencial y no residencial. Como esta base no separa de forma directa el uso comercial del industrial, el adaptador usa la proporción definida por el usuario en `SplitArea`. En el JSON de ejemplo, el área no residencial se distribuye como 50 % comercial y 50 % industrial.

La fuente de vías se usa para infraestructura vial. Además, el adaptador integra la cobertura agrícola sobre la base GHS-BUILT-C. Esta unión permite que los píxeles agrícolas de la cobertura actual sean incorporados al raster de exposición, especialmente donde la base construida no contiene información.

Las categorías y códigos usados son:

| Categoría           | Códigos                                    |
| ------------------- | ------------------------------------------ |
| Residential         | 11, 12, 13, 14, 15                         |
| Commercial          | 21, 22, 23, 24, 25                         |
| Industrial          | 21, 22, 23, 24, 25                         |
| InfraRoads          | 1                                          |
| Agriculture         | 40                                         |
| GHS_BUILT integrado | 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 40 |

El adaptador genera dos rásteres principales para el análisis de daños:

```text
GHS_BUILT_C_Agri.tif
Road.tif
```

El primero contiene la base construida combinada con agricultura. El segundo contiene la infraestructura vial.

---

## 12. Construcción de mosaicos de bases de daño

Las bases de daño están almacenadas como tiles. Para cada caso de estudio, el adaptador construye un mosaico recortado a la cuenca.

La lógica metodológica es:

1. Leer la geometría de la cuenca o área de interés.
2. Construir o reutilizar un índice tabular de bounding boxes de los tiles.
3. Reproyectar el área de interés al sistema de coordenadas de los tiles, si es necesario.
4. Filtrar únicamente los tiles que intersectan el área de interés.
5. Crear un mosaico virtual para no cargar todos los tiles en memoria.
6. Recortar el mosaico al límite exacto de la cuenca.
7. Guardar el raster final con compresión y estructura optimizada.

Esta lógica evita leer tiles que no intersectan la cuenca y reduce el uso de memoria al trabajar con mosaicos virtuales y recorte espacial. Es especialmente útil cuando las bases globales de exposición cubren áreas extensas y el análisis se limita a una cuenca específica.

---

## 13. Lectura de la tabla IDF

El adaptador lee el archivo:

```text
in/06-FLOOD/Raster/IDF.csv
```

Esta tabla contiene intensidades de precipitación por duración y periodo de retorno. El índice se convierte a valores numéricos para representar duraciones, y las columnas se convierten a valores numéricos para representar periodos de retorno.

Durante la ejecución de escenarios, la intensidad usada para un periodo de retorno específico se obtiene como:

```text
I = IDF.loc[DesignStormDuration_Historic, TR]
```

Cuando el escenario climático es histórico, esta intensidad puede pasarse directamente a FastFlood mediante el argumento de lluvia. Cuando el escenario es futuro, además de la intensidad base se incorporan los parámetros climáticos definidos por el usuario: escenario SSP, periodo, cuantil y duración de diseño climática.

---

## 14. Configuración hidráulica previa a la ejecución

Antes de ejecutar FastFlood se revisan dos elementos opcionales: el canal y la condición de borde oceánica.

Si el estado del canal es cero, no se envían parámetros de canal a FastFlood. Si está activo, se construye un vector con:

```text
WidthMul
WidthExp
DepthMul
DepthExp
CrossSection
ChannelManning
```

Estos parámetros permiten representar una aproximación 1D–2D de canales dentro de FastFlood.

La condición de borde oceánica se activa solo si el parámetro `BoundaryCondition` es distinto de cero. En ese caso, se envía a FastFlood mediante el argumento oceánico. Esta opción permite considerar interacción con el mar o con un cuerpo receptor con nivel de agua impuesto.

---

## 15. Ejecución de FastFlood para los escenarios Current, BaU y NbS

La función `RunScenarios` ejecuta FastFlood para tres escenarios:

```text
Current
BaU
NbS
```

y para todos los periodos de retorno definidos en el JSON.

Para cada periodo de retorno y escenario se generan cuatro productos:

```text
Flood_{Escenario}_TR-{TR}.tif
Velocity_{Escenario}_TR-{TR}.tif
Qpeak_{Escenario}_TR-{TR}.tif
TS_Q_{Escenario}_TR-{TR}.csv
```

El flujo por escenario es el siguiente.

### 15.1 Escenario Current

El escenario Current usa los rásteres originales de entrada:

```text
DEM.tif
Manning.tif
Infiltration.tif
```

Para cada periodo de retorno, FastFlood se ejecuta con la intensidad IDF correspondiente, la duración de tormenta de análisis y, si aplica, los parámetros de canal y condición oceánica.

### 15.2 Escenario BaU

El escenario BaU usa los rásteres generados previamente:

```text
Manning_BaU.tif
Infiltration_BaU.tif
```

El DEM se mantiene igual. La diferencia frente a Current está en la parametrización hidráulica e hidrológica derivada de la cobertura BaU.

### 15.3 Escenario NbS

El escenario NbS usa:

```text
Manning_NbS.tif
Infiltration_NbS.tif
```

Estos rásteres representan la condición con intervención del portafolio de SbN. El DEM se mantiene constante, por lo que el efecto de las SbN entra por cambios en rugosidad e infiltración, no por modificación topográfica.

En todos los escenarios se activa el factor de calibración global de FastFlood mediante `d_cal`. Esta opción permite aplicar multiplicadores globales precalibrados de Manning e infiltración, ajustados por escala de cuenca.

---

## 16. Postprocesamiento de salidas hidráulicas

FastFlood genera inicialmente salidas temporales para profundidad, velocidad y caudal pico. Después de cada ejecución, el adaptador recorta esas salidas al límite de la cuenca.

La secuencia por cada salida es:

1. Leer raster temporal.
2. Reproyectar la geometría de la cuenca al sistema de coordenadas del raster si es necesario.
3. Recortar por máscara espacial.
4. Guardar el raster final comprimido.
5. Para velocidad, aplicar un límite máximo de 15 m/s.

El filtro de velocidad actúa como control de valores físicamente improbables o numéricamente inestables. Cualquier valor de velocidad superior al umbral se reemplaza por 15 m/s. Este control no altera las profundidades ni los caudales pico, pero mejora la consistencia de los productos espaciales de velocidad.

---

## 17. Corrección de hidrogramas

Una vez ejecutados los escenarios, el adaptador revisa los hidrogramas generados por FastFlood. Esta revisión busca detectar inestabilidades numéricas que puedan afectar la coherencia temporal de las series de caudal.

La función de corrección recibe:

```text
folder = out/06-FLOOD/Discharge
dem_path = DEM.tif
d_ds = DesignStormDuration_Historic
D = AnalysisStormDuration
inf_raster = Infiltration.tif
idf_table = IDF.csv
inf_correction = FactorCorrect.csv
```

El proceso valida los hidrogramas por tripletas de escenario para cada periodo de retorno:

```text
BaU
Current
NbS
```

Un periodo de retorno se marca como inválido si ocurre alguna de las siguientes condiciones:

| Criterio                   | Regla                                                          |
| -------------------------- | -------------------------------------------------------------- |
| Orden temporal BaU–Current | `tp_BaU < tp_Current`                                          |
| Orden temporal BaU–NbS     | `tp_BaU < tp_NbS`                                              |
| Orden temporal NbS–Current | `tp_NbS < tp_Current`                                          |
| Unicidad de caudales pico  | Los tres `Qp` deben ser distintos                              |
| Ausencia de plateau        | El pico no debe permanecer plano más de dos pasos consecutivos |

Si un periodo de retorno falla, se reconstruyen los tres hidrogramas de ese periodo de retorno.

Cuando existen al menos dos periodos de retorno válidos y consecutivos, se ajustan regresiones entre caudal pico, tiempo al pico, duración y parámetro de forma del hidrograma. Los periodos inválidos se reconstruyen usando el Hidrograma Unitario Sintético SCS:

```text
q(t) = Qp · (t/tp)^(n-1) · exp[(n-1) · (1 - t/tp)]
```

Cuando no existe un bloque válido de al menos dos periodos de retorno consecutivos, la reconstrucción usa una aproximación sintética. En ese caso se calcula el tiempo de concentración con Kirpich, la velocidad de onda mediante una relación tipo Leopold-Maddock, la precipitación efectiva neta y un parámetro de forma fijo.

La precipitación efectiva se calcula como:

```text
pe = (I_TR - Inf_avg) · D
```

Si `pe ≤ 0`, el hidrograma se deja vacío porque, bajo esa condición, la infiltración media iguala o supera la intensidad de lluvia considerada.

El proceso de corrección genera una carpeta de diagnósticos con validaciones, parámetros calibrados, regresiones, reconstrucciones, geometría de cuenca, precipitación efectiva y gráficas de hidrogramas.

---

## 18. Control de coherencia entre profundidades Current, BaU y NbS

Después de corregir hidrogramas, el adaptador revisa los rásteres de profundidad de los tres escenarios para cada periodo de retorno.

El objetivo es evitar combinaciones físicamente incoherentes entre escenarios. La expectativa general es:

```text
BaU no debe tener menor profundidad que Current cuando representa degradación.
NbS no debe tener mayor profundidad que BaU cuando representa intervención.
```

La función de control aplica primero un filtro de valores atípicos. Para cada escenario calcula el percentil 99 de profundidades positivas y reemplaza los valores superiores por ese percentil. Esto reduce el efecto de celdas extremas aisladas.

Luego aplica reglas de ajuste píxel a píxel:

| Condición detectada   | Corrección aplicada               |
| --------------------- | --------------------------------- |
| `Current > BaU > NbS` | `Current = BaU`                   |
| `Current > NbS > BaU` | `BaU = Current` y `NbS = Current` |
| `NbS > Current > BaU` | `BaU = Current` y `NbS = Current` |
| `NbS > BaU > Current` | `NbS = Current`                   |

Estas reglas buscan mantener consistencia relativa entre los escenarios, evitando que el escenario sin inversión aparezca artificialmente mejor que la condición actual o que el escenario con SbN aparezca artificialmente peor que BaU.

---

## 19. Lectura de curvas de daño

El adaptador lee dos archivos principales desde la carpeta de daños:

```text
01-Damage_Factor_Curves.csv
02-Maximum_Damage_Cost.csv
```

El primer archivo contiene curvas de factor de daño por profundidad. El segundo contiene los costos máximos de daño para cada categoría.

La curva monetaria de daño se construye multiplicando el factor de daño por el costo máximo:

```text
Curva_Daño_Monetaria(h) = Factor_Daño(h) · Costo_Máximo
```

Luego se aplica la tasa de cambio definida por el usuario:

```text
Curva_Daño_Monetaria = Curva_Daño_Monetaria · DamagesExchangeRate
```

Las unidades de las funciones de daño son monetarias por metro cuadrado para las cinco categorías:

```text
Residential  ($/m²)
Commercial   ($/m²)
Industrial   ($/m²)
InfraRoads   ($/m²)
Agriculture  ($/m²)
```

---

## 20. Factores de área por categoría

Para convertir daño unitario a daño por píxel, el adaptador multiplica por un área efectiva según la categoría:

| Categoría   |                 Área efectiva |
| ----------- | ----------------------------: |
| Residential |                        100 m² |
| Commercial  | 100 · SplitArea_Commercial m² |
| Industrial  | 100 · SplitArea_Industrial m² |
| InfraRoads  |                         60 m² |
| Agriculture |                        100 m² |

La categoría de vías usa 60 m² porque se asume una vía de 6 m de ancho sobre un píxel de 10 m de longitud. Las categorías comercial e industrial comparten los mismos códigos no residenciales, por lo que se separan usando la proporción definida por el usuario.

---

## 21. Procesamiento espacial de daños por chunks

La estimación de daños puede involucrar millones de píxeles y múltiples periodos de retorno. Para evitar cargar toda la información en memoria, el adaptador usa un esquema de procesamiento por chunks.

La lógica del generador de chunks es:

1. Abrir el raster de exposición o cobertura de daño.
2. Definir una ventana de procesamiento, por defecto de 2048 × 2048 píxeles.
3. Leer únicamente el bloque espacial correspondiente.
4. Aplicar la máscara de la cuenca sobre ese bloque.
5. Filtrar los píxeles cuyo código pertenece a las categorías de daño requeridas.
6. Para cada raster de profundidad asociado a un periodo de retorno, leer solo la ventana correspondiente.
7. Alinear virtualmente cada raster de profundidad a la ventana del raster de exposición.
8. Extraer las profundidades solo en los píxeles válidos.
9. Aplicar un umbral mínimo de profundidad de 0.01 m. Las profundidades menores se consideran cero.
10. Eliminar los píxeles cuya suma de profundidades en todos los periodos de retorno es cero.
11. Entregar el chunk como una tabla con una columna de código de exposición y una columna por periodo de retorno.

La estructura conceptual de cada chunk es:

```text
Code | TR_1000 | TR_500 | TR_200 | ... | TR_2
```

Este diseño permite procesar daños por partes, acumular resultados y liberar memoria después de cada bloque. Cuando `SaveFullCSV=False`, no se guardan todas las tablas intermedias y solo se acumulan los resultados agregados. Cuando `SaveFullCSV=True`, el adaptador guarda también profundidades por categoría, daños por periodo de retorno y EAD por píxel.

---

## 22. Estimación de daño por categoría y escenario

Para cada escenario (`Current`, `BaU`, `NbS`) el adaptador procesa dos fuentes:

```text
GHS_BUILT → Residential, Commercial, Industrial, Agriculture
InfraRoads → InfraRoads
```

Para cada chunk y cada categoría:

1. Se filtran los píxeles cuyo código corresponde a la categoría.
2. Se eliminan los códigos de cobertura para conservar solo profundidades.
3. Se interpola la curva de daño usando las profundidades simuladas.
4. Se limita el daño máximo al máximo de la curva.
5. Se multiplica por el área efectiva de la categoría.
6. Se obtiene una matriz de daños por píxel y periodo de retorno.

La interpolación permite convertir una profundidad simulada cualquiera en daño monetario, aunque la profundidad no coincida exactamente con los puntos tabulados de la curva.

La forma general es:

```text
Daño_píxel,TR,categoría = f_categoría(h_TR) · Área_categoría
```

donde `f_categoría(h_TR)` es el valor monetario interpolado de la curva de daño para la profundidad `h_TR`.

---

## 23. Cálculo del daño anual esperado

Después de estimar daños para todos los periodos de retorno, el adaptador calcula el daño anual esperado por píxel. Para ello convierte cada periodo de retorno en probabilidad anual de excedencia:

```text
P = 1 / TR
```

Luego integra la curva daño-probabilidad con la regla trapezoidal:

```text
EAD = ∫ Daño(P) dP
```

En el script, esta integración se aplica por fila de la matriz de daño. El resultado es un valor de daño anual esperado por píxel. Luego el adaptador suma todos los píxeles del chunk y acumula el valor en un diccionario por categoría.

El proceso se repite para todos los chunks. Al final de cada escenario se obtiene una tabla agregada:

```text
03-Expected_Annual_Damage_Total.csv
```

con esta estructura:

```text
Scenarios | Residential | Commercial | Industrial | InfraRoads | Agriculture
Current
BaU
NbS
```

Este archivo es el resultado económico central del componente de inundación antes de la desagregación temporal.

---

## 24. Generación de insumos de inundación para indicadores

Antes de estimar daños por categoría, el adaptador calcula indicadores hidráulicos globales para cada escenario.

Para cada escenario se calcula, por periodo de retorno:

| Indicador base     | Método                                                                      |
| ------------------ | --------------------------------------------------------------------------- |
| Profundidad máxima | Máximo de profundidad en la cuenca después de aplicar umbral mínimo.        |
| Área inundada      | Número de píxeles con profundidad válida multiplicado por el área de píxel. |
| Caudal pico        | Máximo del hidrograma corregido `TS_Q_{Escenario}_TR-{TR}.csv`.             |

Estos resultados se guardan como:

```text
INDICATORS/in/INPUTS_FLOOD_Current.csv
INDICATORS/in/INPUTS_FLOOD_BaU.csv
INDICATORS/in/INPUTS_FLOOD_NbS.csv
```

Estos archivos conectan el componente hidráulico con el cálculo final de indicadores.

---

## 25. Desagregación temporal de daños

El cálculo de EAD produce valores discretos por escenario. Sin embargo, el módulo ROI necesita series anuales durante el horizonte de análisis. Para esto se usa la función `DesaggregationData`.

Primero se guarda la tabla agregada de daños como entrada del módulo de desagregación:

```text
in/05-DISAGGREGATION/01-INPUTS_Flood.csv
```

Luego se leen tres insumos:

```text
out/05-DISAGGREGATION/02-OUTPUTS_BaU.csv
out/05-DISAGGREGATION/02-OUTPUTS_NBS.csv
in/05-DISAGGREGATION/01-INPUTS_NBS.csv
in/05-DISAGGREGATION/01-INPUTS_Time.csv
```

La desagregación usa una función logística:

```text
W(t) = Wmax / [1 + ((Wmax / W0) - 1) · exp(-r · t)]
```

Para BaU, la función mueve cada categoría desde el valor Current hacia el valor BaU durante el horizonte de análisis. Para NbS, el proceso incorpora el ritmo de implementación del portafolio.

En el escenario NbS se calculan tres elementos:

```text
NBS_Total = suma del área total implementada por SbN
t_NBS = tiempo ponderado de maduración o implementación
p_NBS = porcentaje ponderado de efecto inicial
Factor = proporción acumulada del portafolio implementado por año
```

Luego el beneficio de NbS se aplica de manera gradual:

```text
Daño_NbS(t) = Daño_BaU(t) - [Daño_BaU(t) - Daño_NbS_potencial(t)] · Factor(t)
```

Esta lógica representa que el beneficio hidráulico-económico de las SbN no aparece completamente desde el primer año, sino que depende del cronograma de implementación y consolidación del portafolio.

---

## 26. Integración con carbono

Después de desagregar daños, el adaptador incorpora las series de carbono BaU y NbS. Para esto agrega la columna:

```text
WC (Ton)
```

a las tablas desagregadas de daños.

Los resultados se guardan en cuatro ubicaciones:

```text
out/06-FLOOD/Damages/04-Damage_Dissagregation_BaU.csv
out/06-FLOOD/Damages/04-Damage_Dissagregation_NbS.csv
out/05-DISAGGREGATION/02-OUTPUTS_BaU_Flood.csv
out/05-DISAGGREGATION/02-OUTPUTS_NbS_Flood.csv
```

También se exportan como entradas del módulo ROI:

```text
ROI/in/7-Damages_BaU.csv
ROI/in/8-Damages_NbS.csv
```

Con esto, el ROI puede combinar beneficios por reducción de daños de inundación y beneficios por carbono.

---

## 27. Cálculo de ROI

La función `ROI_FastFlood` lee los archivos de entrada del módulo financiero:

```text
1-NBS_Cost.csv
2-Porfolio_NBS.csv
3-Financial_Parmeters.csv
4-Time.csv
5-CO2_BaU.csv
6-CO2_NBS.csv
7-Damages_BaU.csv
8-Damages_NbS.csv
```

El beneficio por daños evitados se calcula como:

```text
Benefit_Damage = Damage_BaU - Damage_NbS
```

Si el valor es negativo, se reemplaza por cero:

```text
Si Benefit_Damage < 0:
    Benefit_Damage = 0
```

Esto evita contabilizar como pérdida un resultado donde el escenario NbS no reduce el daño para una categoría o año específico.

El cálculo de costos incluye:

| Costo          | Lógica                                                                       |
| -------------- | ---------------------------------------------------------------------------- |
| Implementación | Área anual del portafolio multiplicada por costo unitario de implementación. |
| Mantenimiento  | Costos recurrentes aplicados según frecuencia de mantenimiento.              |
| Oportunidad    | Costo acumulado asociado al área implementada.                               |
| Transacción    | Porcentaje aplicado sobre implementación, mantenimiento y oportunidad.       |
| Plataforma     | Costos fijos o recurrentes definidos en parámetros financieros.              |

El beneficio por carbono se calcula a partir de la diferencia anual de almacenamiento entre escenario NbS y BaU. La diferencia de carbono se transforma a CO₂ usando el factor molecular:

```text
CO₂ = C · 44 / 12
```

Luego se monetiza con el valor definido en los parámetros financieros.

El ROI se calcula en cuatro versiones:

```text
ROI_Total = Beneficios acumulados sin descuento / Costos acumulados sin descuento

ROI_TD_Min = Beneficios descontados con tasa mínima / Costos descontados con tasa mínima

ROI_TD_Mean = Beneficios descontados con tasa media / Costos descontados con tasa media

ROI_TD_Max = Beneficios descontados con tasa máxima / Costos descontados con tasa máxima
```

También se calcula el valor presente neto por componente:

```text
Implementation
Maintenance
Opportunity
Transaction
Platform
Benefit
Total
```

Los principales archivos de salida del ROI son:

```text
8_GlobalTotals.csv
9_Benefit_Sensitivity.csv
10_Cost_Sensitivity.csv
11_ROI_Sensitivity.csv
12_NPV.csv
```

---

## 28. Cálculo de indicadores finales

Al final del flujo se ejecuta `Indicators_BaU_NBS`.

La función calcula primero indicadores de carbono comparando las series NbS y BaU:

```text
Indicador_Carbono(t) = [(Carbon_NbS(t) - Carbon_BaU(t)) / Carbon_BaU(t)] · 100
```

También calcula un indicador integrado usando el área bajo la curva temporal de carbono:

```text
Indicador_Carbono_Total = [(∫ Carbon_NbS dt - ∫ Carbon_BaU dt) / ∫ Carbon_BaU dt] · 100
```

Luego calcula indicadores de inundación comparando NbS contra BaU:

```text
Peak Discharge TR 10 = [(Qp_NbS_TR10 - Qp_BaU_TR10) / Qp_BaU_TR10] · 100

Peak Discharge TR 100 = [(Qp_NbS_TR100 - Qp_BaU_TR100) / Qp_BaU_TR100] · 100

Flooded Area = [(Área_NbS - Área_BaU) / Área_BaU] · 100
```

El archivo final integra los indicadores de caudal pico, área inundada y carbono:

```text
OUTPUTS-Indicators.csv
```

La interpretación esperada es que valores negativos en caudal pico o área inundada representan reducción frente a BaU, mientras que valores positivos en carbono representan ganancia de almacenamiento o beneficio respecto a BaU.

---

## 29. Salidas principales del flujo completo

| Componente     | Archivo o carpeta                     | Contenido                                                     |
| -------------- | ------------------------------------- | ------------------------------------------------------------- |
| Parámetros BaU | `Manning_BaU.tif`                     | Rugosidad superficial del escenario sin inversión.            |
| Parámetros BaU | `Infiltration_BaU.tif`                | Infiltración del escenario sin inversión.                     |
| Parámetros NbS | `Manning_NbS.tif`                     | Rugosidad superficial con portafolio de SbN.                  |
| Parámetros NbS | `Infiltration_NbS.tif`                | Infiltración con portafolio de SbN.                           |
| Profundidad    | `Flood_{Escenario}_TR-{TR}.tif`       | Profundidad de inundación por escenario y periodo de retorno. |
| Velocidad      | `Velocity_{Escenario}_TR-{TR}.tif`    | Velocidad máxima por escenario y periodo de retorno.          |
| Caudal pico    | `Qpeak_{Escenario}_TR-{TR}.tif`       | Caudal pico espacializado.                                    |
| Hidrogramas    | `TS_Q_{Escenario}_TR-{TR}.csv`        | Serie temporal de caudal corregida cuando aplica.             |
| Daños          | `03-Expected_Annual_Damage_Total.csv` | Daño anual esperado por escenario y categoría.                |
| Desagregación  | `04-Damage_Dissagregation_BaU.csv`    | Serie anual de daños BaU.                                     |
| Desagregación  | `04-Damage_Dissagregation_NbS.csv`    | Serie anual de daños NbS.                                     |
| ROI            | `11_ROI_Sensitivity.csv`              | ROI sin descuento y con tasas de descuento.                   |
| ROI            | `12_NPV.csv`                          | Valor presente neto por componente.                           |
| Indicadores    | `OUTPUTS-Indicators.csv`              | Indicadores finales de inundación y carbono.                  |

---

## 30. Supuestos metodológicos principales

El adaptador representa las SbN mediante cambios en rugosidad superficial e infiltración. Esta representación es consistente con el enfoque operativo de FastFlood, pero no modela explícitamente procesos ecológicos transitorios, crecimiento de vegetación, cambios progresivos de suelo o dinámica geomorfológica.

El escenario BaU se trata como una condición sin inversión. Por tanto, sus reglas evitan que la rugosidad o la infiltración mejoren respecto al escenario actual. El escenario NbS se trata como una condición con intervención, por lo que sus reglas fuerzan que la rugosidad y la infiltración sean iguales o mejores que BaU en los píxeles intervenidos.

El análisis de daños depende de la calidad de las curvas de daño, los costos máximos, la distribución espacial de exposición y la asignación de áreas efectivas por categoría. La separación comercial-industrial se basa en proporciones definidas por el usuario porque la base espacial usada identifica áreas no residenciales sin desagregarlas directamente.

El daño anual esperado se estima por integración de la curva daño-probabilidad usando los periodos de retorno configurados. Por tanto, el resultado representa una métrica anualizada de riesgo económico esperado, no el daño de un evento específico.

La corrección de hidrogramas actúa como control de consistencia temporal. No reemplaza la simulación hidráulica de profundidades, pero sí asegura que los hidrogramas usados para caudal pico, indicadores y visualización tengan una forma físicamente razonable.

---

## 31. Referencias

Fast Hazard B.V. (2025). *Integration of FastFlood with WaterProof: Documentation Report*. Documento técnico del proyecto WaterProof–FastFlood.

Kirpich, Z.P. (1940). Time of concentration of small agricultural watersheds. *Civil Engineering*, 10(6), p. 362.

Leopold, L.B. y Maddock, T. (1953). *The Hydraulic Geometry of Stream Channels and Some Physiographic Implications*. USGS Professional Paper 252. Disponible en: https://doi.org/10.3133/pp252

SCS. (1972). *National Engineering Handbook, Section 4: Hydrology*. USDA Soil Conservation Service.

Williams, J.R. y Hann, R.W. (1973). *HYMO: Problem-Oriented Computer Language for Hydrologic Modeling*. USDA Agricultural Research Service.
