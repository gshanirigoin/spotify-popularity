# Predicción de la popularidad de canciones de Spotify

Modelo de regresión que predice la `popularity` (0–100) de un track de Spotify a partir de sus
características de audio, sus artistas, el nombre del track y del álbum y sus géneros.
La métrica de evaluación es el **RMSE** (competición de Kaggle del curso de Computational Machine Learning).

## Estructura

```
.
├── spotify_pipeline.ipynb       # Notebook principal: modelado, evaluación y submissions
├── spotify_gshan_irigoin.ipynb  # Análisis exploratorio de cada variable
├── src/spotify/
│   ├── features.py              # Feature engineering (transformadores compatibles con sklearn)
│   └── tabpfn_runner.py         # TabPFN en un proceso aparte, con caché en disco
├── experiments/
│   └── tabpfn_quick.py          # Prueba rápida de TabPFN frente a XGBoost
├── outputs/                     # Submissions (submission_<modelo>_<n>.csv) y su registro
├── data/                        # Datos de Kaggle (no incluidos en el repositorio)
├── pyproject.toml
└── uv.lock
```

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
uv sync
```

En macOS, LightGBM y XGBoost necesitan OpenMP: `brew install libomp`.

### Datos

Descargar de la competición de Kaggle `train.csv`, `test.csv` y `sample_submission.csv` y copiarlos en `data/`.

### TabPFN (opcional, sección 7 del notebook)

TabPFN descarga sus pesos preentrenados, para lo que hace falta aceptar su licencia una vez:

1. Registrarse en https://ux.priorlabs.ai y aceptar la licencia (sección *Licenses*).
2. Guardar la API key en la variable de entorno `TABPFN_TOKEN` o en el archivo `~/.cache/tabpfn/auth_token`.

Los datos no salen del ordenador: TabPFN se ejecuta en local (usa la GPU MPS en Mac si está disponible).

## Uso

Abrir `spotify_pipeline.ipynb` con el kernel de `.venv` y ejecutar todas las celdas. El notebook:

1. Divide `train` en **train_train** (80 %) y **train_test** (20 %).
2. Ajusta los hiperparámetros con validación cruzada de 5 particiones sobre train_train:
   `RandomizedSearchCV` (LinearRegression, DecisionTree, LightGBM, CatBoost) y **Optuna** (RandomForest, XGBoost).
3. Construye un **ensamble ponderado** de los 3 mejores modelos de árboles (pesos elegidos con predicciones fuera de muestra).
4. Evalúa todos los modelos en train_test.
5. Reentrena el mejor modelo por CV con todo `train` y guarda una submission por modelo en `outputs/`.
6. (Opcional) Evalúa TabPFN y su mezcla con el ensamble, y guarda sus submissions.

Al final imprime qué archivo de `outputs/` subir a Kaggle. Cada submission queda anotada en
`outputs/submissions_log.csv` con su RMSE de CV y de train_test.

Tiempo en un Mac con Apple Silicon (14 núcleos): **~2 h** las secciones 1–6 con la caché vacía (de ellas, 75 min
son el límite de tiempo de Optuna) y ~1 h la sección 7 (TabPFN, estimado).

## Resultados (secciones 1–6)

RMSE en la validación cruzada de 5 particiones sobre train_train y en train_test (20 % reservado):

| Modelo | CV RMSE | RMSE train_test | R² train_test |
|---|---|---|---|
| **Ensamble** (XGBoost 0,52 · LightGBM 0,25 · CatBoost 0,24) | **8,099** | **7,919** | 0,840 |
| XGBoost (Optuna, 38 pruebas) | 8,122 | 7,943 | 0,839 |
| LightGBM | 8,195 | 8,015 | 0,836 |
| CatBoost | 8,199 | 7,994 | 0,837 |
| RandomForest (Optuna, 30 pruebas) | 8,307 | 8,136 | 0,831 |
| DecisionTree | 9,417 | 9,084 | 0,789 |
| LinearRegression | 11,512 | 11,293 | 0,674 |

El modelo final (menor RMSE de CV) es el ensamble ponderado, reentrenado con todo `train`.

## Resultados (sección 7, TabPFN)

TabPFN con 40 000 filas de contexto (GPU MPS: ~21 min en train_test y ~28 min en test). Sin RMSE de CV (cada
ejecución tarda más de 20 minutos), así que se evalúa solo en train_test:

| Modelo | RMSE train_test | R² train_test |
|---|---|---|
| **Ensamble + TabPFN** (50 % / 50 %) | **7,799** | 0,845 |
| TabPFN | 7,823 | 0,844 |
| Ensamble (sección 6) | 7,919 | 0,840 |

La mezcla es la submission recomendada (`outputs/submission_ens_tabpfn_<n>.csv`). Nota: el peso 0,5 se eligió
en la fase de exploración mirando train_test, así que su RMSE en train_test es algo optimista.

### Caché del preprocesamiento

Construir las features tarda unos 17 s por partición de la CV, y la búsqueda de hiperparámetros ajusta unos 600
pipelines. Pero las features de una partición dependen solo de sus filas, no de los hiperparámetros del
modelo, así que sin caché se repetiría el mismo cálculo cientos de veces (2–3 horas).

Los pipelines usan `Pipeline(memory=...)` de scikit-learn: la primera vez que se preprocesa una partición, el
resultado se guarda en disco; los ajustes siguientes con las mismas filas lo cargan y solo entrenan el modelo
(una CV de 5 particiones pasa de ~83 s a ~13 s). Los resultados son **idénticos** con y sin caché.

**¿Qué `random_state` importa para la caché?** La huella de cada entrada de la caché se calcula solo con lo
que entra en los pasos `rows` y `prep` (los datos y los parámetros del preprocesador). Por eso los tres
`random_state` del notebook tienen papeles distintos:

| `random_state` de… | Qué controla | ¿Entra en la huella? | Si fuera distinto en cada modelo… |
|---|---|---|---|
| El modelo (`LGBMRegressor(random_state=…)`) | Aleatoriedad del modelo (submuestreo de filas/columnas…) | No: el paso `model` nunca se guarda en caché | La caché funciona igual |
| La CV (`KFold(…, random_state=…)`) | Qué filas caen en cada partición | Sí, indirectamente: cambia los datos de entrada | Cada búsqueda reutiliza solo sus propias features |
| El preprocesador (`make_tree_preprocessor(RANDOM_STATE)`) | Particiones internas del target encoding y del modelo de texto | Sí: es un parámetro del paso guardado | Cada búsqueda reutiliza solo sus propias features |

- La caché **no** exige que todos los modelos usen la misma semilla: lo que exige es que el preprocesamiento
  sea **determinista** (mismas filas + mismos parámetros → mismas features). Por eso el preprocesador lleva un
  `random_state` fijo: con `random_state=None` cada ajuste haría particiones internas distintas y la caché
  "congelaría" el primer sorteo, con resultados distintos de los de una ejecución sin caché.
- Si cada modelo tuviera sus propias particiones de CV, la caché seguiría ahorrando la mayor parte del tiempo,
  porque la repetición ocurre sobre todo **dentro** de cada búsqueda (mismo modelo, muchas combinaciones,
  mismas particiones); solo dejaría de compartir features entre modelos distintos.
- En este proyecto todos los modelos usan las **mismas particiones** de CV y el mismo preprocesador, así que la
  caché se reutiliza en todas las búsquedas. Usar las mismas particiones para todos es además buena práctica:
  hace comparables los RMSE de CV, porque las diferencias entre modelos no se deben a la suerte de la partición.

- **Ubicación**: `~/.cache/spotify-pipeline/features_<huella>/`, fuera del proyecto (ocupa ~1,5 GB y no debe
  sincronizarse con iCloud ni subirse a git).
- **Invalidación**: `<huella>` depende del contenido de `src/spotify/features.py` y de la versión de scikit-learn.
  Si se modifica el código de las features, se usa una carpeta nueva y todo se recalcula.
- **Limpieza**: se puede borrar la carpeta en cualquier momento (`rm -rf ~/.cache/spotify-pipeline`); solo hace
  que la siguiente ejecución vuelva a calcular las features. Las carpetas de versiones antiguas no se borran solas.

## Feature engineering

Todo el preprocesamiento está dentro de un `Pipeline` de scikit-learn, de modo que se ajusta solo con los
datos de entrenamiento de cada partición:

- **Features fila a fila**: número de artistas, patrones en el nombre del track y del álbum (feat., remix, live,
  remaster, deluxe, recopilatorio...), indicador de single, transformaciones del audio.
- **Features de grupo** (calculadas sobre train + test, sin usar la popularidad): tamaño del álbum, número de
  álbumes y tracks por artista, perfil de audio medio del álbum y desviación del track respecto a su artista.
- **Target encoding con cross-fitting** (sin fuga del target): artistas, track, álbum, combinación de géneros,
  artista × género, estadísticas de popularidad por artista y género.
- **Modelo de texto**: TF-IDF de artistas, track y álbum con una Ridge cuya predicción fuera de muestra se usa como feature.
- **Géneros**: codificación multi-hot de la lista de géneros.

Los modelos lineales usan un preprocesador propio (escalado, términos cuadráticos, deciles y one-hot sin colinealidad).
