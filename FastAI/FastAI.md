# FastAI

Última actualización, 01/07/2022.

## Bibliografía e Información

La mayor parte de la información se ha obtenido del libro "Deep Learning for Coders with Fast-AI & PyTorch". La distribución de los capítulos en este documento no corresponden con el orden presentado en el libro, aquí se ha decidido dividirlo en tipos de problemas, ejemplos y sus soluciones.

# Índice

[TOC]

# Anexo A: Terminología útil

Aquí se reúnen las palabras, términos, conceptos etc. claves a tener en cuenta a modo de recordatorio. No tiene ningún tipo de orden.

+ La letra 'x' se asocia a la variable independiente, lo que usamos para hacer predicciones, por ejemplo imágenes. Mientras que la letra 'y' se asocia a la variable dependiente, lo que se denominan etiquetas y es nuestro objetivo obtener una predicción que tenga una alta probabilidad de parecerse a dicha 'y', un ejemplo de etiquetas pueden ser los nombres de las imágenes que permiten clasificar razas de perros.
+ weigth = pesos: valores aleatorios con los que se inicializan a las neuronas, estos parámetros son fundamentales para determinar el tipo de funcionamiento de una red.

+ + w(tamaño input, número de neuronas)
+ bias = sesgo: 
  + b(1, número de neuronas)
+ Un set de datos de puede dividir en:
  + Set de entrenamiento (train set).
  + Set de desarrollo o validación (dev set o validation set).
  + Set de pruebas (test set)
+ Función de pérdida (Loss function): función que mide el rendimiento del modelo en uno de los ejemplos del set de entrenamiento.
+ Función de coste (Cost function): función que mide el rendimiento del modelo en todos los ejemplos del set de entrenamiento. Sería la media de cada una de las funciones de pérdidas de cada ejemplo del set de entrenamiento.

# Capítulo 0: Útiles

## 0.1. Montar unidad de Google Drive en Google Collab

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

## 0.2. Dudas con métodos/funciones en FastAI, documentación

```python
# Ejemplo
doc(learn.predict)
```

## 0.3. Obtención de imágenes empleando la API de Bing

En el que caso de tener que trabajar con redes en las que los datos con los que trabajamos son imágenes, podemos emplear la API de Bing usando Microsoft Azure, esta API permite descargar de manera gratuita hasta 1000 colas por mes donde cada cola permite hasta 150 imágenes. Esto puede ser escaso por lo que requerirá procesos de aumentación de datos. Cuando tengamos una cuenta hecha en Azure, obtenemos la clave (key) en la ruta:

API Bing Search - Key Setting - Key 1

Usaríamos el API de la siguiente manera:

```python
# Sustituir XXX por la clave KEY 1
key = os.environ.get('AZURE_SEARCH_KEY', 'XXX')

# En este ejemplo se pretendía obtener 3 carpetas con diferentes razas de osos para 
# clasificarlos posteriormente. Ejemplo mejor desarrollado en el Capítulo 1.

# Si el directorio no existe, lo creamos
if not path.exists(): 
    path.mkdir()
    
    # Para cada oso perteneciente a la lista de razas de osos, seleccionamos el destino cuyo nombre
    # de carpeta es igual a la raza, creamos dicho directorio, buscamos los resultados con la API de BING
    # y finalmente descargamos las imágenes
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok = True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))
```

### 0.3.1. Borrar carpetas de datos en caso de fallos

```python
!rm_rf nombre_carpeta
```



# Capítulo 1: Visión computacional

## 1.1. Reconocimiento de perros y gatos

### 1.1.1. Anotaciones

+ Vamos a utilizar el dataset de Oxford_IIT PET Dataset con imágenes perros y gatos (37 razas).  **Para este dataset los nombres de los gatos vienen con la primera letra en mayúsculas.**

+ Usaremos modelos ya entrenados con el fin de realizar fine-tuning. 

  Recordar que un modelo pre-entrenado es un modelo cuyos pesos ya han sido adaptados y ajustados para una tarea. Con fine-tuning o ajuste fino conseguimos eliminar la última capa junto a sus pesos y la ajustamos a la nueva tarea (tener en cuenta que han de ser tareas similares entrenadas anteriormente y ahora con datos del mismo tipo, por ejemplo si el modelo fue entrenado con imágenes el dataset nuevo a emplear han de ser también basado en imágenes)

+ Podríamos emplear en transformaciones en un conjunto reducido de imágenes llamado **"batch"** y que permite ser alojados en la memoria RAM de la GPU, esto permite hacer el proceso de una manera más rápida. **Pero** hay que tener cuidado de no utilizar un **"batch_size"** (un tamaño de lote) muy grande ya que la GPU se podría quedar sin memoria RAM y a la hora de entrenar el modelo daría problemas tales como "CUDA out of memory error", si esto ocurre, tenemos que  reducir el tamaño del batch y reiniciar el kernel de Jupyter en caso de usarlo.

+ **Es importante saber que una clasificación pretende predecir una clase o categoría mientras que un modelo de regresión intenta predecir 1 0 más cantidades numéricas**.

+ **Metrics son distintas a la función de pérdida**

### 1.1.2. Código

```python
# Instalamos dependencias de los cuadernos de FastAI.
!pip install -Uqq fastbook

# Importamos la librería.
import fastbook

# Preparamos el cuaderno.
fastbook.setup_book()

# Como se trata de un problema de visión, usaremos redes neuronales convolucionales (CNN),
# ademas metemos la librería fastbook que incluye funciones para el tratamiento de imágenes
# y también las funciones necesarias para implementar los widgets.
from fastai.vision.all import *
from fastbook import *

# Etiquetado de las imágenes mediante una función "es_gato", donde cada imagen de un gato
# su primera letra es mayúscula.
def es_gato(x):
    return x[0].isupper()

path = untar_data(URLs.PETS)/'images'

# En este caso vamos a usar imágenes para un modelo convolucional por eso empleamos ImageDataLoaders,
# from_name_func permite especificar que vamos a otbener las etiquetas a partir de una función.
# path -> Seleccionamos el directorio en el que se encuentran las imágenes.
# get_image_files -> es la manera en la que vamos a obtener los datos.
# valid_pct -> es el porcentaje de los datos que vamos a usar para validación, el resto 
# se empleará como set de entrenamiento.
# seed -> establece una semilla con el fin de obtener un número aleatorio a partir de ella.
# label_func -> vamos a obtener las etiquetas a partir de una función, en este caso, "es_gato".
# item_tfms -> cada imagen se reajusta a a una resolución de 224 pixeles.
dls = ImageDataLoaders.from_name_func(path, get_image_files(path), 
                                      valid_pct = 0.2, seed = 42, 
                                      label_func = es_gato, 
                                      item_tfms = Resize(224))

# Utilizaremos redes convolucionales (CNN), en concreto, una red residual de 34 capas,
# metrics indica los parámetros que queremos ir mostrando durante el entrenamiento.
# Recordar que accuracy = 1 - error_rate, son probabilidades por lo que son valores
# comprendidos entre 0 y 1.
learn = vision_learner(dls, resnet34, metrics = [accuracy, error_rate])

# Vamos a ajustar el modelo durante 3 pasadas (número de épocas) completas a los datos.
learn.fine_tune(3)

# Colocamos un botón para subir una imagen.
boton_subida = widgets.FileUpload()
boton_subida

# Procesamos el dato almacenado en "boton_subida".
img = PILImage.create(boton_subida.data[0])
es_gato,_,probs = learn.predict(img)

print(f"¿Es un gato? {es_gato}")
print(f"Probabilidad de que sea un gato: {probs[1].item():.6f}")
```

## 1.2. Segmentación de imágenes, UNET

### 1.2.1. Anotaciones

+ Utilizamos una versión reducida del dataset de CAMVID preparada por FastAI.
+ **UNET** **se trata de una arquitectura empleada en segmentación que permite colorear cada elemento correspondiente a la clase a la que pertenece. UNET emplea la convolución transpuesta ya que esta permite aumentar el tamaño del volumen conforme avanzamos en la red (utilizando lo que se conocen como skip connections), con ello, a la salida obtenemos un resultado ya segmentado con la misma dimensión que la imagen introducida como input.**

### 1.2.2. Teoría redes convolucionales y detallado de UNET

![0001](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0001.jpg)

![0002](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0002.jpg)

![0003](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0003.jpg)

![0004](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0004.jpg)

![0005](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0005.jpg)

![0006](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0006.jpg)

![0007](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0007.jpg)

![0008](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0008.jpg)

![0009](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0009.jpg)

![0010](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0010.jpg)

![0011](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0011.jpg)

![0012](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0012.jpg)

![0013](D:\Documentos\Propios\AI_LIB\FastAI\Imagenes\0013.jpg)

### 1.2.3. Código

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastai.vision.all import *

path = untar_data(URLs.CAMVID_TINY)/'images'

# SegmentationDataLoaders permite especificar que vamos a tratar con un modelo de segmentación,
# aquí los datos se etiquetarán a partir de una función lambda (función anónima) de Python
# bs -> es el batchsize (el tamaño de los subconjuntos de imágenes que vamos a ir arrojando a la GPU)
# el tamaño dependerá de la cantidad de RAM que tenga la GPU.
# np.loadtext(...) -> es una función de NumPy que te permite cargar datos de un txt
dls = SegmentationDataLoaders.from_label_func(path, bs = 8, 
											  fnames = get_image_files(path),
											  label_func = lambda o:path/'labels'/f'{o.stream}_P{o.suffix}',
											  codes = np.loadtext(path/'codes.txt', dtype = str))
								
learn = unet_learner(dls, resnet34)
learn.fine_tune(8)

# Mostramos en 2 columnas, donde en la primera se encuentran las imágenes de validación (y),
# y en la segunda se encuentran las imágenes de predicción (^y). Mostramos 6 imágenes con un
# tamaño de 7 x 8
learn.show_results(max_n = 6, figsize = (7, 8))
```

## 1.3. Clasificación de osos

### 1.3.1. Anotaciones

+ En este ejercicio no se parte de ningún dataset, **vamos a crear el nuestro propio dataset** con el fin de ver que con **FastAI podemos crear modelos que se adapten a nuestros datos**. Para ello **usaremos los DataLoaders y DataBlocks**.

### 1.3.2. Código

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *

# En este caso vamos a descargar las imágenes utilizando la API de BING empleando 
# Microsoft Azure.

# Sustituir XXX por la clave KEY 1
key = os.environ.get('AZURE_SEARCH_KEY', 'XXX')

# En este ejemplo se pretendía obtener 3 carpetas con diferentes razas de osos para 
# clasificarlos posteriormente.
bear_types = 'grizzly','black','teddy'
path = Path('bears')

# Si el directorio no existe, lo creamos
if not path.exists(): 
    path.mkdir()
    
    # Para cada oso perteneciente a la lista de razas de osos, seleccionamos el destino cuyo nombre
    # de carpeta es igual a la raza, creamos dicho directorio, buscamos los resultados con la API de BING
    # y finalmente descargamos las imágenes
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok = True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))
        
# Obtenemos el directorio de la carpeta 'bears' con todas las imágenes
fns = ge_image_files(path)

# Verificamos si existen fallos en algunas de las imágenes descargadas
imagenes_fallos = verify_images(fns)
imagenes_fallos.map(Path.unlink)

# Descargados los datos, tenemos que preparalos para que sean compatibles con FastAI
# para ello creamos un DataLoaders personalizado
class DataLoaders(GetAttr):
    
    def __init__(self, *loaders):
        self.loaders = loaders
    
    def __getitem__(self, i):
        return self.loaders[i]
    
    train, valid = add_props(lambda i, self: self[i])

bears = DataBlock(
    		# Tupla para especificar (x, y), siendo 'x' la variable independiente, lo que usamos
    		# para hacer predicciones, en este caso imágenes e 'y' la varibale dependiente,
    		# que son las etiquetas, en este caso las categorías de los osos
			blocks = (ImageBlock, CategoryBlock),
    		get_items = get_item_files,
    		splitter = RandomSplitter(valid_pct = 0.2, seed = 42),
    		get_y = parent_label,
    		item_tfms = Resize(128)
			)
```



# Capítulo 2: Procesamiento del Lenguaje Natural, NLP

## 2.1. Análisis de reseñas/sentimientos, dataset IMDB

### 2.1.2. Código

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

# Los datos empleados en este problema son textos, las mejores arquitecturas para procesamiento
# de textos podría ser RNN (Redes Neuronales Recurrentes), Transformers o aplicar
# algoritmos de Machine Learning y no Deep Learning. 
# En este caso, queremos ver la relación de las palabras en el contexto de la oración
# al completo con el fin de conocer el contexto y aprender de este.
from fastai.text.all import *

path = untar_data(URLs.IMDB)

# El dataset cuenta con una carpeta con el texto para el entrenamiento y la validacion, 
# es decir, se encuentra separado por carpetas. Seleccionamos la carpeta para los
# datos de validación.
dls = TextDataLoaders.from_folders(path = path,
								   valid = 'test')

# En concreto vamos a utilizar redes LSTM (Long Short Term Memory)
# LSTM tienen múltiples probabilidades de abandono (drop out) para diferentes cosas. 
# Una vez establecidas, drop_mult escalará todas ellas, permitiendo cambiar todas las probabilidades 
# del drop out simultáneamente usando drop_mult.
leran = text_classifier_learner(dls = dls,
                                arch = AWD_LSTM, drop_mult = 0.5, metrics = accuracy)

# Realizamos 4 número de épocas.
# base_lr -> learning rate, ratio de aprendizaje, es el tamaño del paso a la hora de realizar
# SGD (Stochastic Gradient Descent) para obtener el mínimo en la función. 
# Un learning rate alto, hará que nos alejemos del mínimo de la función,
# mientras que un learning rate bajo el aprendizaje será muy lento.
learn.fine_tune(4, base_lr = 1e-2)

learn.predict("""
				El otro día vimos la película de Buzz Lightyear en cines y la verdad que nos encantó la película,
             	nos recordó a nuestra infancia.
			""")
```



# Capítulo 3: Datos tabulares

## 3.1. Detectar un alto nivel de ingresos basándose en el entorno socioeconómico

### 3.1.2. Código

```python
!pip install -Uqq fastbook
import fastbook()
fastbook.setup_book()

from fastai.tabular.all import *

path = untar_data(URLs.ADULT_SAMPLE)

# Datos categóricos hacen referencia a que contienen valores que son uno de un conjunto discreto
# de opciones.
categorical_names = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race']

# Datos continuos son datos que contienen un número como representación de una cantidad.
continuous_names = ['age', 'fnlwgt', 'education-num']

dls = TabularDataLoaders.from_csv(
    path/'adult.csv', path = path, y_names="salary",
    cat_names = categorical_names,
    cont_names = continuous_names,
    procs = [Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics = accuracy)

learn.fit_one_cycle(3)

# Mostrará todas las columnas del fichero CSV junto con la predicción de si s una persona con altos ingresos
# o no. Habrá una columna en el CSV con los datos reales (Salary) y otra columna con los datos obtenidos
# de la predicción (Salary_pred), indicando con un '1' si tiene un alto nivel de ingresos o
#'0' si no tiene un alto nivel de ingresos.
learn.show_results()
```

# Capítulo 4: Sistemas de recomendación

## 4.1. Predecir clasificaciones de una película

### 4.1.2. Código

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

path = untar_data(URLs.ML_SAMPLE)/'ratings.csv'

dls = CollabDataLoaders.from_csv(path)

# y_range -> especifica el rango de predicción
learn = collab_learner(dls, y_range = (0.5 , 5.5))

learn.fit_one_cycle(10)

learn.show_results()
```

