# FastAI

# Índice

[TOC]

# Bibliografía

La mayor parte de la información se ha obtenido del libro "Deep Learning for Coders with Fast-AI & PyTorch".

# Útiles



# Capítulo 1

## 1.1. Reconocimiento de perros y gatos

### 1.1.1 Código

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

### 1.1.2. Anotaciones

+ Vamos a utilizar el dataset de Oxford_IIT PET Dataset con imágenes perros y gatos (37 razas).  **Para este dataset los nombres de los gatos vienen con la primera letra en mayúsculas.**

+ Usaremos modelos ya entrenados con el fin de realizar fine-tuning. 

  Recordar que un modelo pre-entrenado es un modelo cuyos pesos ya han sido adaptados y ajustados para una tarea. Con fine-tuning o ajuste fino conseguimos eliminar la última capa junto a sus pesos y la ajustamos a la nueva tarea (tener en cuenta que han de ser tareas similares entrenadas anteriormente y ahora con datos del mismo tipo, por ejemplo si el modelo fue entrenado con imágenes el dataset nuevo a emplear han de ser también basado en imágenes)

+ Podríamos emplear en transformaciones en un conjunto reducido de imágenes llamado **"batch"** y que permite ser alojados en la memoria RAM de la GPU, esto permite hacer el proceso de una manera más rápida. **Pero** hay que tener cuidado de no utilizar un **"batch_size"** (un tamaño de lote) muy grande ya que la GPU se podría quedar sin memoria RAM y a la hora de entrenar el modelo daría problemas relacionados con CUDA.

+ **Es importante saber que una clasificación pretende predecir una clase o categoría mientras que un modelo de regresión intenta predecir 1 0 más cantidades numéricas**.

+ **Metrics son distintas a la función de pérdida**

## 1.2. Segmentación de imágenes

### 1.2.1. Código

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

### 1.2.2. Anotaciones

+ Utilizamos una versión reducida del dataset de CAMVID preparada por FastAI.
+ **UNET** **se trata de una arquitectura empleada en segmentación que permite colorear cada elemento correspondiente a la clase a la que pertenece. UNET emplea la convolución transpuesta ya que esta permite aumentar el tamaño del volumen conforme avanzamos en la red (utilizando lo que se conocen como skip connections), con ello, a la salida obtenemos un resultado ya segmentado con la misma dimensión que la imagen introducida como input.**

### 1.2.3. Teoría redes convolucionales y detallado de UNET

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

