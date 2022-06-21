# FastAI

# Índice

[TOC]

# Bibliografía

La mayor parte de la información se ha obtenido del libro "Deep Learning for Coders with Fast-AI & PyTorch".

# Útiles



# Capítulo 1

## 1. Reconocimiento de perros y gatos

### 1.1. Anotaciones

+ Vamos a utilizar el dataset de Oxford_IIT PET Dataset con imágenes perros y gatos (37 razas).

+ Usaremos modelos ya entrenados con el fin de realizar fine-tuning. 

  Recordar que un modelo pre-entrenado es un modelo cuyos pesos ya han sido adaptados y ajustados para una tarea. Con fine-tuning o ajuste fino conseguimos eliminar la última capa junto a sus pesos y la ajustamos a la nueva tarea (tener en cuenta que han de ser tareas similares entrenadas anteriormente y ahora con datos del mismo tipo, por ejemplo si el modelo fue entrenado con imágenes el dataset nuevo a emplear han de ser también basado en imágenes)

### 1.2. Código

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastai.vision.all import *
from fastbook import *

def es_gato(x):
    return x[0].isupper()

path = untar_data(URLs.PETS)/'images'

dls = ImageDataLoaders.from_name_func(path, get_image_files(path), 
                                      valid_pct = 0.2, seed = 42, 
                                      label_func = es_gato, 
                                      item_tfms = Resize(224))

learn = vision_learner(dls, resnet34, metrics = [accuracy, error_rate])

learn.fine_tune(3)

boton_subida = widgets.FileUpload()
boton_subida

img = PILImage.create(boton_subida.data[0])
es_gato,_,probs = learn.predict(img)

print(f"¿Es un gato? {es_gato}")
print(f"Probabilidad de que sea un gato: {probs[1].item():.6f}")
```

