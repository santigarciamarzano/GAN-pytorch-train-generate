# GAN-Project: Generación de Imágenes de Perros (CIFAR-10)

El proyecto está organizado de manera modular para facilitar su uso y comprensión.

## Estructura del Proyecto

La estructura del proyecto está organizada de la siguiente manera:

´´´sh
/red-GAN-train-pytorch
    ├── src/
    │   ├── generator.py        # Red generadora
    │   ├── discriminator.py    # Red discriminadora
    │   ├── training.py         # Función de entrenamiento
    │   ├── utils.py            # Funciones auxiliares
    ├── main_notebook.ipynb     # Archivo Jupyter Notebook
    ├── README.md               # Documentación del proyecto
    ├── requirements.txt        # Librerías necesarias

'''

## Descripción de Archivos:

- src/generator.py: Contiene la definición de la red generadora, que toma como entrada un vector de ruido aleatorio y genera imágenes de perros.

- src/discriminator.py: Define la red discriminadora, que clasifica si una imagen es real (del dataset) o falsa (generada por el generador).

- src/training.py: Implementa la función de entrenamiento, encargada de alternar entre el entrenamiento del generador y del discriminador.

- src/utils.py: Incluye funciones auxiliares, como las funciones de pérdida (loss) y la generación de ruido aleatorio para alimentar al generador.

- main_notebook.ipynb: Es el notebook principal desde el cual se ejecuta el código del proyecto. El notebook está dividido en bloques que llaman a las diferentes funciones y clases definidas en los archivos de la carpeta src/.

## Requisitos

Primero, instala las dependencias necesarias para ejecutar el proyecto. Estas están listadas en el archivo requirements.txt.

´´´sh
pip install -r requirements.txt

´´´

## Ejecutar el Proyecto desde el Notebook

El código del proyecto se ejecuta en bloques utilizando el notebook run_all_functions.ipynb. Este archivo sirve como guía para ejecutar cada una de las partes del proyecto de manera secuencial.

Pasos:

- Cargar el Dataset CIFAR-10: El notebook carga y procesa el dataset CIFAR-10, seleccionando únicamente las imágenes correspondientes a la clase "perros" (etiqueta 5 en el dataset).

- Entrenar el Modelo: Desde el notebook se invocan las funciones de entrenamiento ubicadas en src/training.py, las cuales se encargan de entrenar tanto la red generadora como la discriminadora.

- Visualización de Resultados: Al final de cada época de entrenamiento, el notebook genera y muestra imágenes producidas por la red generadora.

## Visualizar las Imágenes Generadas

A medida que el entrenamiento progresa, las imágenes generadas por la red se mostrarán en el notebook para que puedas evaluar visualmente su calidad. Estas imágenes se desnormalizan para su correcta visualización.

## Dataset

El proyecto utiliza el dataset CIFAR-10, que contiene imágenes de 32x32 píxeles clasificadas en 10 categorías. Solo se utiliza la clase correspondiente a perros (etiqueta 5) para entrenar el modelo.

## Resultados Esperados

El entrenamiento genera imágenes de perros que mejoran en calidad a medida que las épocas avanzan. Al final del entrenamiento, el generador debería ser capaz de producir imágenes realistas de perros.
