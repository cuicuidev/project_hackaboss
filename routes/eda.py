from components.section import Section
import streamlit as st

eda_header = Section(text = 'Dado que estamos trabando con imágenes, en esta sección solamente podemos mostrar la distribución de los datos para entrenamient. \
                     validación y test, así como la distribución de las diferentes categorías a predecir.', title = 'Análisis de datos exploratorio/descriptivo')

train_test_distribution = Section(text = 'El dataset en su forma original venía con las imágenes ya distribuidas por las carpetas "train", "test" y "val", \
                                  por lo que no hizo falta separarlos manualmente. Según el creador del dataset, este mismo se ha encargado de \
                                  distribuir las imágenes de manera stratificada. En la figura de la derecha pueden ver la distribución de los datos \
                                  en entrenamiento, validación y test.',
                                  header = 'Entrenamiento, validación y test',
                                  media = lambda x: x.image('assets/Tensorflow_logo.png', width = 300),
                                  media_on_left = False,
                                  flex = (2,5),
                                  )

category_distribution = Section(text = 'En total hay 39 marcas de coches que podemos predecir, y no todas son igual de frecuentes. Aun así, la distribución de las \
                                categorías no está demasiado desbalanceada. En la figura a la izquierda pueden apreciar la frecuencia con la que aparece cada marca en el dataset. \
                                Exceptuando las 2 categorías menos frecuentes, la frecuencia del resto de marcas de coches es bastante similar.',
                                header = 'Categorías',
                                media = lambda x: x.image('assets/Tensorflow_logo.png', width = 300),
                                media_on_left = True,
                                flex = (2,5),
                                )

eda_ending = Section(text = 'La distribución de los datos por entrenamiento, validación y test sigue unas proporciones muy estandar y acorda a las buenas \
                     prácticas dentro de aprendizaje automático, por lo que creemos que debemos dejarlo como está. Asimismo, también hemos decidido continuar con el diseño de \
                     la arquitectura sin aplicar ningúna técnica de balance de categorías. Aunque creemos que podríamos habernos beneficiado ligeramente si lo hubiésemos \
                     hecho, la decisión fue tomada en base a nuestro limitado tiempo para desarrollar el modelo, ya que no nos podíamos permitir probar demasiadas permutaciones \
                     de posibles cambios y estrategías a utilizar tanto en los datos como en los hiperparámetros del modelo e incluso arquitecutra de este.', header = 'Conclusión')

def edaRoute():

    eda_header.render()
    train_test_distribution.render()
    category_distribution.render()
    eda_ending.render()