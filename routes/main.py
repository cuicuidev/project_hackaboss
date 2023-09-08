import streamlit as st

# header = Section(text = '##### _Para este proyecto hemos decidido utilizar redes convolucionales para clasificar imágenes de coches por marca_',
#                  title = 'Clasificación de coches por marca')

# section1 = Section(text = 'Para este proyecto teníamos la curiosidad de hacer uso, "pelearnos" y aprender más sobre una de las tantas herramientas que rodean el Deep Learning como son las CNN(Convolutional Neural Network) o redes neuronales convolucionales. Nos pareció interesante ver cómo podiamos entrenar un modelo para que reconociese una parte de una imagen y fuese capaz de clasificarla',
#                    header = 'Definición del proyecto',
#                    media = lambda x: x.image('assets/Tensorflow_logo.png', width = 500))

# section2 = Section(text = 'Ya que necesitabamos algo concreto como un dataset de imágenes decidimos buscar en Kaggle a ver que podíamos encontrar y después de charlarlo un poco entre todos nos decantamos por un dataset de imágenes que contenia entorno a unos 17.000 archivos de unas 35 marcas de coches diferentes, tenían buena calidad y se veían bien de tamaño para trabajar con ellas',
#                    header = 'Dataset',
#                    media = lambda x: x.image('assets/Kagglecom_logo.png', width = 500))

def mainRoute():
    
    st.title('Clasificación de coches por marca')
    st.write(' _Para este proyecto hemos decidido utilizar redes convolucionales para clasificar imágenes de coches por marca_')
    st.write(
        '---'
    )

    col1, _, col2 = st.columns((3,1,8))

    col2.header('Definición del proyecto')
    col2.write('Para este proyecto teníamos la curiosidad de hacer uso, "pelearnos" y aprender más sobre una de las tantas herramientas que rodean el Deep Learning como son las CNN(Convolutional Neural Network) o redes neuronales convolucionales. Nos pareció interesante ver cómo podiamos entrenar un modelo para que reconociese una parte de una imagen y fuese capaz de clasificarla')
    col1.write('')
    col1.write('')
    col1.write('')
    col1.write('')
    col1.image('assets/Tensorflow_logo.png', use_column_width = True)
    st.write('')
    st.write('')
    st.write('')

    col1, _, col2 = st.columns((8, 1, 3))

    col1.header('Dataset')
    col1.write('Ya que necesitabamos algo concreto como un dataset de imágenes decidimos buscar en Kaggle a ver que podíamos encontrar y después de charlarlo un poco entre todos nos decantamos por un dataset de imágenes que contenia entorno a unos 17.000 archivos de unas 39 marcas de coches diferentes, tenían buena calidad y se veían bien de tamaño para trabajar con ellas')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.write('')
    col2.image('assets/Kaggle_logo.png', width = 180)