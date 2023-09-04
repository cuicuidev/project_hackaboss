import streamlit as st

from get_distribution_plot import plot_category_distribution

def edaRoute():
    header()

def header():
    st.title("Análisis de datos exploratorio/descriptivo")

    st.write("Dado que estamos trabando con imágenes, en esta sección solamente podemos mostrar la distribución de los datos para entrenamiento, \
             validación y test, así como la distribución de las diferentes categorías a predecir.")
    
    st.write("---")

    st.write("### Separación del dataset")

    st.write("El dataset en su forma original venía con las imágenes ya distribuidas por las carpetas 'train', 'test' y 'val', \
             por lo que no hizo falta separarlos manualmente. Según el creador del dataset, este mismo se ha encargado de \
             distribuir las imágenes de manera estratificada. En la figura de la derecha pueden ver la distribución de los datos \
             en entrenamiento, validación y test.")
    
    st.write("### Ditribución de categorías")
    
    st.write("En total hay 39 marcas de coches que podemos predecir, y no todas son igual de frecuentes. Aun así, la distribución de las \
             categorías no está demasiado desbalanceada. En la figura a la izquierda pueden apreciar la frecuencia con la que aparece cada marca en el dataset. \
             Exceptuando las 2 categorías menos frecuentes, la frecuencia del resto de marcas de coches es bastante similar.")

        
    fig = plot_category_distribution(sort_by='Total')  # You can replace 'Total' with 'train', 'test', or 'val'
    st.plotly_chart(fig, use_container_width = True, config={'displayModeBar': False})
    
    st.write("### Conclusiones")
    
    st.write("La distribución de los datos por entrenamiento, validación y test sigue unas proporciones muy estandar y acorda a las buenas \
             prácticas dentro de aprendizaje automático, por lo que creemos que debemos dejarlo como está. Asimismo, también hemos decidido continuar con el diseño de \
             la arquitectura sin aplicar ningúna técnica de balance de categorías. Aunque creemos que podríamos habernos beneficiado ligeramente si lo hubiésemos \
             hecho, la decisión fue tomada en base a nuestro limitado tiempo para desarrollar el modelo, ya que no nos podíamos permitir probar demasiadas permutaciones \
             de posibles cambios y estrategías a utilizar tanto en los datos como en los hiperparámetros del modelo e incluso arquitecutra de este.")