from components.section import Section

header = Section(text = '##### _Haciendo uso de redes neurales convolucionales para clasificar imagenes de vehículos en base a sus marcas_', title = 'Clasificación de marcas de vehículos')
section1 = Section(text = 'Section 1 '*100, header = 'Definición del problema', media = lambda x: x.image('assets/Tensorflow_logo.png', width = 500))
section2 = Section(text = 'Section 2 '*200, header = 'Dataset')

def mainRoute():

    header.render()
    section1.render()
    section2.render()