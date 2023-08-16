from components.section import Section

header = Section(text = 'Text1 '*100, title = 'Example section without media', header = 'header')
section1 = Section(text = 'Text2 '*100, title = 'Example section with media', media = lambda x: x.image('assets/Tensorflow_logo.png', width = 300))

def mainRoute():

    header.render()
    section1.render()