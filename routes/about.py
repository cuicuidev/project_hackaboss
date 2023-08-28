from components.section import Section
import streamlit as st

about_project = Section(text = 'Esta aplicación web forma parte del último proyecto de la tercera edición del bootcamp de Data Science de [Hackaboss](https://www.hackaboss.com/). \
                        Somos Jacobo y Dmitry, y siendo tan solo un equipo de dos personas, hemos estado invertiendo nuestro tiempo libre en \
                        hacer este proyecto realidad. Ambos apasionados por la tecnología, la programación y la inteligencia artificial, hemos decidido poner aquí en práctica todo lo que \
                        aprendimos acerca del aprendizaje profundo y las redes neurales de convolución durante el bootcamp.',
                        title = 'Acerca de :mag:')

def aboutRoute():
    about_project.render()
    st.write('---')
    contactUs()


def contactUs():
    st.write('## Contáctanos :email:')
    dmi, jac = st.columns((1,1))

    dmi.write('#### Dmitry Ryzhenkov')
    dmi.write('_Data Scientist, Software Engeneer_')
    dmi.write('Email :e-mail:: **dmitryryzhenkov.dev@gmail.com**')
    dmi.write('LinkedIn :page_facing_up:: [**cuicuidev**](https://www.linkedin.com/in/cuicuidev/)')
    dmi.write('GitHub :cat::computer:: [**cuicuidev**](https://github.com/cuicuidev)')
    dmi.write('Página web :newspaper:: [**cuicui.dev**](https://cuicui.dev)')

    
    jac.write('#### Jacobo Brandariz Morano')
    jac.write('_Data Scientist_')
    jac.write('Email :e-mail:: **___________**')
    jac.write('LinkedIn :page_facing_up:: [**___________**](https://www.linkedin.com/)')
    jac.write('GitHub :cat::computer:: [**___________**](https://github.com/)')
