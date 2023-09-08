import streamlit as st

def aboutRoute():
    about()
    st.write('---')
    contactUs()

def about():
    st.title('Acerca de :mag:')
    st.write('Esta aplicación web forma parte del último proyecto del bootcamp de Data Science de [Hackaboss](https://www.hackaboss.com/). \
             Somos Jacobo y Dmitry, dos muchachos muy apasionados por la ciencia de datos y la tecnología de la información. \
             Juntos hemos entrenado una red neural que permite clasificar turismos por su marca de fabricante. Asimismo, hemos creado esta aplicación web como medio \
             para interactuar con el modelo y documentar su desarrollo, las técnicas que hemos utilizado y las barreras que hemos tenido \
             que superar en el proceso. Ha sido una dura e intensa trayectoria, pero no solo hemos podido aplicar un montón de conocimientos adquiridos durante \
             nuestra formación, sino que también hemos expandido nuestros conocimientos acerca del aprendizaje profundo. \
             Si bien el modelo final no está a la altura de nuestas expectativas, el nivel en el que hemos podido profundizar en la comprension de \
             las redes neurales convolutivas las ha superado.')

def contactUs():
    st.write('## Contáctanos :email:')
    dmi, jac = st.columns((1,1))

    dmi.write('#### Dmitry Ryzhenkov')
    dmi.write('_Data Scientist, Software Developer_')
    dmi.write('Email :e-mail:: **dmitryryzhenkov@cuicui.dev**')
    dmi.write('LinkedIn :page_facing_up:: [**cuicuidev**](https://www.linkedin.com/in/cuicuidev/)')
    dmi.write('GitHub :cat::computer:: [**cuicuidev**](https://github.com/cuicuidev)')
    dmi.write('Página web :newspaper:: [**cuicui.dev**](https://cuicui.dev)')

    
    jac.write('#### Jacobo Brandariz Morano')
    jac.write('_Data Scientist_')
    jac.write('Email :e-mail:: **jacobobm.dsp@gmail.com**')
    jac.write('LinkedIn :page_facing_up:: [**JacoboBM**](https://www.linkedin.com/in/jacobo-brandariz-morano)')
    jac.write('GitHub :cat::computer:: [**JacoboBM**](https://github.com/JacoboBM)')