import streamlit as st

from root.config import PAGE_CONFIG
from root.route_paths import routepaths

def main():

    st.set_page_config(**PAGE_CONFIG)

    try:
        route_name = st.sidebar.selectbox(label = "Menu", options = list(routepaths.keys()), index = 0)

        routepaths[route_name]()
    except KeyError as e:
        st.error(f'Ha acurrido un error seleccionando la página. La clave `{e}` no existe en `routepaths`')

if __name__ == '__main__':
    main()