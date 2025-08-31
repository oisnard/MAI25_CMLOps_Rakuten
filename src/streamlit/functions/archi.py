import streamlit as st
from pathlib import Path
from src.streamlit.functions.tools_st import show_zoomable_image


def archi_show():
    st.header("Architecture MLOps du projet")
    st.write("Ce projet est versionné et partagé sur GitHub : https://github.com/oisnard/MAI25_CMLOps_Rakuten")
    st.write("L'architecture du projet est résumée de manière simpliste dans l'image ci-dessous :")
    img_path = Path("data/dataviz/schema_archi.png")
    show_zoomable_image(img_path=img_path, caption="Schéma simplifié de l'architecture MLOps du projet")
