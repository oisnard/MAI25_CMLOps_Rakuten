import base64
import streamlit as st

# Fonction pour obtenir une image en base64
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Fonction pour afficher une image zoomable dans streamlit
def show_zoomable_image(img_path, caption=""):
    img_b64 = get_base64(img_path)

    # Charger Lightbox
    st.markdown("""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js"></script>
    """, unsafe_allow_html=True)

    # Affichage image + légende
    st.markdown(f"""
    <div style="text-align:center;">
        <a href="data:image/png;base64,{img_b64}" data-lightbox="archi" data-title="{caption}">
            <img src="data:image/png;base64,{img_b64}" width="100%" style="cursor: zoom-in;">
        </a>
        <div style="font-size:0.9em; color:gray; margin-top:5px;">{caption}</div>
    </div>
    """, unsafe_allow_html=True)

# Fonction pour convertir un nombre en chaîne de caractères avec un format spécifique
def convert_to_str(value):
    result = str("%.2f%%" % value)
    return result
