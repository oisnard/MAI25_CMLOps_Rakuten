import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from src.streamlit.functions import context
from src.streamlit.functions import exploration_text
from src.streamlit.functions import exploration_image
from src.streamlit.functions import archi
from src.streamlit.functions import portail
from src.streamlit.functions import tools_st
from src.streamlit.functions import model_st
from src.streamlit.functions import demo
from src.tools import tools_ntw
from src.tools import tools
from pathlib import Path
import os 



# Load RAW data
x_train = tools.load_xtrain_raw_data()
y_train = tools.load_ytrain_raw_data()
x_test = tools.load_xtest_raw_data()

# Load file with human interpretation of prdtypecode
df_cat = pd.read_csv(Path(tools.DATA_VIZ_DIR +"/features/classification_manuelle.csv"), sep=';', index_col=0)
X = pd.merge(left=x_train, right=y_train, how='left', left_index=True, right_index=True)
code_count = y_train.squeeze().value_counts(ascending=False)
df_stats = pd.DataFrame(code_count)
code_ratio = y_train.squeeze().value_counts(ascending=False, normalize=True)
df_stats['pourcentage'] = code_ratio.values * 100.
df_stats['prdtypecode'] = df_stats.index
df_stats['prdtypecode'] = df_stats['prdtypecode'].astype(str)
df_stats['pourcentage'] = df_stats['pourcentage'].apply(tools_st.convert_to_str)
df_stats = pd.merge(left=df_stats, right=df_cat, how='left', left_index=True, right_index=True)    





# Main app
def main_old():
    IP_ADDRESS = tools_ntw.get_ip()

    with st.sidebar:
        st.image(Path(tools.DATA_VIZ_DIR +"/assets/logo.png"), use_container_width=True)  # si tu as un logo
        selected = option_menu(
            "Projet Rakuten - MLOps", 
            ["Contexte", 
             "Exploration des données", 
             "Modèles", 
             "Architecture", 
             "Portail MLOps", 
             "Démonstrateur"],
            icons=["house", "bar-chart", "cpu", "diagram-3", "gear", "play-circle"],
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"font-size": "20px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )

    # ========================
    # PAGES
    # ========================

    if selected == "Contexte":
        context.context_show()

    elif selected == "Architecture":
        archi.archi_show()

    elif selected == "Portail MLOps":
        portail.portail_show(IP_ADDRESS)

    elif selected == "Exploration des données":
        st.header("📊 Exploration des données")
        submenu = option_menu(
            None,
            ["Résumé", "Données textuelles", "Images", "Données suspectes"],
            icons=["file-text", "type", "image", "exclamation-triangle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#f9f9f9"},
                "nav-link": {"font-size": "15px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )
        if submenu == "Résumé":
            exploration_text.explo_summary(X, df_cat, df_stats)
        elif submenu == "Données textuelles":
            exploration_text.explo_textdata(X, x_train, y_train, x_test, df_cat)
        elif submenu == "Images":
            exploration_image.exploration_image(y_train)
        elif submenu == "Données suspectes":
            exploration_text.explo_suspectdata(X, df_cat, df_stats, y_train)

    elif selected == "Modèles":
        st.header("Modèles")
        submenu_model = option_menu(
            None,
            ["Meilleur modèle challenge", "Modèles retenus (MLOps)"],
            icons=["trophy", "layers"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#f9f9f9"},
                "nav-link": {"font-size": "15px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )
        if submenu_model == "Meilleur modèle challenge":
            model_st.best_model(x_train, df_cat)
        elif submenu_model == "Modèles retenus (MLOps)":
            model_st.retained_models(x_train, df_cat)

    elif selected == "Démonstrateur":
        st.header("🧪 Démonstrateur interactif")
        st.caption("Testez la prédiction de catégories produit selon trois scénarios : texte seul, texte + image, ou batch de données.")
        submenu_demo = option_menu(
            None,
            ["Données textuelles", "Texte + Image", "Batch de données"],
            icons=["type", "image", "bar-chart-line"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#f9f9f9"},
                "nav-link": {"font-size": "15px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )
        if submenu_demo == "Données textuelles":
            demo.textdata_page(x_train, y_train, df_cat, IP_ADDRESS)
        elif submenu_demo == "Texte + Image":
            demo.textimage_page(x_train, y_train, df_cat, IP_ADDRESS)
        elif submenu_demo == "Batch de données":
            demo.batchdata_page(x_train, y_train, df_cat, IP_ADDRESS)


# Main app
def main():
    IP_ADDRESS = tools_ntw.get_ip()
    logo_path = Path(tools.DATA_VIZ_DIR +"/assets/logo.png")
    encoded_logo = tools_st.get_base64(logo_path)
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center;">
                <img src="data:image/png;base64,{encoded_logo}" width="90" style="margin-right:10px;">
                <h2 style="margin:0;">Projet Rakuten</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        selected = option_menu(
            None, 
            ["Contexte", 
             "Exploration des données", 
             "Modèles", 
             "Architecture", 
             "Portail MLOps", 
             "Démonstrateur"],
            icons=["house", "bar-chart", "cpu", "diagram-3", "gear", "play-circle"],
            menu_icon=None,
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"font-size": "20px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )

    # ========================
    # PAGES
    # ========================

    if selected == "Contexte":
        context.context_show()

    elif selected == "Architecture":
        archi.archi_show()

    elif selected == "Portail MLOps":
        portail.portail_show(IP_ADDRESS)

    elif selected == "Exploration des données":
        st.header("📊 Exploration des données")
        submenu = option_menu(
            None,
            ["Résumé", "Données textuelles", "Images", "Données suspectes"],
            icons=["file-text", "type", "image", "exclamation-triangle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#f9f9f9"},
                "nav-link": {"font-size": "15px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )
        if submenu == "Résumé":
            exploration_text.explo_summary(X, df_cat, df_stats)
        elif submenu == "Données textuelles":
            exploration_text.explo_textdata(X, x_train, y_train, x_test, df_cat)
        elif submenu == "Images":
            exploration_image.exploration_image(y_train)
        elif submenu == "Données suspectes":
            exploration_text.explo_suspectdata(X, df_cat, df_stats, y_train)

    elif selected == "Modèles":
        st.header("Modèles")
        submenu_model = option_menu(
            None,
            ["Meilleur modèle challenge", "Modèles retenus (MLOps)"],
            icons=["trophy", "layers"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#f9f9f9"},
                "nav-link": {"font-size": "15px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )
        if submenu_model == "Meilleur modèle challenge":
            model_st.best_model(x_train, df_cat)
        elif submenu_model == "Modèles retenus (MLOps)":
            model_st.retained_models(x_train, df_cat)

    elif selected == "Démonstrateur":
        st.header("🧪 Démonstrateur interactif")
        st.caption("Testez la prédiction de catégories produit selon trois scénarios : texte seul, texte + image, ou batch de données.")
        submenu_demo = option_menu(
            None,
            ["Données textuelles", "Texte + Image", "Batch de données"],
            icons=["type", "image", "bar-chart-line"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"background-color": "#f9f9f9"},
                "nav-link": {"font-size": "15px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#29A989", "color": "white"},
            }
        )
        if submenu_demo == "Données textuelles":
            demo.textdata_page(x_train, y_train, df_cat, IP_ADDRESS)
        elif submenu_demo == "Texte + Image":
            demo.textimage_page(x_train, y_train, df_cat, IP_ADDRESS)
        elif submenu_demo == "Batch de données":
            demo.batchdata_page(x_train, y_train, df_cat, IP_ADDRESS)


if __name__ == "__main__":
    main()
    

