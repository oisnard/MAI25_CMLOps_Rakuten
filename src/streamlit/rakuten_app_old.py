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
def main():

    IP_ADDRESS = tools_ntw.get_ip()

    with st.sidebar:
        header = "Test",
        selected = option_menu(
            "Projet Rakuten - MLOps", 
            ["Contexte", "Exploration des données", "Modèles", "Architecture", "Portail MLOps", "Démonstrateur"],
            icons=["house",  "graph-up", "cpu", "diagram-3", "tools", "display"],
            default_index=0,
            styles = {
                "nav-link-selected" : {
                    "background-color" : "#29A989" 
                    }
                }
        )

    if selected == "Contexte":
        context.context_show()

    if selected == "Architecture":
        archi.archi_show()

    if selected == "Portail MLOps":
        portail.portail_show(IP_ADDRESS)

    if selected == "Exploration des données":
        st.header("Jeux de données mises-à-disposition")
        submenu = option_menu(
            None,  # No title for submenu
            ["Résumé", "Données textuelles", "Images", "Données suspectes"],
            menu_icon="cast",
            default_index=0,
            styles = {
                "nav-link-selected" : {
                    "background-color" : "#29A989" 
                    }
                }
        )
        if submenu == "Résumé":
            exploration_text.explo_summary(X, df_cat, df_stats)
        if submenu == "Données textuelles":
            exploration_text.explo_textdata(X, x_train, y_train, x_test, df_cat)
        if submenu == "Images":
            exploration_image.exploration_image(y_train)
        if submenu == "Données suspectes":
            exploration_text.explo_suspectdata(X, df_cat, df_stats, y_train)

    elif selected == "Modèles":
        submenu_model = option_menu(
            None,
            ["Meilleur modèle pour le challenge", "Modèles retenus pour le projet MLOps"],
            menu_icon="cast",
            default_index=0,
            styles = {
                "nav-link-selected" : {
                    "background-color" : "#29A989" 
                    }
            }
        )
        if submenu_model == "Meilleur modèle pour le challenge":
            model_st.best_model(x_train, df_cat)
        if submenu_model == "Modèles retenus pour le projet MLOps":
            model_st.retained_models(x_train, df_cat)

    elif selected == "Démonstrateur":
        st.header("Démonstrateur")
        submenu_demo = option_menu(
            None,  # No title for submenu
            ["Données textuelles", 
            "Données textuelles + Images",
            "Batch de données"],
            menu_icon="cast",
            default_index=0,
            styles = {
                "nav-link-selected" : {
                    "background-color" : "#29A989" 
                    }
                }
        )
        if submenu_demo=="Données textuelles":
            demo.textdata_page(x_train, y_train, df_cat, IP_ADDRESS)
        if submenu_demo=="Données textuelles + Images":
            demo.textimage_page(x_train, y_train, df_cat, IP_ADDRESS)
        if submenu_demo=="Batch de données":
            demo.batchdata_page(x_train, y_train, df_cat, IP_ADDRESS)


if __name__ == "__main__":
    main()
    

