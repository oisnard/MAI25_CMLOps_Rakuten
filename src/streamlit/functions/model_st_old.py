import numpy as np
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import altair as alt
from functions.tools_st import convert_to_str, show_zoomable_image
import pickle
import src.tools.tools as tools
from pathlib import Path




@st.cache_data
def get_camconv_struct():
    img_struct = plt.imread(Path(tools.DATA_VIZ_DIR+"/models/camconv.png"))
    return img_struct



def best_model(x_train, df_cat):
    url_hugging = "https://huggingface.co/docs/transformers/index"
    url_camembert = "https://huggingface.co/docs/transformers/model_doc/camembert"
    url_tensor = "https://www.tensorflow.org/"    
    url_keras = "https://keras.io/api/applications/"
    url_conv = "https://www.tensorflow.org/api_docs/python/tf/keras/applications/ConvNeXtLarge"

    st.header("Meilleur Modèle")
    st.write("Dans cette section, nous présentons le meilleur modèle obtenu pour le challenge. Ce modèle est le résultat du projet Rakuten réalisé par Olivier ISNARD, Julien TREVISAN et Bruno RAMAYE dans le cadre de leur formation Data Scientist.")
    st.markdown(
    'Ce modèle a permis d\'atteindre la [première place du classement](https://challengedata.ens.fr/participants/challenges/35/ranking/public) du challenge 🏆' 
    )   
    st.write("Le modèle retenu est basé sur :")
    st.write("- Un modèle textuel basé sur [CamemBERT](%s) de [Transformers](%s) ;" %(url_camembert, url_hugging))
    st.write("- Un modèle 'image' base sur [ConvNeXtLarge](%s) des librairies [keras](%s) de [TensorFlow](%s)." %(url_conv, url_keras, url_tensor))

    st.subheader("Préprocessing des données")
    st.write("Le préprocessing retenu et donnant les meilleures performances de F1-score est le suivant :")
    st.write("- Pour les données textuelles : remplacement des NaNs par une chaîne de caractère vide, suppression des balises html avec les librairies de [BeautifulSoup](%s), fusion de la désignation et description séparées par les caractères '. ' ;" % "https://pypi.org/project/beautifulsoup4/")
    st.write("- Pour les images : seule la fonction tf.keras.applications.convnext.preprocess_input est appliquée. La taille des images est maintenue à 500x500 pixels.")
    st.subheader("Tokenisation des données textuelles")
    st.write("Le tokenizer CamemBERT pré-entraîné sur [camembert-base](%s) est appliqué sur les données textuelles. Différentes valeurs du nombre maximal de tokens par produit ont été testées (128, 256 et 512)." % "https://huggingface.co/almanach/camembert-base")
    st.write("Les meilleurs résultats ont été obtenus avec un nombre limite de 128 et 256 tokens par produit (paramètre *MAX_LEN*).")
    st.subheader("Structure du modèle fusionné")
    st.write("La partie CamemBERT du modèle fusionné est basé sur la classe [TFCamembertModel](%s) de transformers. En sortie du modèle TFCamembertModel, un Reshape de la hidden layer est appliqué puis une couche Dense à 768 unités avec une fonction d'activation tangente hyperbolique." %("https://huggingface.co/docs/transformers/model_doc/camembert#transformers.TFCamembertModel"))
    st.write("En sortie du modèle ConvNeXtLarge (dont les poids sont initialisés de manière aléatoire) , une couche GlobalAveragePooling2D et une couche Dense de 768 unités (fonction d'activation Relu) sont appliquées. Les 30 dernières couches du modèle sont rendues entraînables, exceptées les éventuelles couches de BatchNormalization.")
    st.write("La fusion des 2 modèles est réalisée avec une couche Concatenate. Deux couches Dense (de 1536 et 512 unités avec la fonction Relu) sont ensuite appliquées ainsi qu'une dernière couche Dense de 27 unités (fonction d'activation softmax) pour la classification.")
    st.write("La structure et le résumé du modèle généré sont affichés dans la figure suivante :")
    show_zoomable_image(Path(tools.DATA_VIZ_DIR+"/models/camconv.png"), caption="Structure du modèle fusionné CamemBERT + ConvNeXtLarge")
 
    st.write("Le nombre de paramètres du modèle dépend de la valeur *MAX_LEN* (le nombre maximal de tokens par produit) :")
    df_summary = pd.read_csv(Path(tools.DATA_VIZ_DIR +"/models/model_summary2.csv"), sep=';')
    df_summary_melt = df_summary.melt(id_vars="MAX LEN", var_name="Type", value_name="Nombre")
    df_summary_melt['MAX LEN'] = df_summary_melt['MAX LEN'].astype(str)
    df_summary_melt['Nombre'] /= 1e6
    titre_chart = "Nombre de paramètres en fonction du paramètre MAX_LEN"
    chart_param = alt.Chart(df_summary_melt).mark_bar().encode(
        y='MAX LEN',
        x=alt.X('Nombre:Q', title="Nombre (Millions)"),
        color='Type',
        tooltip=['MAX LEN', 'Type', 'Nombre']
    ).properties(title=alt.TitleParams(titre_chart, align='center')).configure_title(anchor='middle')
    st.altair_chart(chart_param)

    st.subheader("Entraînement du modèle")
    st.write("Dans une première étape, le jeu de données X_train a été découpé en 3 parties afin de pouvoir comparer les résultats du modèle fusionné sur le même jeu de données pour calculer le F1-Score des modèles textuels et images.")
    st.write("Le jeu de données X_train a été découpé en 3 parties avec la fonction *train_test_split* de *scikit-learn* en utilisant le paramètre *stratify* afin de conserver les mêmes proportions de labels dans les différents jeux de données générés :")
    st.write("- Un jeu d'entraînement représentant 72.25% des données du jeu X_train ;")
    st.write("- Un jeu de validation pendant l'entraînement représentant 12.75% du jeu X_train ;")
    st.write("- Un jeu de test représentant 15% du jeu de X_train.")
    st.write("La fonction de perte permettant d'obtenir les meilleurs résultats est la Spare Categorical Entropy et l'optimiseur est celui d'AdamW avec le paramètre *Amsgrad* activé et un taux d'apprentissage initial de 2e-5.")

    st.subheader("F1-Score sur le jeu de test issu du découpage de X_train")
    st.write("Le F1-score pondéré obtenu après avoir entraîné le modèle fusionné sur 72.25% du jeu X_train est d'environ 91.8%. La déclinaison du F1-score par code type est la suivante :")

    filename = tools.DATA_VIZ_DIR + "/models/cr_fusion_cam_conv-1-0bis.pkl"
    with open(filename, "rb") as f:
        cr_dict_ref = pickle.load(f)

    df_fusion = pd.DataFrame.from_dict(cr_dict_ref).T.iloc[:-3,:]
    df_fusion = df_fusion.drop(columns=['precision', 'recall', 'support'], axis=1)
    df_fusion['PrdTypeCode'] = df_fusion.index.astype(int)
    df_fusion = df_fusion.sort_values(by="f1-score", ascending=False)
    df_fusion['F1-Score'] = df_fusion['f1-score']*100.
    df_fusion['F1-Score'] = df_fusion['F1-Score'].apply(convert_to_str)
    df_fusion = pd.merge(left=df_fusion, right=df_cat, left_on="PrdTypeCode", right_on="prdtypecode", how='left')
        
    chart_f1_details_fusion = alt.Chart(df_fusion).mark_bar().encode(
        x=alt.X('PrdTypeCode:N', sort=list(df_fusion['f1-score'].values)),
        y='f1-score',
        tooltip = ['PrdTypeCode', 'Catégorie', 'F1-Score'],
        color=alt.Color('f1-score', scale=alt.Scale(scheme='viridis', reverse=True))  # Apply Viridis color scale
    ).properties(width=600, height=400).interactive()
    chart_f1_global_fusion = alt.Chart(pd.DataFrame({'f1-score' : [cr_dict_ref["weighted avg"]["f1-score"]]})).mark_rule(color='orange', size=2).encode(y='f1-score')
    chart_comb_fusion = chart_f1_details_fusion + chart_f1_global_fusion
    st.altair_chart(chart_comb_fusion, use_container_width=True)        
    st.write("En comparant le F1-Score obtenu pour chaque code type de produit avec le modèle fusionné et le modèle CamemBERT sur le même jeu de test (issu du découpage de X_train), on peut observer les bénéfices de l'ajout du modèle image ConvNeXtLarge.")
    df_fusion["Modèle"] = "Modèle fusionné"

    filename = tools.DATA_VIZ_DIR + "/models/cr_testCam5-17_pretrain_adamW.pkl"
    with open(filename, "rb") as f:
        cr_dict_cam = pickle.load(f)

    df_cam = pd.DataFrame.from_dict(cr_dict_cam).T.iloc[:-3,:]
    df_cam = df_cam.drop(columns=['precision', 'recall', 'support'], axis=1)
    df_cam['PrdTypeCode'] = df_cam.index.astype(int)
    df_cam = df_cam.sort_values(by="f1-score", ascending=False)
    df_cam['F1-Score'] = df_cam['f1-score']*100.
    df_cam['F1-Score'] = df_cam['F1-Score'].apply(convert_to_str)
    df_cam = pd.merge(left=df_cam, right=df_cat, left_on="PrdTypeCode", right_on="prdtypecode", how='left')
    df_cam["Modèle"] = "Modèle CamemBERT"
    df_merge = pd.concat([df_fusion, df_cam], ignore_index=True)
    chart_f1_details_both = alt.Chart(df_merge).mark_bar().encode(
        x=alt.X('PrdTypeCode:N', sort=list(df_fusion['f1-score'].values)),
        y=alt.Y('f1-score', scale=alt.Scale(domain=[0.6, 1.])),
        tooltip = ['PrdTypeCode', 'Catégorie', 'Modèle', 'F1-Score'],
        color='Modèle',
        xOffset='Modèle:N'
    ).properties(width=600, height=400, title="Comparaison des F1-Score du modèle fusionné et du modèle CamemBERT").configure_title(anchor='middle').interactive()
    st.altair_chart(chart_f1_details_both, use_container_width=True)   

    st.subheader("Entraînement du modèle pour les prédictions du jeu X_test")
    st.write("En vue de soumettre les prédictions des codes types des produits référencés dans le jeu X_test, le modèle fusionné a été entraîné sur 90% du jeu de données X_train avec 10% des données conservées pour la validation et activer les callbacks d'ajustements du learning rate pendant l'entraînement.")
    st.write("Voici le F1-Score calculé sur le jeu de données de validation après l'entraînement du modèle fusionné sur 10 epochs : ")
    st.write("- Le F1 Score du modèle fusionné avec le paramètre *MAX_LEN* fixé à 128 est de 92.37% ;")
    st.write("- Le F1 Score du modèle fusionné avec le paramètre *MAX_LEN* fixé à 256 est de 92.36% ;")
    st.write("- Le F1 Score du modèle fusionné avec le paramètre *MAX_LEN* fixé à 512 est de 91.30%.")
    st.write("L'augmentation du paramètre MAX_LEN dégrade la prédiction des produits dont le nombre de tokens est faible (<64) alors que ces produits représentent presque la moitié du jeu de données.")
    st.write("La figure suivante montre le F1-Score décliné par code type obtenu par le paramètre *MAX_LEN* à 128 : ")
    filename = tools.DATA_VIZ_DIR + "/models/cr_fusion_cam_conv_90train_128.pkl"
    with open(filename, "rb") as f:
        cr_dict_ref90 = pickle.load(f)

    df_fusion90 = pd.DataFrame.from_dict(cr_dict_ref90).T.iloc[:-3,:]
    df_fusion90 = df_fusion90.drop(columns=['precision', 'recall', 'support'], axis=1)
    df_fusion90['PrdTypeCode'] = df_fusion90.index.astype(int)
    df_fusion90 = df_fusion90.sort_values(by="f1-score", ascending=False)
    df_fusion90['F1-Score'] = df_fusion90['f1-score']*100.
    df_fusion90['F1-Score'] = df_fusion90['F1-Score'].apply(convert_to_str)
    df_fusion90 = pd.merge(left=df_fusion90, right=df_cat, left_on="PrdTypeCode", right_on="prdtypecode", how='left')
        
    chart_f1_details_fusion90 = alt.Chart(df_fusion90).mark_bar().encode(
        x=alt.X('PrdTypeCode:N', sort=list(df_fusion90['f1-score'].values)),
        y='f1-score',
        tooltip = ['PrdTypeCode', 'Catégorie', 'F1-Score'],
        color=alt.Color('f1-score', scale=alt.Scale(scheme='viridis', reverse=True))  # Apply Viridis color scale
    ).properties(width=600, height=400).interactive()
    chart_f1_global_fusion90 = alt.Chart(pd.DataFrame({'f1-score' : [cr_dict_ref90["weighted avg"]["f1-score"]]})).mark_rule(color='orange', size=2).encode(y='f1-score')
    chart_comb_fusion90 = chart_f1_details_fusion90 + chart_f1_global_fusion90
    st.altair_chart(chart_comb_fusion90, use_container_width=True)       

    st.write("Un entraînement sur le jeu complet des données X_train est réalisé sur une seule epoch (afin de limiter le sur-apprentissage du modèle). Les prédictions du jeu X_test ont été soumises au site organisateur du challenge Rakuten. Les résultats obtenus sont les suivants :")
    st.write("- Avec le paramètre *MAX_LEN* fixé à 128, le F1-Score pondéré obtenu est de 91.89% ;")
    st.write("- Avec le paramètre *MAX_LEN* fixé à 256, le F1-Score pondéré est de 91.96%.")
    st.write("Les résultats obtenus nous ont permis d'atteindre la première place au classement public (en date du 28 Février 2025) et d'améliorer sensiblement le meilleur résultat précédent qui datait de Décembre 2020.")
    fig = plt.figure()
    img_ranking = plt.imread(tools.DATA_VIZ_DIR + "/models/classement_rakuten.png")
    plt.imshow(img_ranking)
    plt.axis('off')
    plt.title("Extrait du classement public du challenge Rakuten (28 Février 2025)", fontsize=8)
    st.pyplot(fig)        

    # Code pour afficher le meilleur modèle

def retained_models(x_train, df_cat):
    st.header("Modèles retenus")
    st.write("Dans cette section, nous présentons les modèles retenus pour le projet MLOps.")

