import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import src.tools.tools as tools
import requests
from src.api.middleware import create_jwt_token  

# Get test data
@st.cache_data
def get_test_data():
    X_test, y_test = tools.load_test_datasets()
    return X_test, y_test

def textdata_page(X_train, y_train, df_cat, IP_ADDRESS):
    st.header("Prédictions sur des données textuelles")
    st.write("Cette section propose de prédire la catégorie d'un produit à partir de ses caractéristiques textuelles à l'aide l'API mise en production.")

    X_test, y_test = get_test_data()

    df = pd.merge(left=X_train, right=y_train, how='left', left_index=True, right_index=True)

    df = df.loc[df.index.isin(X_test.index)]

    df_cat["title"] = df_cat.index.astype(str) + " - " + df_cat['Catégorie']

    st.subheader("Méthode : tirage aléatoire dans l'échantillon de test")

    list_selection = ["Toutes les catégories"]
    list_selection.extend(list(df_cat['title'].values))
    selected_title = st.selectbox("Choisir une catégorie de produits pour le tirage aléatoire d'un produit : ", list_selection)
    if selected_title == list_selection[0]:
        selected_code_type = -1
    else:
        selected_code_type = df_cat.loc[df_cat['title']==selected_title].index[0]

    if st.button("Tirage aléatoire d'un produit"):
        if selected_code_type == -1:
            product_index = df.sample(n=1).index[0]
        else:
            product_index = df.loc[df['prdtypecode']==selected_code_type].sample(n=1).index[0]
        st.write("Produit choisi aléatoirement : ")
        select_train = df.iloc[df.index==product_index][['designation', 'description', 'imageid', 'productid']]
        select_train = select_train.fillna("")
        designation = select_train['designation'].values[0]
        description = select_train['description'].values[0]
        imageid = select_train['imageid'].values[0]
        productid = select_train['productid'].values[0]
        filepath = tools.get_filepath_train(productid, imageid)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(select_train[['designation', 'description']].T, use_container_width=True)
        with col2:
            fig = plt.figure()
            img = plt.imread(filepath)
            plt.imshow(img)
            plt.axis('off')
            st.pyplot(fig)

        true_label = y_train.iloc[y_train.index==product_index]['prdtypecode'].values[0]
        true_cat = df_cat.iloc[df_cat.index==true_label]['Catégorie'].values[0]
        st.write("- Code type réel du produit : ", true_label, " - Catégorie : ", true_cat)



        designation = select_train['designation'].values[0]
        description = select_train['description'].values[0]
        payload = {
            "designation": designation,
            "description": description
        }
        token = create_jwt_token("test_user")
        API_URL = f"http://{IP_ADDRESS}/predict_product"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        with st.spinner("Appel de l'api pour prédiction en cours..."):
            response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            st.write("Prédiction réussie :")
            predicted_prdtypecode = response.json()['predicted prdtypecode']
            predicted_category = df_cat.iloc[df_cat.index==predicted_prdtypecode]['Catégorie'].values[0]
            st.write("- Code type prédit du produit : ", predicted_prdtypecode, " - Catégorie : ", predicted_category)
        else:
            st.write("Erreur lors de la prédiction :")
            st.write(response.text)

    st.subheader("Méthode : saisie manuelle de la désignation et description d'un produit")
    if "designation_txt" not in st.session_state:
        st.session_state.designation_txt = ""
    if "description_txt" not in st.session_state:
        st.session_state.description_txt = ""

    st.text_input("Désignation :", key="designation_txt")
    st.text_area("Description :", key="description_txt")

    if st.button("Valider"):
        if len(st.session_state.designation_txt.strip())==0:
            st.write("Veuillez saisir une désignation.")
        else:
            
            payload = {
                "designation": st.session_state.designation_txt,
                "description": st.session_state.description_txt
            }
            token = create_jwt_token("test_user")
            API_URL = f"http://{IP_ADDRESS}/predict_product"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            with st.spinner("Appel de l'api pour prédiction en cours..."):
                response = requests.post(API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                st.write("Prédiction réussie :")
                predicted_prdtypecode = response.json()['predicted prdtypecode']
                predicted_category = df_cat.iloc[df_cat.index==predicted_prdtypecode]['Catégorie'].values[0]
                st.write("- Code type prédit du produit : ", predicted_prdtypecode, " - Catégorie : ", predicted_category)
            else:
                st.write("Erreur lors de la prédiction :")
                st.write(response.text)


def aleatoire_page(x_train, y_train, df_cat):
    st.subheader("Prédictions sur des produits choisis de manière aléatoire")
    st.write("Cette section propose de réaliser des prédictions avec le modèle CamConv#1-0 (avec le paramètre MAX_LEN fixé à 128) sur des produits choisis aléatoirement dans 10% du fichier X_train (utilisé comme test pour calculer le F1-Score pondéré lors de la phase d'entraînement).")
    MAX_LEN = 128

    model = get_merged_model(MAX_LEN)    
    @tf.function
    def predict(x):
        return model(x)   

    list_prdtypecode = get_target_prdtypecode(y_train) #Pour le mapping entre range(0, 27) et les codes de la BD
    train = get_data()
    train = pd.merge(left=train, right=y_train, left_index=True, right_index=True, how='left')
    df_cat["title"] = df_cat.index.astype(str) + " - " + df_cat['Catégorie']
    list_selection = ["Toutes les catégories"]
    list_selection.extend(list(df_cat['title'].values))
    selected_title = st.selectbox("Choisir une catégorie de produits pour le tirage aléatoire d'un produit : ", list_selection)
    if selected_title == list_selection[0]:
        selected_code_type = -1
    else:
        selected_code_type = df_cat.loc[df_cat['title']==selected_title].index[0]

    tokenizer = get_tokenizer()
    if st.button("Tirage aléatoire d'un produit"):
        if selected_code_type == -1:
            product_index = train.sample(n=1).index[0]
        else:
            product_index = train.loc[train['prdtypecode']==selected_code_type].sample(n=1).index[0]
        st.write("Produit choisi aléatoirement : ")
        select_train = x_train.iloc[x_train.index==product_index][['designation', 'description', 'imageid', 'productid']]
        designation = select_train['designation'].values[0]
        description = select_train['description'].values[0]
        imageid = select_train['imageid'].values[0]
        productid = select_train['productid'].values[0]
        filepath = lib_file.get_image_train_filepath(imageid, productid)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(select_train[['designation', 'description']].T, use_container_width=True)
        with col2:
            fig = plt.figure()
            img = plt.imread(filepath)
            plt.imshow(img)
            plt.axis('off')
            st.pyplot(fig)

        true_label = y_train.iloc[y_train.index==product_index]['prdtypecode'].values[0]
        true_cat = df_cat.iloc[df_cat.index==true_label]['Catégorie'].values[0]
        st.write("- Code type réel du produit : ", true_label, " - Catégorie : ", true_cat)

        select_train['filepath'] = filepath
        with st.spinner("Préprocessing et calcul des prédictions en cours..."):
            pred_dataset = preprocess_train(select_train, tokenizer, MAX_LEN)
            input_img = tf.convert_to_tensor(pred_dataset['input_img'], dtype=tf.string)
            input_ids = tf.convert_to_tensor(pred_dataset['input_ids'])
            attention_masks = tf.convert_to_tensor(pred_dataset['attention_mask'])
            pred = predict([input_ids, attention_masks, input_img])
            val_pred = np.argmax(pred, -1)[0]
            val_pred = list_prdtypecode[val_pred]
        st.write(" - Code prédit = ", val_pred, " - Catégorie : ", df_cat.iloc[df_cat.index==val_pred]['Catégorie'].values[0])


