import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import src.tools.tools as tools
import requests
from src.api.middleware import create_jwt_token  
from sklearn.metrics import classification_report
from functions.tools_st import convert_to_str

# Get test data
@st.cache_data
def get_test_data():
    X_test, y_test = tools.load_test_datasets()
    return X_test, y_test

def textdata_page(X_train, y_train, df_cat, IP_ADDRESS):
    st.header("📦 Démonstration de prédiction produit")
    st.caption("Prédire la catégorie d’un produit à partir de sa désignation et description.")

    # Données
    X_test, y_test = get_test_data()
    df = pd.merge(left=X_train, right=y_train, how="left", left_index=True, right_index=True)
    df = df.loc[df.index.isin(X_test.index)]
    df_cat["title"] = df_cat.index.astype(str) + " - " + df_cat["Catégorie"]

    # Onglets
    tab1, tab2 = st.tabs(["🎲 Tirage aléatoire", "⌨️ Saisie manuelle"])

    # --- Tirage aléatoire ---
    with tab1:
        st.subheader("Prédire à partir d’un produit de l’échantillon test")

        # Sélection de catégorie
        list_selection = ["Toutes les catégories"] + list(df_cat["title"].values)
        selected_title = st.selectbox(
            "Choisir une catégorie pour le tirage aléatoire :", list_selection
        )
        selected_code_type = (
            -1 if selected_title == list_selection[0]
            else df_cat.loc[df_cat["title"] == selected_title].index[0]
        )

        if st.button("🔀 Tirer un produit"):
            # Tirage produit
            if selected_code_type == -1:
                product_index = df.sample(n=1).index[0]
            else:
                product_index = df.loc[df["prdtypecode"] == selected_code_type].sample(n=1).index[0]

            select_train = df.loc[df.index == product_index][
                ["designation", "description", "imageid", "productid"]
            ].fillna("")

            designation = select_train["designation"].values[0]
            description = select_train["description"].values[0]
            imageid = select_train["imageid"].values[0]
            productid = select_train["productid"].values[0]
            filepath = tools.get_filepath_train(productid, imageid)

            # Affichage produit
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**📝 Désignation :** {designation}")
                st.markdown(f"**📖 Description :** {description[:1000]}")
            with col2:
                st.image(filepath, caption="Image du produit", use_container_width=True)

            # Vérité terrain
            true_label = y_train.loc[y_train.index == product_index]["prdtypecode"].values[0]
            true_cat = df_cat.loc[df_cat.index == true_label]["Catégorie"].values[0]
            st.info(f"✅ Code type réel : {true_label} - {true_cat}")

            # Prédiction API
            payload = {"designation": designation, "description": description}
            token = create_jwt_token("test_user")
            API_URL = f"http://{IP_ADDRESS}/predict_product"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            with st.spinner("Appel de l'API en cours..."):
                response = requests.post(API_URL, json=payload, headers=headers)

            if response.status_code == 200:
                predicted_code = response.json()["predicted prdtypecode"]
                predicted_cat = df_cat.loc[df_cat.index == predicted_code]["Catégorie"].values[0]
                # --- Comparatif prédiction ---
                colA, colB = st.columns(2)
                with colA:
                    st.metric(label="Vérité terrain", value=f"{true_label}", delta=true_cat, border=True)
                with colB:
                    if true_label != predicted_code:
                        predicted_cat = "-" + predicted_cat
                    st.metric(label="Prédiction", value=f"{predicted_code}", delta=predicted_cat, border=True)                
                st.success(f"🎯 Prédiction : {predicted_code} - {predicted_cat}")
            else:
                st.error(f"Erreur API : {response.text}")

    # --- Saisie manuelle ---
    with tab2:
        st.subheader("Prédire à partir d’une saisie manuelle")

        st.text_input("Désignation :", key="designation_txt", placeholder="Ex: Smartphone Xiaomi 128Go")
        st.text_area("Description :", key="description_txt", placeholder="Ex: Téléphone portable 6,5 pouces, 128Go, double SIM")

        if st.button("🚀 Lancer la prédiction"):
            if len(st.session_state.designation_txt.strip()) == 0:
                st.warning("Veuillez saisir une désignation.")
            else:
                payload = {
                    "designation": st.session_state.designation_txt,
                    "description": st.session_state.description_txt,
                }
                token = create_jwt_token("test_user")
                API_URL = f"http://{IP_ADDRESS}/predict_product"
                headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

                with st.spinner("Appel de l'API en cours..."):
                    response = requests.post(API_URL, json=payload, headers=headers)

                if response.status_code == 200:
                    predicted_code = response.json()["predicted prdtypecode"]
                    predicted_cat = df_cat.loc[df_cat.index == predicted_code]["Catégorie"].values[0]

                    st.success(f"🎯 Prédiction : {predicted_code} - {predicted_cat}")
                else:
                    st.error(f"Erreur API : {response.text}")






def textimage_page(X_train, y_train, df_cat, IP_ADDRESS):
    st.header("🖼️📦 Démonstration : Prédiction produit (Texte + Image)")
    st.caption("Cette page illustre la prédiction de catégorie d’un produit en utilisant sa désignation, sa description **et** son image.")

    # Données
    X_test, y_test = get_test_data()
    df = pd.merge(left=X_train, right=y_train, how="left", left_index=True, right_index=True)
    df = df.loc[df.index.isin(X_test.index)]
    df_cat["title"] = df_cat.index.astype(str) + " - " + df_cat["Catégorie"]

    st.subheader("🎲 Tirage aléatoire d’un produit")

    # Sélection de catégorie
    list_selection = ["Toutes les catégories"] + list(df_cat["title"].values)
    selected_title = st.selectbox("Filtrer par catégorie :", list_selection)
    selected_code_type = (
        -1 if selected_title == list_selection[0]
        else df_cat.loc[df_cat["title"] == selected_title].index[0]
    )

    if st.button("🔀 Tirer un produit"):
        # --- Tirage ---
        if selected_code_type == -1:
            product_index = df.sample(n=1).index[0]
        else:
            product_index = df.loc[df["prdtypecode"] == selected_code_type].sample(n=1).index[0]

        select_train = df.loc[df.index == product_index][
            ["designation", "description", "imageid", "productid"]
        ].fillna("")

        designation = select_train["designation"].values[0]
        description = select_train["description"].values[0]
        imageid = select_train["imageid"].values[0]
        productid = select_train["productid"].values[0]
        filepath = tools.get_filepath_train(productid, imageid)

        # --- Carte produit ---
        with st.container():
            st.markdown("### 🛍️ Produit sélectionné")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**📝 Désignation :** {designation}")
                st.markdown(f"**📖 Description :** {description[:1000]}")
            with col2:
                st.image(filepath, caption="Image du produit", use_container_width=True)

        # Vérité terrain
        true_label = y_train.loc[y_train.index == product_index]["prdtypecode"].values[0]
        true_cat = df_cat.loc[df_cat.index == true_label]["Catégorie"].values[0]

        st.info(f"✅ Vérité terrain : {true_label} — {true_cat}")

        # --- Appel API ---
        payload = {
            "designation": designation,
            "description": description,
            "image_filepath": filepath
        }
        token = create_jwt_token("test_user")
        API_URL = f"http://{IP_ADDRESS}/predict_product"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        with st.spinner("⏳ Envoi à l’API..."):
            response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            predicted_code = response.json()["predicted prdtypecode"]
            predicted_cat = df_cat.loc[df_cat.index == predicted_code]["Catégorie"].values[0]

            # --- Comparatif prédiction ---
            colA, colB = st.columns(2)
            with colA:
                st.metric(label="Vérité terrain", value=f"{true_label}", delta=true_cat, border=True)
            with colB:
                if true_label != predicted_code:
                    predicted_cat = "-" + predicted_cat
                st.metric(label="Prédiction", value=f"{predicted_code}", delta=predicted_cat, border=True)

            st.success(f"🎯 Catégorie prédite : **{predicted_cat}**")
        else:
            st.error(f"🚨 Erreur API : {response.text}")


def batchdata_page_old(X_train, y_train, df_cat, IP_ADDRESS):
    st.title("🖼️📦 Prédiction sur un batch de données")
    st.caption("Cette page illustre la prédiction des catégories de plusieurs produits en batch.")

    # Données
    X_test, y_test = get_test_data()
    df = pd.merge(left=X_train, right=y_train, how="left", left_index=True, right_index=True)
    df = df.loc[df.index.isin(X_test.index)]
    df_cat["title"] = df_cat.index.astype(str) + " - " + df_cat["Catégorie"]

    if "nb_products" not in st.session_state:
        st.session_state.nb_products = 100
    nb_products = st.session_state.nb_products
    nb_products = st.slider("Sélectionnez le nombre de produits tirés aléatoirement dans l'échantillon de test: ",
                            min_value=10,
                            max_value=df.shape[0],
                            value=nb_products,
                            step=1) 
    st.write(f"Nombre de produits sélectionnés : {nb_products}")

    select_train = df.sample(n=nb_products)

    if st.button("Lancer les prédictions"):
        select_train = select_train.fillna("")
        token = create_jwt_token("test_user")
        API_URL = f"http://{IP_ADDRESS}/predict_product_batch"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "designations": select_train["designation"].tolist(),
            "descriptions": select_train["description"].tolist(),
            "image_filepaths": [tools.get_filepath_train(row["productid"], row["imageid"]) for _, row in select_train.iterrows()]
        }
        y_true = select_train["prdtypecode"].tolist()
        with st.spinner("⏳ Envoi à l’API..."):
            response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            y_pred = list(response.json()["predicted prdtypecodes"])
#            st.write("Vérité terrain :")
#            st.success(f"🎯 Prédictions : {y_pred}")
            cr_dict_ref = classification_report(y_true, y_pred, output_dict=True)
            df_fusion = pd.DataFrame.from_dict(cr_dict_ref).T.iloc[:-3, :]
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
            st.markdown("La ligne orange indique le F1-score global pondéré.")

        else:
            st.error(f"🚨 Erreur API : {response.text}")


def batchdata_page(X_train, y_train, df_cat, IP_ADDRESS):
    st.header("📊 Prédiction en batch (texte + image)")
    st.caption("Cette page illustre la prédiction des catégories de plusieurs produits en une seule requête.")

    # Données
    X_test, y_test = get_test_data()
    df = pd.merge(left=X_train, right=y_train, how="left", left_index=True, right_index=True)
    df = df.loc[df.index.isin(X_test.index)]
    df_cat["title"] = df_cat.index.astype(str) + " - " + df_cat["Catégorie"]

    # Sélection
    st.subheader("🎲 Sélection du batch")
    if "nb_products" not in st.session_state:
        st.session_state.nb_products = 100
    nb_products = st.slider(
        "Nombre de produits tirés aléatoirement dans l'échantillon de test :",
        min_value=10,
        max_value=df.shape[0],
        value=st.session_state.nb_products,
        step=1,
    )
    st.caption(f"➡️ {nb_products} produits seront analysés.")

    select_train = df.sample(n=nb_products)

    if st.button("🚀 Lancer les prédictions"):
        select_train = select_train.fillna("")
        token = create_jwt_token("test_user")
        API_URL = f"http://{IP_ADDRESS}/predict_product_batch"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "designations": select_train["designation"].tolist(),
            "descriptions": select_train["description"].tolist(),
            "image_filepaths": [
                tools.get_filepath_train(row["productid"], row["imageid"]) 
                for _, row in select_train.iterrows()
            ]
        }
        y_true = select_train["prdtypecode"].tolist()

        with st.spinner("⏳ Envoi à l’API..."):
            response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            y_pred = list(response.json()["predicted prdtypecodes"])
            
            # Classification report
            cr_dict_ref = classification_report(y_true, y_pred, output_dict=True)
            df_fusion = pd.DataFrame.from_dict(cr_dict_ref).T.iloc[:-3, :]
            df_fusion = df_fusion.drop(columns=['precision', 'recall', 'support'], axis=1)
            df_fusion['PrdTypeCode'] = df_fusion.index.astype(int)
            df_fusion = df_fusion.sort_values(by="f1-score", ascending=False)
            df_fusion['F1-Score'] = df_fusion['f1-score']*100
            df_fusion['F1-Score'] = df_fusion['F1-Score'].apply(lambda x: f"{x:.2f} %")
            df_fusion = pd.merge(df_fusion, df_cat, left_on="PrdTypeCode", right_on="prdtypecode", how='left')

            st.subheader("📈 Résultats")

            # Scores globaux
            global_f1 = cr_dict_ref["weighted avg"]["f1-score"] * 100
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📦 Produits testés", nb_products)
            with col2:
                st.metric("🎯 F1-score global", f"{global_f1:.2f} %")

            # Graphique
            chart_f1_details_fusion = alt.Chart(df_fusion).mark_bar().encode(
                x=alt.X('PrdTypeCode:N', sort=list(df_fusion['f1-score'].values)),
                y='f1-score',
                tooltip=['PrdTypeCode', 'Catégorie', 'F1-Score'],
                color=alt.Color('f1-score', scale=alt.Scale(scheme='viridis', reverse=True))
            ).properties(
                width=700, height=400, title="Scores F1 par catégorie"
            ).interactive()

            chart_f1_global_fusion = alt.Chart(
                pd.DataFrame({'f1-score': [cr_dict_ref["weighted avg"]["f1-score"]]})
            ).mark_rule(color='orange', size=2).encode(y='f1-score')

            chart_comb_fusion = chart_f1_details_fusion + chart_f1_global_fusion
            st.altair_chart(chart_comb_fusion, use_container_width=True)

            st.caption("La ligne orange indique le **F1-score global pondéré**.")
            st.success("✅ Analyse batch terminée avec succès !")

        else:
            st.error(f"🚨 Erreur API : {response.text}")