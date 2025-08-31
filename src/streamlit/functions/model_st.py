import numpy as np
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import altair as alt
from functions.tools_st import convert_to_str, show_zoomable_image, get_base64
import pickle
import src.tools.tools as tools
from pathlib import Path




@st.cache_data
def get_camconv_struct():
    img_struct = plt.imread(Path(tools.DATA_VIZ_DIR+"/models/camconv.png"))
    return img_struct

url_hugging = "https://huggingface.co/docs/transformers/index"
url_camembert = "https://huggingface.co/docs/transformers/model_doc/camembert"
url_tensor = "https://www.tensorflow.org/"
url_keras = "https://keras.io/api/applications/"
url_conv = "https://www.tensorflow.org/api_docs/python/tf/keras/applications/ConvNeXtLarge"
url_effB1 = "https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB1"

def best_model(x_train, df_cat):
    # URLs


    # === HEADER ===
    st.header("✨ Meilleur Modèle")

    st.markdown("""
    Dans cette section, nous présentons le **meilleur modèle** obtenu pour le challenge Rakuten.  
    Ce modèle est le fruit du travail de **Olivier ISNARD, Julien TREVISAN et Bruno RAMAYE**  
    dans le cadre de leur formation **Data Scientist**.
    """)

    st.success(
        "🏆 Ce modèle a permis d'atteindre la [**première place du classement**](https://challengedata.ens.fr/participants/challenges/35/ranking/public) du challenge."
    )

    st.subheader("🧩 Architecture du modèle")
    st.markdown("""
    Le modèle retenu est basé sur :
    - **Texte** : [CamemBERT](%s) (via [Transformers](%s))  
    - **Image** : [ConvNeXtLarge](%s) (via [Keras](%s) de [TensorFlow](%s))  
    """ % (url_camembert, url_hugging, url_conv, url_keras, url_tensor))

    # === PREPROCESSING ===
    with st.expander("🧹 Préprocessing des données", expanded=False):
        st.markdown("""
        Le préprocessing retenu et donnant les meilleures performances de F1-score est le suivant :
        - **Texte** : remplacement des NaNs, suppression HTML avec [BeautifulSoup](https://pypi.org/project/beautifulsoup4/),  
          concaténation désignation + description.  
        - **Images** : taille conservée à **500x500** + `tf.keras.applications.convnext.preprocess_input`.
        """)

    with st.expander("🔑 Tokenisation des données textuelles", expanded=False):
        st.markdown("""
        - Utilisation du tokenizer CamemBERT pré-entraîné [**camembert-base**](https://huggingface.co/almanach/camembert-base)  
        - Différentes valeurs testées pour *MAX_LEN* : **128, 256, 512**  
        - Meilleurs résultats obtenus avec **128 et 256 tokens** par produit.
        """)

    # === STRUCTURE ===
    with st.expander("🏗️ Structure du modèle fusionné", expanded=True):
        st.markdown("""
        - **CamemBERT** : sortie de `TFCamembertModel` → Reshape → Dense(768, tanh)  
        - **ConvNeXtLarge** : GlobalAveragePooling2D → Dense(768, relu) (30 dernières couches entraînables)  
        - **Fusion** : Concatenate → Dense(1536, relu) → Dense(512, relu) → Dense(27, softmax)  
        """)
        show_zoomable_image(
            Path(tools.DATA_VIZ_DIR + "/models/camconv.png"),
            caption="Structure du modèle fusionné CamemBERT + ConvNeXtLarge"
        )

    # === PARAMETRES ===
    with st.expander("📊 Nombre de paramètres selon MAX_LEN"):
        df_summary = pd.read_csv(Path(tools.DATA_VIZ_DIR + "/models/model_summary2.csv"), sep=";")
        df_summary_melt = df_summary.melt(id_vars="MAX LEN", var_name="Type", value_name="Nombre")
        df_summary_melt["MAX LEN"] = df_summary_melt["MAX LEN"].astype(str)
        df_summary_melt["Nombre"] /= 1e6

        chart_param = (
            alt.Chart(df_summary_melt)
            .mark_bar(size=30)
            .encode(
                y=alt.Y("MAX LEN:N", title="Taille max des tokens"),
                x=alt.X("Nombre:Q", title="Nombre de paramètres (Millions)"),
                color="Type",
                tooltip=["MAX LEN", "Type", "Nombre"]
            )
            .properties(
                title=alt.TitleParams("📊 Nombre de paramètres selon MAX_LEN", align="center"),
                width=600, height=350
            )
        )
        st.altair_chart(chart_param, use_container_width=True)

    # === ENTRAINEMENT ===
    with st.expander("⚙️ Entraînement du modèle"):
        st.markdown("""
        - Split **72.25% / 12.75% / 15%** (train / valid / test) avec `train_test_split` (*stratify*)  
        - Fonction de perte : **SparseCategoricalCrossentropy**  
        - Optimiseur : **AdamW** (*amsgrad=True*, learning rate = 2e-5)
        """)

    # === F1 SCORE ===
    st.subheader("📈 F1-Score pondéré")
    st.markdown("""
    - Résultat global après entraînement : **≈ 91.8%**  
    - Détails par type de produit :
    """)

    filename = tools.DATA_VIZ_DIR + "/models/cr_fusion_cam_conv-1-0bis.pkl"
    with open(filename, "rb") as f:
        cr_dict_ref = pickle.load(f)

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

    # === CLASSEMENT ===
    st.subheader("🏆 Classement Rakuten")
    st.write("Extrait du classement public du challenge Rakuten (28 Février 2025) :")
    fig = plt.figure()
    img_ranking = plt.imread(tools.DATA_VIZ_DIR + "/models/classement_rakuten.png")
    plt.imshow(img_ranking)
    plt.axis("off")
    plt.title("Classement public (28 Février 2025)", fontsize=8)
    st.pyplot(fig)

    

def retained_models(x_train, df_cat):

    st.header("🧩 Modèles retenus")

    st.markdown("""
    Dans cette section, nous présentons les modèles retenus pour le projet **MLOps**.  
    L'objectif n'est pas d’améliorer à tout prix le meilleur score du challenge Rakuten,  
    mais de mettre en place des modèles **adaptés à un cycle de vie industriel**.
    """)

    st.warning("""
    ⚙️ **Contraintes de mise en production** :  
    - L’exploitation de modèles lourds comme **CamemBERT** nécessite une **GPU**.  
    - Le projet utilise une **EC2 AWS** (ec2-g4dn-xlarge), ce qui engendre des **coûts d’usage**.  
    - Les modèles retenus sont volontairement **dimensionnés plus modestement**  
      afin de limiter les coûts AWS.
    """)

    st.markdown("### ✅ Modèles")

    logo_hf = get_base64("./data/dataviz/assets/logo_hf.png")
    logo_tf = get_base64("./data/dataviz/assets/logo_tf.png")
    logo_fusion = get_base64("./data/dataviz/assets/logo_fusion.png")

    # CSS pour centrer et ajouter un effet hover
    st.markdown("""
    <style>
    .model-card {
        text-align: center;
        padding: 15px;
        border-radius: 12px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .model-card:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        background-color: #f9f9f9;
    }
    .model-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        height: 80px;       /* Hauteur uniforme */
        width: auto;        /* Conserve le ratio */
        object-fit: contain; /* S'assure que l'image rentre bien */                
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="model-card">
                <img src="data:image/png;base64,{logo_hf}" class="model-logo" width="80">
                <p><b>Modèle Textuel</b><br>
                Basé sur <a href="{url_camembert}">CamemBERT</a> via <a href="{url_hugging}">Transformers</a><br>
                Nombre de tokens limité à <b>16</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="model-card">
                <img src="data:image/png;base64,{logo_tf}" class="model-logo" width="80">
                <p><b>Modèle Image</b><br>
                <b>EfficientNetB1</b> via <a href="{url_keras}">Keras</a> et <a href="{url_tensor}">TensorFlow</a><br>
                Bon compromis <i>perf / coût</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="model-card">
                <img src="data:image/png;base64,{logo_fusion}" class="model-logo" width="80">
                <p><b>Modèle Fusionné</b><br>
                Combinaison du <b>modèle textuel</b> et du <b>modèle image</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "📂 Les paramètres détaillés sont configurés dans [`params.yaml`](https://github.com/oisnard/MAI25_CMLOps_Rakuten/blob/master/params.yaml) et versionnés dans le repo GitHub."
    )

    #st.write("Mettre en production 3 types de modèles (textuels, images, fusionnés) permet d'offrir une API permettant de prédire la catégorie d'un produit en fonction des données fournies (textuelles seulement, image seulement, textuelles et images).")
    st.markdown("""
    <style>
    .info-card {
        border: 1px solid #29A989;
        border-radius: 12px;
        padding: 15px;
        margin: 20px 0;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 4px 12px rgba(0,0,0,0.2);
        background-color: #fdfefe;
    }
    </style>
    """, unsafe_allow_html=True)

    # Paragraphe reformulé avec effet hover
    st.markdown("""
    <div class="info-card">
        🚀 La mise en production de trois modèles (<b>textuel</b>, <b>image</b>, et <b>fusionné</b>) 
        permet de proposer une <b>API flexible</b> capable de prédire la catégorie d’un produit 
        selon les données disponibles :  
        <ul>
            <li>📝 uniquement <b>textuelles</b></li>
            <li>🖼️ uniquement <b>visuelles</b></li>
            <li>🔗 une <b>combinaison des deux</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Optimisation")

    st.markdown("""
    - **Optimiseur** : [`Adam`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)  
    (`tf.keras.optimizers.Adam`)  
    - **Fonction de perte** : [`SparseCategoricalCrossentropy`](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy)  
    (`tf.keras.losses.SparseCategoricalCrossentropy()`)  
    """)

    st.markdown("### 🚀 Performances")

    st.markdown("""
    Pour la suite du projet, il est possible d'améliorer les performances de prédiction au besoin :  
    - en **Réajustant** le paramètre `MAX_LEN` à **128** ou **256** tokens pour le modèle textuel  
    - en remplaçant le modèle **EfficientNetB1** par un modèle **ConvNeXtLarge**  
    """)


