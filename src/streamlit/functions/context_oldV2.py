import streamlit as st
from src.streamlit.functions.tools_st import get_base64
import base64



def context_show():
    # Conversion des logos en base64
    rakuten_logo = get_base64("./data/dataviz/assets/logo_rakuten_small.png")
    ds_logo = get_base64("./data/dataviz/assets/logo_datascientest_small.png")

    # === Titre principal avec logos alignés ===
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;">
        <img src="data:image/png;base64,{rakuten_logo}" style="width:100px;">
        <h1 style="margin:0; text-align:center; flex-grow:1;">Projet Rakuten</h1>
        <img src="data:image/png;base64,{ds_logo}" style="width:100px;">
    </div>
    """, unsafe_allow_html=True)

    # === CSS global pour encadrés ===
    st.markdown("""
    <style>
    .section-card {
        border: 1px solid #29A989;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # === Section Auteurs ===
    st.header("Auteurs")
    st.markdown("""
    <div class="section-card">
        <ul>
            <li>Olivier ISNARD</li>
            <li>Christian SEGNOU</li>
        </ul>
        <p>Projet mené dans le cadre de la formation MLOps de DataScientest, avec l’accompagnement de Maria.</p>
    </div>
    """, unsafe_allow_html=True)

    # === Section Contexte du projet ===
    st.header("Contexte du projet")
    st.markdown("""
    <div class="section-card">
        <p>
        Cette application présente les résultats du projet réalisé dans le cadre du challenge 
        <a href="https://challengedata.ens.fr/challenges/35" target="_blank">Rakuten France Multimodal Product Data Classification</a>.
        </p>
        <p>
        Rakuten est un site de e-commerce spécialisé dans la distribution de produits neufs, d’occasion et reconditionnés. 
        Le site recense plus de 100 millions d’utilisateurs et des dizaines de milliers de commerçants. 
        </p>
        <p>
        Cataloguer efficacement les produits (à partir de textes et d’images) est essentiel 
        pour améliorer la recherche, la recommandation et la personnalisation dans l’e-commerce.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === Section Objectifs du challenge ===
    st.header("Objectifs du challenge")
    st.markdown("""
    <div class="section-card">
        <p>
        L’objectif principal est de prédire la <b>catégorie d’un produit</b> à partir de ses 
        désignations textuelles et de ses images.
        </p>
        <p>
        La métrique officielle du challenge est le <i>Weighted F1-Score</i>. 
        Les modèles de référence obtiennent :
        </p>
        <ul>
            <li>81.13% pour le modèle textuel</li>
            <li>55.34% pour le modèle image</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # === Section Objectifs du projet MLOps ===
    st.header("Objectifs du projet MLOps")
    # Header logos seuls


    # CSS hover pour cartes des phases
    st.markdown("""
    <style>
    .card {
        border: 1px solid #29A989;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 4px 12px rgba(0,0,0,0.2);
    }
    .card h4 {
        color: #29A989;
        margin-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Phases MLOps
    phases = {
        "Phase 1 : Fondations & Conteneurisation": [
            "Définir les objectifs du projet et les métriques clés",
            "Mettre en place un environnement de développement reproductible",
            "Collecter et prétraiter les données",
            "Construire et évaluer un modèle ML de base, implémenter des tests unitaires",
            "Implémenter une API d’inférence basique"
        ],
        "Phase 2 : Microservices, Suivi & Versioning": [
            "Mettre en place le suivi des expériences avec MLflow",
            "Implémenter le versioning des données et des modèles",
            "Décomposer l’application en microservices et concevoir une orchestration simple"
        ],
        "Phase 3 : Orchestration & Déploiement": [
            "Finaliser l’orchestration de bout en bout",
            "Créer un pipeline CI",
            "Optimiser et sécuriser l’API",
            "Implémenter la scalabilité avec Docker/Kubernetes"
        ],
        "Phase 4 : Monitoring & Maintenance": [
            "Mettre en place le monitoring des performances avec Prometheus/Grafana",
            "Implémenter la détection de dérive avec Evidently",
            "Développer des mises à jour automatisées du modèle et des composants",
            "Finaliser la documentation technique"
        ]
    }

    # Affichage des phases en 2 colonnes
    cols = st.columns(2)
    for i, (title, tasks) in enumerate(phases.items()):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="card">
                    <h4>{title}</h4>
                    <ul>
                        {''.join([f"<li>{task}</li>" for task in tasks])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
