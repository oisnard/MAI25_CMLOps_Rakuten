import streamlit as st

def context_show():
    st.title("Projet Rakuten")
    st.header("Auteurs")
    st.markdown("""
    - Olivier ISNARD
    - Christian SEGNOU
    """)    
    st.write("Encadrés dans le projet qui s'inscrit dans le cadre de la formation MLOps par Maria de DataScientest.")
    st.header("Contexte du projet")
    url = "https://challengedata.ens.fr/challenges/35"
    st.write("Cette application présente les résultats du projet s'inscrivant dans le cadre du challenge [Rakuten France Multimodal Product Data Classification](%s)." % url)
    st.write("Rakuten est un site de e-commerce spécialisé dans la distribution de produits neufs, d’occasion et reconditionnés. Le site de Rakuten est disponible dans plusieurs pays. La société Rakuten Shopping Mall recense 105 millions d'utilisateurs et 44 201 commerçants (recensement de 2017 - https://fr.wikipedia.org/wiki/Rakuten).")
    st.write("Les sites de e-commerces proposent un vaste catalogue de produits (tech, librairie, électroménager, etc.), contenant un titre, une image, et une description. Chacun d'entre eux est fourni par un vendeur. Ses produits sont classifiés par catégorie/sous-catégorie. Le site de e-commerce Rakuten est disponible dans plusieurs langues.")
    st.write("Cataloguer les produits selon des données différentes (textes et images) est important pour les e-commerces puisque cela permet de réaliser des applications diverses telles que la recommandation de produits et la recherche personnalisée. Il s’agit alors de prédire le code type des produits sachant des données textuelles (désignation et description des produits) ainsi que des données images (image du produit).")
    st.subheader("Objectifs du challenge")
    st.write("Ce projet ambitionne d'instuire le problème de classification des produits sur la base de leurs désignations et descriptions textuelles ainsi que de leur images.")
    st.write("Le challenge définit la métrique *Weighted F1-Score*. Les modèles de référence définis par le challenge obtiennent un F1-Score de 81.13% pour le modèle textuel et 55.34% pour le modèle image.")
    st.subheader("Objectifs du projet MLOps")
    st.markdown("#### Phase 1 : Fondations & Conteneurisation")
    st.markdown("""
    - Définir les objectifs du projet et les métriques clés  
    - Mettre en place un environnement de développement reproductible  
    - Collecter et prétraiter les données  
    - Construire et évaluer un modèle ML de base, implémenter des tests unitaires  
    - Implémenter une API d’inférence basique
    """)

    st.markdown("#### Phase 2 : Microservices, Suivi & Versioning")
    st.markdown("""
    - Mettre en place le suivi des expériences avec MLflow  
    - Implémenter le versioning des données et des modèles  
    - Décomposer l’application en microservices et concevoir une orchestration simple
    """)

    st.markdown("#### Phase 3 : Orchestration & Déploiement")
    st.markdown("""
    - Finaliser l’orchestration de bout en bout  
    - Créer un pipeline CI  
    - Optimiser et sécuriser l’API  
    - Implémenter la scalabilité avec Docker/Kubernetes
    """)

    st.markdown("#### Phase 4 : Monitoring & Maintenance")
    st.markdown("""
    - Mettre en place le monitoring des performances avec Prometheus/Grafana  
    - Implémenter la détection de dérive avec Evidently  
    - Développer des mises à jour automatisées du modèle et des composants  
    - Finaliser la documentation technique
    """)