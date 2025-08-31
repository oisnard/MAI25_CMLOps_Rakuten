import streamlit as st
from pathlib import Path
import base64

def img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def portail_show(IP_ADDRESS: str):

    st.title("Portail MLOps")

    st.markdown("""
    Ce portail regroupe l’ensemble des outils utilisés pour assurer le suivi et le bon fonctionnement du projet MLOps :

    - **MLflow** : suivi des expériences, gestion des modèles et des métriques  
    - **Evidently** : détection de dérive des données et des modèles  
    - **Airflow** : orchestration et planification des pipelines  
    - **Prometheus** : collecte de métriques de performance  
    - **Grafana** : visualisation et monitoring en temps réel  
    - **Swagger API** : documentation interactive de l’API d’inférence  

    Cliquez sur les logos ci-dessous pour accéder directement aux consoles du projet.
    """)

    # Chemin du dossier assets
    LOGO_DIR = Path("./data/dataviz/assets")

    # Dictionnaire des outils avec leurs URLs et logos

    tools = {
    "MLflow": {"url": f"http://{IP_ADDRESS}:5000", "logo": LOGO_DIR / "logo_mlflow.png"},
    "Evidently": {"url": f"http://{IP_ADDRESS}:9000", "logo": LOGO_DIR / "logo_evidently.png"},
    "Airflow": {"url": f"http://{IP_ADDRESS}:8080", "logo": LOGO_DIR / "logo_airflow.png"},
    "Prometheus": {"url": f"http://{IP_ADDRESS}:30900", "logo": LOGO_DIR / "logo_prometheus.png"},
    "Grafana": {"url": f"http://{IP_ADDRESS}:30300", "logo": LOGO_DIR / "logo_grafana.png"},
    "Swagger API": {"url": f"http://{IP_ADDRESS}/docs", "logo": LOGO_DIR / "logo_swagger.png"},
    }

    # injection CSS pour hover
    st.markdown("""
    <style>
    .card {
        border: 1px solid #29A989;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background-color: #f9f9f9;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 4px 4px 12px rgba(0,0,0,0.2);
    }
    .card img {
        height: 70px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns(3)

    for i, (tool, data) in enumerate(tools.items()):
        with cols[i % 3]:
            img_b64 = img_to_base64(data["logo"])
            st.markdown(
                f"""
                <a href="{data['url']}" target="_blank" style="text-decoration: none;">
                    <div class="card">
                        <img src="data:image/png;base64,{img_b64}" alt="{tool} logo"/>
                    </div>
                </a>
                """,
                unsafe_allow_html=True
            )