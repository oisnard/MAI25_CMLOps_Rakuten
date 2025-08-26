import os
import subprocess
import requests
import time
import random
from src.api.middleware import create_jwt_token  # Assure-toi que le chemin est correct

import src.tools.tools as tools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔐 Token generation
token = create_jwt_token("test_user")

# 🔁 Retrieve IP Ingress automatically through kubectl
#def get_ingress_ip():
#    try:
#        result = subprocess.run(
#            ["kubectl", "get", "ingress", "rakuten-api", "-n", "apps", "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"],
#            capture_output=True, text=True, check=True
#        )
#        ip = result.stdout.strip()
#        if ip:
#            return f"http://{ip}"
#        else:
#            raise RuntimeError("❌ Aucune IP d'Ingress disponible.")
#    except subprocess.CalledProcessError as e:
#        raise RuntimeError(f"❌ Erreur récupération IP Ingress : {e}")
# 🌐 Récupérer l'adresse d'Ingress via une variable d'environnement
INGRESS_IP = os.getenv("INGRESS_IP")
if not INGRESS_IP:
    raise RuntimeError("❌ Variable d’environnement INGRESS_IP manquante.")

logging.info(f"📡 L’IP d’Ingress utilisée est : {INGRESS_IP}")

API_URL = f"{INGRESS_IP}/predict_product_batch"
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# 🧪 Exemples de produits
PRODUCTS = [
    {"designation": "Montre connectée GPS", "description": "Montre étanche avec écran AMOLED"},
    {"designation": "Aspirateur robot intelligent", "description": "Navigation laser et nettoyage automatique"},
    {"designation": "Ordinateur portable gaming", "description": "Intel i7, RTX 4060, 16Go RAM"},
    {"designation": "Casque audio Bluetooth", "description": "Réduction active du bruit"},
    {"designation": "Liseuse numérique", "description": "Écran e-ink 300 ppi, étanche"}
]

# 🚀 Boucle de génération de trafic
def send_traffic(n=100, delay=0.5):
    print(f"📡 Envoi de {n} requêtes vers {API_URL}")
    success = 0
    for i in range(n):
        product = random.choice(PRODUCTS)
        response = requests.post(API_URL, json=product, headers=HEADERS)
        if response.status_code == 200:
            print(f"[{i+1}/{n}] ✅ {response.json()}")
            success += 1
        else:
            print(f"[{i+1}/{n}] ❌ Erreur {response.status_code} : {response.text}")
        time.sleep(delay)
    print(f"\n🎯 {success}/{n} requêtes réussies.")

if __name__ == "__main__":
        # Load parameters from YAML file
    try:
        # Load dataset parameters from YAML file
        params = tools.load_dataset_params_from_yaml()    
    except Exception as e:
        logger.error(f"An unexpected error occurred when loading params.yaml file: {e}")
        raise

    # Load the model type from params
    MODEL_TYPE = params['model_selection']['model_type']
    if MODEL_TYPE not in ['text', 'image', 'merged']:
        logger.error(f"Invalid model type: {MODEL_TYPE}. It should be one of ['text', 'image', 'merged'].")
        raise ValueError(f"Invalid model type: {MODEL_TYPE}. It should be one of ['text', 'image', 'merged'].")

    logger.info(f"Model type selected: {MODEL_TYPE}")

    X_test = tools.load_xtest_raw_data()
    logger.info("X_test dataset loaded successfully.")

    BATCH_SIZE = 2000

    if MODEL_TYPE == 'text':
        X_test_clean = X_test.iloc[:BATCH_SIZE].fillna("")
        payload = {
            "designations": X_test_clean['designation'].tolist(),
            "descriptions": X_test_clean['description'].tolist()
        }


    response = requests.post(API_URL, json=payload, headers=HEADERS)
    if response.status_code == 200:
        print(f"✅ {response.json()}")
    else:
        print(f"❌ Erreur {response.status_code} : {response.text}")

 