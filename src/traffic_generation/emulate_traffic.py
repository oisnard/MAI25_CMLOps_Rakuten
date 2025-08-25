import os
import subprocess
import requests
import time
import random
from src.api.middleware import create_jwt_token  # Assure-toi que le chemin est correct

# 🔐 Génération du token
token = create_jwt_token("test_user")

# 🔁 Récupération de l'IP Ingress automatiquement via kubectl
def get_ingress_ip():
    try:
        result = subprocess.run(
            ["kubectl", "get", "ingress", "rakuten-api", "-n", "apps", "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"],
            capture_output=True, text=True, check=True
        )
        ip = result.stdout.strip()
        if ip:
            return f"http://{ip}"
        else:
            raise RuntimeError("❌ Aucune IP d'Ingress disponible.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Erreur récupération IP Ingress : {e}")

API_URL = f"{get_ingress_ip()}/predict_product"
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
    send_traffic(n=50, delay=1)