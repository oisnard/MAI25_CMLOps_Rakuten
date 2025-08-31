# Library to get IP adress
import requests
import socket

# Fonction utilitaire pour vérifier si on est sur une instance EC2
def is_ec2_instance(timeout=0.1):
    """Vérifie si on est sur une instance EC2 (IMDSv2)"""
    try:
        response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=timeout
        )
        return response.status_code == 200
    except requests.RequestException:
        return False

# Fonction utilitaire pour récupérer l'IP publique d'une instance EC2
def get_public_ip_ec2(timeout=0.5):
    """Tente de récupérer l'IP publique via IMDSv2"""
    try:
        token_response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=timeout
        )
        token = token_response.text

        ip_response = requests.get(
            "http://169.254.169.254/latest/meta-data/public-ipv4",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=timeout
        )
        public_ip = ip_response.text.strip()
        return public_ip if public_ip else None
    except requests.RequestException:
        return None

# Fonction utilitaire pour récupérer l'IP locale
def get_local_ip():
    """Récupère l'adresse IP locale"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Force la détection de l'interface réseau
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Fonction utilitaire pour récupérer l'IP
def get_ip():
    """Détecte l'environnement et retourne l'IP appropriée"""
    if is_ec2_instance():
        ip = get_public_ip_ec2()
        if ip:
            print("🟢 Instance EC2 détectée. IP publique :", ip)
            return ip
        else:
            print("🟡 EC2 détectée mais IP publique indisponible. Fallback vers IP locale.")
    else:
        print("🔵 Pas une instance EC2. Utilisation de l'IP locale.")
    
    return get_local_ip()


if __name__ == "__main__":
    # Exécution
    ip_address = get_ip()
    print("Adresse IP utilisée :", ip_address)