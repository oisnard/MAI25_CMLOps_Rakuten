from dotenv import find_dotenv, load_dotenv
import logging 
import os
import requests 
from bs4 import BeautifulSoup   
from pathlib import Path
from tqdm import tqdm
from zipfile import ZipFile
import src.tools.tools as tools

# Constants for the ENS data challenge
url_login = "https://challengedata.ens.fr/login/"
endpoint_download_xtrain = "/participants/challenges/35/download/x-train"
endpoint_download_xtest = "/participants/challenges/35/download/x-test"
endpoint_download_ytrain = "/participants/challenges/35/download/y-train"
endpoint_download_images = "https://challengedata.ens.fr/participants/challenges/35/download/supplementary-files"

# Load csv file from challengedata.ens.fr using login and password
def load_csv_file(login, password, endpoint, target_folder, target_name):
    """Load csv data from a remote source using login and password."""
    logging.info(f"Loading {target_name} into {target_folder} with login {login}")

    session = requests.Session()

    # First, get CSRF token
    try:
        page_login = session.get(url_login)
        soup = BeautifulSoup(page_login.text, 'html.parser')
        csrf_token = soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]

        # Prepare data pour POST request
        login_data  = {
            "csrfmiddlewaretoken": csrf_token,
            "username": login,
            "password": password,
            "next": endpoint
        }

        headers = {
            "Referer": url_login
        }
        reponse_login = session.post(url_login, data=login_data, headers=headers)
    except Exception as e:
        logging.error(f"Error during login: {e}")
        exit(1)
    # Vérifier si la connexion a réussi (ex: redirection ou présence d'une session)
    #print(reponse_login.text)
    if reponse_login.status_code == 200:
        logging.info("Successfull Connexion to ens data.")
    else:
        logging.error("Failed to connect to ens data.")
        exit()

    # Download x_train_file
    filename = os.path.join(target_folder, target_name)
    with open(filename, "wb") as f:
        f.write(reponse_login.content)

# Load images from challengedata.ens.fr using login and password
def download_images_data(login, password, endpoint, target_folder, target_name):
    """Download images data from a remote source using login and password."""
    logging.info(f"Downloading images data into {target_folder} with login {login}")

    session = requests.Session()

    try:
        # First, get CSRF token
        page_login = session.get(url_login)
        soup = BeautifulSoup(page_login.text, 'html.parser')
        csrf_token = soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]

        # Prepare data for POST request
        login_data  = {
            "csrfmiddlewaretoken": csrf_token,
            "username": login,
            "password": password
        }

        headers = {
            "Referer": url_login
        }
        reponse_login = session.post(url_login, data=login_data, headers=headers)

        # Vérifier si la connexion a réussi (ex: redirection ou présence d'une session)
        if reponse_login.status_code == 200:
            logging.info("Successful Connexion to ens data")
        else:
            logging.error(f"Failed to connect to ens data. status code: {reponse_login.status_code}")
            return

        # Downloading images data
        response_download = session.get(endpoint, stream=True)
        filename = os.path.join(target_folder, target_name)
        with open(filename, "wb") as f:
            f.write(reponse_login.content)
        if response_download.status_code == 200:
            # Taille totale du fichier (si fournie)
            taille_totale = int(response_download.headers.get('Content-Length', 0))
            # Si la taille totale n'est pas fournie, on peut l'estimer à partir de la longueur du contenu
            if taille_totale == 0:
                taille_totale = len(response_download.content)
                logging.warning("Taille totale du fichier non fournie, estimation basée sur le contenu.")
            # Si la taille totale est connue, on peut afficher une barre de progression
            logging.info(f"Downloading {target_name} to {filename}")
            # Barre de progression
            with open(filename, 'wb') as f, tqdm(
                total=taille_totale, unit='B', unit_scale=True, desc=filename, ncols=80
            ) as pbar:
                for chunk in response_download.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            logging.info(f"File {target_name} downloaded successfully to {filename}.")
        else:
            logging.error(f"Error during download: {response_download.status_code}")
    except Exception as e:
        logging.error(f"Error during login or download: {e}")
        exit(1)


def extract_zip_file(zip_file_path, extract_to, overwrite=True):
    """Extract a zip file to a target folder."""
    logging.info(f"Extracting {zip_file_path} to {extract_to}")
    with ZipFile(zip_file_path, 'r') as zip_ref:
        # Lister les fichiers à extraire (filtrés et triés)
        members = [
            m for m in zip_ref.namelist()
            if m.startswith("images/image_train/") or m.startswith("images/image_test/")
        ]

        # Barre de progression
        with tqdm(total=len(members), desc="Décompression", ncols=80) as pbar:
            for member in members:
                relative_path = os.path.relpath(member, "images")
                output_path = os.path.join(extract_to, relative_path)

                # Crée le dossier cible s’il n’existe pas
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Si c’est un fichier (pas un dossier)
                if not member.endswith('/'):
                    if overwrite or not os.path.exists(output_path):
                        with zip_ref.open(member) as source, open(output_path, "wb") as target:
                            target.write(source.read())
                    # Sinon : ignorer le fichier existant
                pbar.update(1)

    logging.info(f"Extraction completed to {extract_to}")


def main(raw_data_relative_path = tools.DATA_RAW_DIR,
        filenames=[tools.X_TRAIN_RAW_FILENAME, tools.X_TEST_RAW_FILENAME, tools.Y_TRAIN_RAW_FILENAME],):


    list_endpoints = [endpoint_download_xtrain, endpoint_download_xtest, endpoint_download_ytrain]

    ensdata_login = tools.ENSDATA_LOGIN         #os.getenv("ENSDATA_LOGIN")
    ensdata_password = tools.ENSDATA_PASSWORD   #os.getenv("ENSDATA_PASSWORD")

    if len(ensdata_login) == 0:
        logging.ERROR('ENSDATA_LOGIN is not properly set in the .env file')
        exit(1)
    if len(ensdata_password) == 0:
        logging.ERROR('ENSDATA_PASSWORD is not properly set in the .env file')
        exit(1)

    # Check if the raw data directory exists, if not create it
    if not os.path.isdir(raw_data_relative_path):
        logging.info(f"Creating directory: {raw_data_relative_path}")
        os.makedirs(raw_data_relative_path)
    
    # Downloading (if needed) the raw csv data files
    for filename, endpoint in zip(filenames, list_endpoints):
        print(f"Filename: {filename}, Endpoint: {endpoint}")

        # Download the X_train file if it does not already exist
        if Path(os.path.join(raw_data_relative_path, filename)).exists():
            logging.info(f"File {filename} already exists in {raw_data_relative_path}. Skipping download.")
        else:
            logging.info(f"Downloading {filename} to {raw_data_relative_path}")
            load_csv_file(login=ensdata_login, 
                        password=ensdata_password, 
                        endpoint=endpoint,
                        target_folder=raw_data_relative_path,
                        target_name=filename)
            logging.info('Download completed.')

    # Downloading the images data 
    target_images_zip = "images.zip"
    target_images_train_folder = tools.DATA_RAW_IMAGES_TRAIN_DIR    #os.path.join(raw_data_relative_path, "image_train")
    target_images_test_folder = tools.DATA_RAW_IMAGES_TEST_DIR      #os.path.join(raw_data_relative_path, "image_test")
    download_required = False
    process_zip = False
    # Check if the images train data folder already exists
    if Path(target_images_train_folder).exists() and Path(target_images_test_folder).is_dir():
        logging.info(f"Folder {target_images_train_folder} already exists.")
    else:
        download_required = True
    # Check if the images test data folder already exists
    if Path(target_images_test_folder).exists() and Path(target_images_test_folder).is_dir():
        logging.info(f"Folder {target_images_test_folder} already exists.")
    else:
        download_required = True

    # If download is required, Check if the images zip file already exists
    if download_required:
        if Path(os.path.join(raw_data_relative_path, target_images_zip)).exists():
            logging.info(f"File {target_images_zip} already exists in {raw_data_relative_path}. Skipping download.")
        else:     
            download_images_data(
                login=ensdata_login,
                password=ensdata_password,
                endpoint=endpoint_download_images,
                target_folder=raw_data_relative_path,
                target_name="images.zip"
            )
        extract_zip_file(
            zip_file_path=os.path.join(raw_data_relative_path, target_images_zip),
            extract_to=raw_data_relative_path,      # image_train et image_test seront créés ici
            overwrite=False                         # ne pas écraser les fichiers existants
        )

    # Remove the zip file after extraction
    if Path(os.path.join(raw_data_relative_path, target_images_zip)).exists():
        logging.info(f"Removing zip file {target_images_zip} from {raw_data_relative_path}")
        os.remove(os.path.join(raw_data_relative_path, target_images_zip))
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

