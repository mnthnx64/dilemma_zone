# Download a video from a google drive link
# Usage: get_video.py

import requests
import os

def get_file_id(link):
    """Get file id from google drive link"""
    file_id = link.split('/')[-1]
    return file_id
    

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


link = 'https://drive.google.com/file/d/1gp6DaVoWakeVLpSVdeOH1RDO8zsd_yfd/view?usp=sharing'
os.mkdir('data/videos')
file_id = get_file_id(link)
destination = '/Users/mx98/workspace/dilemma_zone/data/videos/D0_0_0.mp4'
download_file_from_google_drive(file_id, destination)