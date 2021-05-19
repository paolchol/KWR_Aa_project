# -*- coding: utf-8 -*-
"""
Script to obtain data from KNMI data platform

Historic gridded daily precipitation sum (1910 - 2010):
    https://dataplatform.knmi.nl/dataset/rd1h-1

Daily precipitation sum (1951 - present):
    https://dataplatform.knmi.nl/dataset/rd1-5

"""

import requests
precipitation = requests.get('https://api.dataplatform.knmi.nl/open-data/v1/datasets/Rd1nrt/versions/2/files')
print(precipitation.status_code)
#401
#I need the autentification

#Try with the autentification with this tutorial: https://www.digitalocean.com/community/tutorials/how-to-use-web-apis-in-python-3

import requests
import json

#Anonymous key provided by KNMI here: https://developer.dataplatform.knmi.nl/get-started#obtain-an-api-key
api_token = '5e554e19274a9600012a3eb1b626f95624124cf89e9b5d74c3304520'
api_url_base = 'https://api.dataplatform.knmi.nl/open-data/v1/datasets/Rd1/versions/5/files'

headers = {'Content-Type': 'application/json',
           'Authorization': 'Bearer {0}'.format(api_token)}

def get_account_info():

    api_url = '{0}account'.format(api_url_base)

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        return None
    
account_info = get_account_info()

if account_info is not None:
    print("Here's your info: ")
    for k, v in account_info['account'].items():
        print('{0}:{1}'.format(k, v))

else:
    print('[!] Request Failed')

#The request fails again

#Third try: arrange the code available directly from KNMI

#The code below is used for listing the first 10 files of today and retrieving
#the first one

import logging
import sys
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

api_url = "https://api.dataplatform.knmi.nl/open-data"
api_version = "v1"

#https://api.dataplatform.knmi.nl/open-data/v1/datasets/Rd1/versions/5/files

def main():
    # Parameters
    api_key = "5e554e19274a9600012a3eb1b626f95624124cf89e9b5d74c3304520"
    dataset_name = "Rd1"
    dataset_version = "5"
    max_keys = "10"

    # Use list files request to request first 10 files of the day.
    timestamp = "20210512" #datetime.utcnow().date().strftime("%Y%m%d")
    start_after_filename_prefix = f"KMDS__OPER_P___10M_OBS_L2_{timestamp}"
    list_files_response = requests.get(
        f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files",
        headers={"Authorization": api_key},
        params={"maxKeys": max_keys, "startAfterFilename": start_after_filename_prefix},
    )
    list_files = list_files_response.json()

    logger.info(f"List files response:\n{list_files}")
    dataset_files = list_files.get("files")

    # # Retrieve first file in the list files response
    # filename = dataset_files[0].get("filename")
    # logger.info(f"Retrieve file with name: {filename}")
    # endpoint = f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
    # get_file_response = requests.get(endpoint, headers={"Authorization": api_key})
    # if get_file_response.status_code != 200:
    #     logger.error("Unable to retrieve download url for file")
    #     logger.error(get_file_response.text)
    #     sys.exit(1)

    # download_url = get_file_response.json().get("temporaryDownloadUrl")
    # dataset_file_response = requests.get(download_url)
    # if dataset_file_response.status_code != 200:
    #     logger.error("Unable to download file using download URL")
    #     logger.error(dataset_file_response.text)
    #     sys.exit(1)

    # # # Write dataset file to disk
    # # p = Path(filename)
    # # p.write_bytes(dataset_file_response.content)
    # # logger.info(f"Successfully downloaded dataset file to {p}")


if __name__ == "__main__":
    main()

#riprovare in un altro modo
#alle 16 iniziare invece a usare i file nelle cartelle

