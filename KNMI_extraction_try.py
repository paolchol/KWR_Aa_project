# -*- coding: utf-8 -*-
"""
Script to obtain data from KNMI data platform

1. (Grid data) Historic gridded daily precipitation sum (1910 - 2010):
    https://dataplatform.knmi.nl/dataset/rd1h-1

2. (Grid data) Daily precipitation sum (1951 - present):
    https://dataplatform.knmi.nl/dataset/rd1-5

3. (Point data) Duration, amount and intensity at a 10 minute interval (2003 - present):
    https://dataplatform.knmi.nl/dataset/neerslaggegevens-1-0

@author: colompa
"""

import sys
#from datetime import datetime, timedelta
#from pathlib import Path
import requests
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

api_url = "https://api.dataplatform.knmi.nl/open-data"
api_version = "v1"

# The filename from source 2 has this format below:
#   INTER_OPER_R___RD1_____L3__19510101T080000_19510102T080000_0005.nc
# This was obtained by extracting a list of files from the API

def main():
    # Parameters
    api_key = "5e554e19274a9600012a3eb1b626f95624124cf89e9b5d74c3304520"
    dataset_name = "Rd1"
    dataset_version = "5"
    #maxkeys = 20 #The default value is 10, but we want every file after a certain date, how to do this?

    #To get files after a certain date, set the date in time stamp
    # timestamp = "19800101"
    # start_after_filename = f"INTER_OPER_R___RD1_____L3__{timestamp}T080000_{timestamp}T080000_0005.nc"
    # logger.info(f"Beginning date of the extraction: {timestamp}")
    
    endpoint = f"{api_url}/{api_version}/datasets/{dataset_name}/versions/{dataset_version}/files"

    get_file_response = requests.get(endpoint, headers = {"Authorization": api_key})

    # get_file_response = requests.get(endpoint, headers={"Authorization": api_key},
    #                                  params = {"maxKeys": maxkeys, "startAfterFilename": start_after_filename})

    if get_file_response.status_code != 200:
        logger.error("Unable to retrieve download url for file")
        logger.error(get_file_response.text)
        sys.exit(1)
    else:
        logger.info("Successfully retrieved data")
