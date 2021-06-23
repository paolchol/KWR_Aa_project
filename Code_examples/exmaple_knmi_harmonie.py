# -*- coding: utf-8 -*-
"""
An example to load and process KNMI-Harnomoie model

April, 2021
tianx
kwr
"""

import extract_weather_forecast as ewf
# import numpy as np
# import pandas as pd
import api_key_tx
from pathlib import Path
import logging 
import sys
from datetime import datetime

# Initializatoin: set up loggers
logging.basicConfig(filename='knmi.log',  
                    encoding='utf-8', 
                    filemode='a',
                    # stream=sys.stdout, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Step 1: load api key. A key can query ~500 times per day. Here Load your own key
key = api_key_tx.key_tianx

# Step2: Define an url and model name/version for accessing the database
API_URL = "https://api.dataplatform.knmi.nl/open-data"
DATASET_NAME = "harmonie_arome_cy40_p1" # NL model 300*300, see detail: https://www.knmidata.nl/data-services/knmi-producten-overzicht/atmosfeer-modeldata/data-product-1
DATASET_VERSION = "0.2"  

# Step3: Sepcify where data will be downloaded
DATA_DIR = Path.cwd() /'KNMIdata' # where the converted grib file will be stored
TMP_DIR = DATA_DIR / 'tmp_grib' # where temporary grib files will be stored. !!!All temp files will be removed before downloading. 

# Step4: Sepcify date and time 
# date = '20990101' # just use yyyymmdd 
# hour = '00' # options are 00, 06, 12, 18 !
file_names = ewf.file_list(API_URL, DATASET_NAME, DATASET_VERSION, key) # list all files 
file_latest = file_names[-1]
date = file_latest['filename'][-14:-6]
hour = file_latest['filename'][-6:-4]
logging.info(f"The latest prediction is {file_latest['filename']}, modified at {file_latest['lastModified']} by KNMI.")

# Step5: Sepcify a location, either a point or a polygon 
point = {'lat0':51.4366, 'lon0':5.4803} #Eindhoven
# polygon = [ # Example: lon goes first, followed by lat
#     [4.297280, 52.122645],
#     [5.323, 51.5363],
#     [5.24, 51.44927],
#     [5.334, 51.3684],
#     [5.5340, 51.372],
#     [5.6311, 51.4432],      
#     [2, 52],
# ]
# polygon = ewf.load_polygon('eindhove.xlsx') # Eindhoven's poly have been created and saved in 'eindhove.xlsx'

# Step6: download the grib file for the given date and time. Skip downloading if converted grb file exists'.
filename = f"harmonie_xy_{datetime.strptime(date+hour, '%Y%m%d%H').strftime('%Y-%m-%d_%H')}.grb"

if Path(DATA_DIR/filename).exists():
    logging.info(f'{filename} exists. We will skip downloading and converting it.')
else:
    logging.info(f'{filename} does not exists. Start downloading it.')
    ewf.cron(date, hour, 
             API_URL, DATASET_NAME, DATASET_VERSION, 
             key, 
             temdir=TMP_DIR, desdir=DATA_DIR)
    # Step 7: Convert the downloaded grib if it does not exist
    ewf.convert(temdir=TMP_DIR, desdir=DATA_DIR)

# Step 8: Read data from converted grib file: only supportin point now; can add polygon later
df48 = ewf.read_grib_file(DATA_DIR, str(DATA_DIR/filename), point, export_to_excel = True)

