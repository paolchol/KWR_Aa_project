# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:42:23 2021

To process the 48-hr weather forecast from knmi
https://www.knmidata.nl/data-services/knmi-producten-overzicht/atmosfeer-modeldata/data-product-1


the following pkgs are needed:
xarray
pygrib
(optional:) eccodes

Additional information: 
- What are GRIB files
    GRIB files are the file format used at ECMWF.  
    GRIB is a WMO standard and consists of GRIB Edition 1 and Edition 2.
    The OpenIFS/IFS models output in GRIB format. 
    These files are a mix of GRIB-1 & GRIB-2 messages, 
    the multi-level fields are encoded as GRIB-2, 
    whereas surface fields are GRIB-1. 
    GRIB files can also contain multiple vertical coordinates: 
        pressure levels, model levels, sub-surface levels etc. 
    This can cause a problem with some 3rd party tools, 
        as the same GRIB variable code is used for each axis. 
        
- codes of parameters: 
    Mean sea level pressure: 2
    Relative humidity: 52
    Temperature 2m: 11
    U-wind: 33
    V-wind: 34
    Precipication Intensity: 61              
    Cumulative Precipication: 62
    UWG	U-component max wind gust	: 162	
    VWG	V-component max wind gust: 163
    SWR (starting from 1st/2nd step):111
    
@author: tianxin, martin korevaar
"""

import xarray as xr
import os
import numpy as np
import pandas as pd
import datetime as dt
import pyproj
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import netCDF4 as nc

import json
from urllib.request import urlopen, Request
from datetime import datetime, timedelta
import tarfile
from pathlib import Path
import pygrib
from numpy import sqrt
import shutil
import sys


#%% child functions 
def read_grib(file_name, engine='cfgrib'):
    '''
    Parameters
    ----------
    file_name : str
        file dir+name.
    engine : str, optional
        Only supporting the engine 'cfgrib' for the time being.
        The cfgrib engine supports all read-only features of xarray like:
        - merge the content of several GRIB files into a single dataset using xarray.open_mfdataset,
        - work with larger-than-memory datasets with dask,
        - allow distributed processing with dask.distributed.
    Returns
    -------
    ds : xarray dataset
        the weather forecast from the model.
    '''
    ds = xr.open_dataset(file_name, engine='cfgrib')
    return ds 

def grib2netcdf(output_file_name, ds_grib):
    return ds_grib.to_netcdf(output_file_name)

def file_list(API_URL, DATASET_NAME, DATASET_VERSION, key):
    req = Request(
        # f"{API_URL}/datasets/{DATASET_NAME}/versions/{DATASET_VERSION}/files?startAfterFilename=harm40_v1_p1_{date}{hour}.tar",
        f"{API_URL}/datasets/{DATASET_NAME}/versions/{DATASET_VERSION}/files",
        headers={"Authorization": key}
    )
    with urlopen(req) as list_files_response:
        files = json.load(list_files_response).get("files")

    return files

def get_file(filename, tmpdirname, key, API_URL, DATASET_NAME, DATASET_VERSION):
    req = Request(
        f"{API_URL}/datasets/{DATASET_NAME}/versions/{DATASET_VERSION}/files/{filename}/url",
        headers={"Authorization": key}
    )
    with urlopen(req) as get_file_response:
        url = json.load(get_file_response).get("temporaryDownloadUrl")

    with urlopen(url) as remote:
        with tarfile.open(fileobj=remote, mode='r|*') as tar:
            for tarinfo in tar:
                print(tarinfo.name, flush=True)
                tar.extract(tarinfo, tmpdirname)

def clean_temp_folder(tmpdirname):
    # delete old files
    if os.path.isdir(tmpdirname):
        files_in_dir = os.listdir(tmpdirname)     # get list of files in the directory    
        for file in files_in_dir:                  # loop to delete each file in folder
            os.remove(f'{tmpdirname}/{file}')     

def cron(date, hour, API_URL, DATASET_NAME, DATASET_VERSION, key, temdir, desdir):
    # 
    if os.path.isdir(temdir):
        files_in_dir = os.listdir(temdir)
        if len(files_in_dir)>0:            
            txt = 'yes' #input(f"Need to remove all files in {temdir} to proceed, yes/no?")
            if txt =='yes' or 'y':
                clean_temp_folder(temdir)    
    
    hour_adj = str(float(hour)-0.01) # needed to get 00, 06, 12, 18 sharp
    date_adj = str(float(date)-0.01)
    file_names = file_list(API_URL, DATASET_NAME, DATASET_VERSION, key) # list all files 
    file_name = "harm40_v1_p1_" + date + hour + ".tar" # the requested file
    name_check = 0
    for file in file_names:
        if file["filename"] == file_name:
            name_check = 1
    if name_check == 0:
        print(file_name+' does not exist.')
        file_name = file_list(date, hour_adj, API_URL, DATASET_NAME, DATASET_VERSION, key)[0]["filename"]
        print('The requested file does not exist. Will continue with the oldest available file: '+ file_name)
     
    # run_time = file_names[0]["filename"][-14:-4]
    # run_time_date = datetime.strptime(run_time, '%Y%m%d%H').strftime('%Y-%m-%d_%H')
  
    if (desdir / file_name).exists():
        print(f"Skipping download, {file_name} already downloaded")
    else:
        get_file(file_name, temdir, key, API_URL, DATASET_NAME, DATASET_VERSION)
    # print('Downloaded and now start converting, which will take ~ 3 minutes')

def convert(temdir, desdir):
    tmp_dir = Path(temdir)
    files = sorted(tmp_dir.glob('*_GB'))

    if len(files) != 49:
        print(f'There are unexpected files in the temporary folder. CLEAR UNRELATED FILES Manually')
        sys.exit(1)

    tmp_grib = tmp_dir / 'temp.grb'
    tmp_grib_wind = tmp_dir / 'temp_wind.grb'

    with tmp_grib.open('wb') as gribout, tmp_grib_wind.open('wb') as gribout_wind:
        def writeGribMessage(message, wind=False):
            message['generatingProcessIdentifier'] = 96
            message['centre'] = 'kwbc'

            gribout.write(message.tostring())
            if wind:
                gribout_wind.write(message.tostring())

        for filename in files:
            with pygrib.open(str(filename)) as grbs:
                # see: https://www.knmidata.nl/data-services/knmi-producten-overzicht/atmosfeer-modeldata/data-product-1
                # to link IndicatorOfParameter value to right parameter
                # Mean sea level pressure
                msg_mslp = grbs.select(indicatorOfParameter=1)[0]
                msg_mslp.indicatorOfParameter = 2
                msg_mslp.indicatorOfTypeOfLevel = 'sfc'
                msg_mslp.typeOfLevel = 'meanSea'
                writeGribMessage(msg_mslp)

                # Relative humidity
                msg_rh = grbs.select(indicatorOfParameter=52)[0]
                msg_rh.values = msg_rh.values * 100
                writeGribMessage(msg_rh)

                # Temperature 2m
                msg_t = grbs.select(indicatorOfParameter=11)[0]
                writeGribMessage(msg_t)

                # U-wind
                msg_u = grbs.select(indicatorOfParameter=33)[0]
                writeGribMessage(msg_u, wind=True)

                # V-wind
                msg_v = grbs.select(indicatorOfParameter=34)[0]
                writeGribMessage(msg_v, wind=True)
                
                # short wave radiation (SWR)
                if msg_v.endStep > 0:
                    msg_swr = grbs.select(indicatorOfParameter=111, level=0, stepType='accum')[0]
                    msg_swr.typeOfLevel = 'surface'
                    if msg_swr['P2'] > 0:
                        msg_swr['P1'] = msg_swr['P2'] - 1
                    writeGribMessage(msg_swr)

                # Precipication Intensity
                msg_ip = grbs.select(indicatorOfParameter=181, level=0, stepType='instant')[0]
                msg_ip.indicatorOfParameter = 61
                msg_ip.typeOfLevel = 'surface'
                #msg_ip.level = 0
                msg_ip.values = msg_ip.values * 3600  # mm/s => mm/h
                if msg_ip['P2'] > 0: #??
                    msg_ip['P1'] = msg_ip['P2'] - 1
                writeGribMessage(msg_ip)
                
                # Cumulative Precipication
                msg_ip = grbs.select(indicatorOfParameter=181, level=0, stepType='accum')[0]
                msg_ip.indicatorOfParameter = 62
                msg_ip.typeOfLevel = 'surface'
                #msg_ip.level = 0
                msg_ip.values = msg_ip.values  # unit: mm or ml
                if msg_ip['P2'] > 0:
                    msg_ip['P1'] = msg_ip['P2'] - 1
                writeGribMessage(msg_ip)

                # Wind gusts
                msg_ug = grbs.select(indicatorOfParameter=162)[0]
                msg_vg = grbs.select(indicatorOfParameter=163)[0]
                msg_ug.values = sqrt(msg_ug.values ** 2 + msg_vg.values ** 2)
                msg_ug.indicatorOfParameter = 180
                msg_ug.typeOfLevel = 'surface'
                msg_ug.level = 0
                if msg_ug['P2'] > 0:
                    msg_ug['P1'] = msg_ug['P2'] - 1
                writeGribMessage(msg_ug, wind=True)

    run_time = str(files[0])[-21:-11]
    run_time_date = datetime.strptime(run_time, '%Y%m%d%H').strftime('%Y-%m-%d_%H')
    # print(run_time_date)

    # dst_dir = DATA_DIR / f'{run_time_date}'
    # dst_dir.mkdir(exist_ok=True)

    filename_fmt = f'harmonie_xy_{run_time_date}_{{}}.grb'

    # def bounded_slice(src, dst_dir, name, bounds):
    #     dst = dst_dir / filename_fmt.format(name)
    #     print(f'Writing {name} to {dst}')

    #     cmd = [GGRIB_BIN, src, dst]
    #     cmd.extend(map(str, bounds[0] + bounds[1]))
    #     subprocess.call(cmd)

    # if GGRIB_BIN is None:
    #     print('ggrib binary not found, please install by typing `make ggrib`')
    # else:
    #     for area in BOUNDS:
    #         bounded_slice(tmp_grib, desdirname, area['abbr'], area['bounds'])
    #         bounded_slice(tmp_grib_wind, desdirname, area['abbr'] + '_wind', area['bounds'])

    shutil.move(tmp_grib, Path(desdir) / f'harmonie_xy_{run_time_date}.grb')
    shutil.move(tmp_grib_wind, Path(desdir) / filename_fmt.format('wind'))

    # (DATA_DIR / 'new').symlink_to(dst_dir.relative_to(DATA_DIR))
    # (DATA_DIR / 'new').rename(DATA_DIR / 'latest')
    clean_temp_folder(temdir) 
    


def get_inside_polygon(xgrid, ygrid, data, selection_polygon):
    """
    To get the measurement (data) within a defined polygon
    """
    temp_data = data.copy()
    XY = np.dstack((xgrid, ygrid))
    XY_flat = XY.reshape((-1, 2)) # Transform from 2D to 1D

    path = mpltPath.Path(selection_polygon)
    inside_flat = path.contains_points(XY_flat) # 1D mask of points that are inside
    inside = inside_flat.reshape(xgrid.shape) # 2D mask of points that are inside
    
    temp_data[~inside] = np.nan
    return temp_data

def get_inside_point(x, y, data, point):
    """
    To get the measurement (data) at a given location (point)
    """
    import bisect
    idx = bisect.bisect_left(x, point['lon0'])
    idy = bisect.bisect_left(y, point['lat0'])
    temp_data = data[idx, idy]
    return temp_data            

def convert_coordinates(): # do not need it for now
    pass
    # projparams = {
    #     'proj': 'stere',
    #     'lat_0': 90,
    #     'lon_0': 0,
    #     'lat_ts': 60,
    #     'a': 6378.137,
    #     'b': 6356.752,
    #     'x_0': 0,
    #     'y_0':0    
    # }
    # p1 = pyproj.Proj(projparams)
    
    # wgs84 = pyproj.Proj(init='epsg:4326')
    # new_points = []
    # for cur_point in polygon:
    #     cur_lat = cur_point[0]
    #     cur_lon = cur_point[1]
    #     new_x, new_y = pyproj.transform(wgs84, p1, cur_lon, cur_lat, radians=False)
    #     new_points.append([new_x, new_y])
    # print(np.array(new_points))
    
def plot_grid(xgrid, ygrid, z):
    # test plot
    # z = p_temp[48].values - 273
    fig = plt.figure()
    plt.pcolormesh(xgrid, ygrid, z)
    plt.colorbar()
    
def load_polygon(filename):
    import pandas as pd
    df = pd.read_excel(filename, usecols=['xcoord', 'ycoord'])
    lst = []
    for i in range(df.shape[0]):
        lst.append([df.loc[i, 'xcoord'], df.loc[i, 'ycoord']])
    return lst

def read_grib_file(desdir, filename, point, export_to_excel = False):
    
    grbs = pygrib.open(filename)
    
    # extract parameters    
    p_temp = grbs.select(indicatorOfParameter=11)
    p_hmdt = grbs.select(indicatorOfParameter=52)
    p_uwnd = grbs.select(indicatorOfParameter=33)
    p_vwnd = grbs.select(indicatorOfParameter=34)  
    p_prec = grbs.select(indicatorOfParameter=61) 
    p_cprc = grbs.select(indicatorOfParameter=62) 
    p_prss = grbs.select(indicatorOfParameter=2) 
    p_rdtn = grbs.select(indicatorOfParameter=111) # note the size of radiation is 48, others' size is 49 
    
    # get lats and lons from any parameter and generate a meshgrid
    lats, lons = p_temp[0].latlons()
    y = np.array(lats.transpose()[0])
    x = np.array(lons[0])
    xgrid, ygrid=np.meshgrid(x, y)

    # get df (time series) for a given location 
    date = filename[-17:-13] + filename[-12:-10] + filename[-9:-7]
    hour = filename[-6:-4]
    df48 = pd.DataFrame(columns=[date+hour, 'Temp', 'Hmdt', 'UWnd', 'VWnd', 'Prec', 'CPrc', 'Prss', 'CRdn'])
    for i in range(49):
        df48.loc[i, 'Temp'] = get_inside_point(x, y, p_temp[i].values, point)-273
        df48.loc[i, 'Hmdt'] = get_inside_point(x, y, p_hmdt[i].values, point)
        df48.loc[i, 'UWnd'] = get_inside_point(x, y, p_uwnd[i].values, point)
        df48.loc[i, 'VWnd'] = get_inside_point(x, y, p_vwnd[i].values, point)
        df48.loc[i, 'Prec'] = get_inside_point(x, y, p_prec[i].values, point)
        df48.loc[i, 'CPrc'] = get_inside_point(x, y, p_cprc[i].values, point)
        df48.loc[i, 'Prss'] = get_inside_point(x, y, p_prss[i].values, point)
        if i >= 1:
            df48.loc[i, 'CRdn'] = get_inside_point(x, y, p_rdtn[i-1].values, point)/3600
        df48.loc[i, date+hour] = '+{:02d}hour'.format(i)
    # a quick check of cumulative precipitation and radiation --> look good
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.plot(df48.loc[1:,'CPrc'])
    # plt.show()
    df48 = df48.iloc[1: , :] # drop the first row
    if export_to_excel == True:
        temp_name = "predictionOver48hours_"+filename[-29:-4]+"_lat"+str(point['lat0'])+"lon"+str(point['lon0'])+"_"+".xlsx"
        df48.to_excel(desdir / temp_name)

    return df48



# # get df for a polygon too (if it is aimed to move an animation to show the change over next 48 hours)
# data = get_inside_polygon(xgrid, ygrid, p_temp[1].values, polygon)
# fig = plt.figure()
# plt.pcolormesh(xgrid, ygrid, data)
# plt.colorbar()
# plt.xlim(5.32, 5.6)
# plt.ylim(51.3, 51.6)
# plt.show()






    

    
    
    
    
    
    
    
    
    
    
    
    
    
    