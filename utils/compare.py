import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import asf_search as asf
from shapely.geometry import Polygon
from celluloid import Camera # getting the camera
import rioxarray
import distinctipy

def make_difference_gif(tif_1, tif_2, vmin, vmax, bounds=[], title='', save_path=''):
    """_summary_

    Args:
        tif_1 (_type_): _description_
        tif_2 (_type_): _description_
        vmin (_type_): _description_
        vmax (_type_): _description_
        bounds (list, optional): format of x1,x2,y1,y2. Defaults to [].
        title (str, optional): _description_. Defaults to ''.
        save_path (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """

    if isinstance(tif_1, str):
        with rasterio.open(tif_1) as src:
            tif_1 = src.read(1)
    if isinstance(tif_2, str):
        with rasterio.open(tif_2) as src:
            tif_2 = src.read(1)

    if bounds:
        x1,x2,y1,y2 = bounds
        tif_1 = tif_1[y1:y2,x1:x2]
        tif_2 = tif_2[y1:y2,x1:x2]
    
    fig, ax = plt.subplots() # make it bigger
    camera = Camera(fig)# the camera gets our figure
    for i,img in enumerate([tif_1, tif_2]):
        im = ax.imshow(img,
                  vmin=vmin,
                  vmax=vmax) # plotting
        ax.set_title(f'{title}')
        camera.snap()
    animation = camera.animate()
    if save_path:
        animation.save(save_path)
    return animation

def assign_crs(tif_path, crs):
    with rasterio.open(tif_path, 'r+') as rds:
        print(f'existing crs: {rds.crs}')
        rds.crs = CRS.from_epsg(crs)
        print(f'assigned crs: {rds.crs}')

def plot_histograms(
        tifs, 
        titles, 
        colours, 
        suptitle, 
        convert_dB=True, 
        save_path='',
        show=True,
        nodata=0):
    # plot the histogram 

    if not colours:
        # generate N visually distinct colours
        colours = distinctipy.get_colors(len(tifs))
        
    for i, tif in enumerate(tifs):
        if isinstance(tif,str):
            # read in the tif
            print(f'loading: {tif}')
            with rasterio.open(tif) as src:
                data = src.read(1)
                nodata = src.nodata
        else:
            # data is array
            data = tif
        # covert no data to nan
        data[data==nodata] = np.nan
        # covert from linear to db
        if convert_dB:
            data = 10*np.log10(np.array(data))
        hist_data = data[(np.isfinite(data))]

        u, std = np.mean(hist_data), np.std(hist_data)
        plt.hist(hist_data, 
                density=True,
                bins=60, 
                alpha=0.5, 
                label=f'{titles[i]}; u={u:.3f}, std={std:.3f}', 
                color=colours[i],
                histtype='step')

    plt.title(f'{suptitle}')
    plt.xlabel('Gamma0 RTC')
    plt.ylabel('Frequency')
    plt.legend(loc='best')
    plt.grid(True)

    if save_path:
        print(f'saving image to : {save_path}')
        plt.savefig(save_path)

    if show:
        plt.show()

def plot_tifs(
    tif_1, 
    tif_2, 
    titles= ['arr_1', 'arr_2'],
    colours=['red','blue'],
    suptitle='suptitle',
    convert_dB = True,
    scale=[-40,10],
    cmap = 'binary_r',
    save_path='',
    nodata=0,
    show=True):

    # place to store data
    meta = []
    
    # plot the tif
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,10))
    for i, tif in enumerate([tif_1, tif_2]):
        if isinstance(tif,str):
            # read in the tif
            with rasterio.open(tif) as src:
                data = src.read(1)
                nodata = src.nodata
                meta.append(src.meta.copy())
        else:
            # data is array
            data = tif
        # covert from linear to db
        if convert_dB:
            data = 10*np.log10(data)
        # covert no data to nan
        data[data==nodata] = np.nan
        if scale is not None:
            im = ax[i].imshow(data, cmap=cmap, vmin=scale[0], vmax=scale[1])
        else:
            im = ax[i].imshow(data, cmap=cmap)
        ax[i].set_title(f'{titles[i]}')

    plt.suptitle(f'{suptitle}', y=0.9)
    cbar_ax = f.add_axes([0.95, 0.15, 0.04, 0.7])
    f.colorbar(im, cax=cbar_ax)

    for i, tif in enumerate([tif_1, tif_2]):
        # print metadata if rasterio tif
        if isinstance(tif,str):
            print(tif)
            for k in meta[i].keys():
                print(f'{k} : {meta[i][k]}')
            print('\n')

    if save_path:
        print(f'saving image to : {save_path}')
        plt.savefig(save_path)

    if show:
        plt.show()

    
def plot_difference_maps(
    arr_1, 
    arr_2, 
    titles=['arr_1','arr_2','diff (arr_1 - arr_2)'],
    scales = [[-40,10],[-40,10],[-1,1]],
    ylabels=['decibels (dB)','decibels (dB)','decibels (dB)'],
    save_path='',
    exclude_nodata=True):
    
    if isinstance(arr_1, str):
        with rasterio.open(arr_1) as src:
                arr_1 = src.read(1)
    if isinstance(arr_2, str):
        with rasterio.open(arr_2) as src:
                arr_2 = src.read(1)
    
    # ensure shapes are the samed, required for difference
    assert arr_1.shape == arr_2.shape, "tifs are two different shapes, must be identical for difference calculation"
    
    # only compare where we have data in both tifs
    if exclude_nodata:
        arr_2[(~np.isfinite(arr_1))] = np.nan
        arr_1[(~np.isfinite(arr_2))] = np.nan

    diff = arr_1 - arr_2
    arrs = [arr_1, arr_2, diff]
    stats_arr = np.array(diff)[np.array((np.isfinite(diff)))]
    print('Difference Stats')
    print(f'min: {stats_arr.min()}', 
        f'max: {stats_arr.max()}',
        f'mean: {stats_arr.mean()}',
        f'median: {np.percentile(stats_arr, 50)}',
        f'5th percentile: {np.percentile(stats_arr, 5)}',
        f'90th percentile: {np.percentile(stats_arr, 95)}',
        )

    cmaps = ['binary_r','binary_r','bwr']

    f, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,40))
    for i,arr in enumerate(arrs):
        im = ax[i].imshow(arr, 
                vmin = scales[i][0], 
                vmax = scales[i][1],
                cmap = cmaps[i])
        ax[i].set_title(titles[i])
        f.colorbar(im, ax=ax[i], label=ylabels[i])
        
    # plot the histogram
    colors = ['red','blue']
    for i in [0,1]:
        # only get real values 
        hist_data = np.array(arrs[i])[
                (np.isfinite(np.array(arrs[i])))
                ]
        u, std = np.mean(hist_data), np.std(hist_data)
        ax[3].hist(hist_data, 
                density=True,
                bins=60, 
                alpha=0.5, 
                label=f'{titles[i]}; u={u:.3f}, std={std:.3f}', 
                color=colors[i],
                histtype='step')
        ax[3].set_title('Pixel distribution')

    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)

def reproject_match_tifs(
        tif_1, 
        tif_2, 
        target_crs,
        scene_poly=None,
        save_path_1='',
        save_path_2='',
        convert_dB=True,
        set_nodata=False):
    
    tif_1_f = rioxarray.open_rasterio(tif_1)
    tif_2_f = rioxarray.open_rasterio(tif_2)
    crs_1 = tif_1_f.rio.crs
    crs_2 = tif_2_f.rio.crs
    nodata_1 = tif_1_f.rio.nodata
    nodata_2 = tif_2_f.rio.nodata

    # reproject raster to target crs if not already
    print('reprojecting arrays if not in target crs')
    print(f'target crs: {target_crs}, crs_1: {crs_1}, crs_2: {crs_2}')
    tif_1_reproj = tif_1_f.rio.reproject(f"EPSG:{target_crs}") if str(tif_1_f.rio.crs) != str(target_crs) else tif_1_f
    tif_2_reproj = tif_2_f.rio.reproject(f"EPSG:{target_crs}") if str(tif_2_f.rio.crs) != str(target_crs) else tif_2_f
    
    del tif_1_f, tif_2_f
    print('Shape of arrays after reprojection to target crs')
    print(tif_1_reproj.shape, tif_2_reproj.shape)
    # clip by the scene geometry
    if scene_poly is not None:
        print('clip arrays by the scene bounds')
        tif_1_clipped = tif_1_reproj.rio.clip([scene_poly], CRS.from_epsg(4326))
        tif_2_clipped = tif_2_reproj.rio.clip([scene_poly], CRS.from_epsg(4326))
        print('Shape of arrays after being clipped by scene bounds')
        print(tif_1_clipped.shape, tif_2_clipped.shape)
    else:
        tif_1_clipped = tif_1_reproj.copy()
        tif_2_clipped = tif_2_reproj.copy()

    del tif_1_reproj, tif_2_reproj
    # match the shape and resolution of the two tifs as they may be slighly off
    # match tif 2 to tif 1 if tif 1 did not require reprojection originally
    print('matching shape and resolutions')
    if str(crs_1) == str(target_crs):
        print('reprojecting tif 2 to tif 1')
        tif_2_matched = tif_2_clipped.rio.reproject_match(tif_1_clipped)
        tif_1_matched = tif_1_clipped.copy()
    else:
        print('reprojecting tif 1 to tif 2')
        tif_1_matched = tif_1_clipped.rio.reproject_match(tif_2_clipped)
        tif_2_matched = tif_2_clipped.copy()
    del tif_1_clipped, tif_2_clipped
    print('Shape of arrays after matching tifs')
    print(tif_1_matched.shape, tif_2_matched.shape)

    if set_nodata:
        print(f'setting nodata values to {set_nodata}')
        tif_1_matched.where(nodata_1,set_nodata)
        tif_2_matched.where(nodata_2,set_nodata)
        tif_1_matched.rio.write_nodata(set_nodata, encoded=True, inplace=True)
        tif_2_matched.rio.write_nodata(set_nodata, encoded=True, inplace=True)

    if convert_dB:
        print('converting from linear to db')
        tif_1_matched = 10*np.log10(tif_1_matched)
        tif_2_matched = 10*np.log10(tif_2_matched)

    if save_path_1:
        print(f'Saving to: {save_path_1}')
    if save_path_2:
        print(f'Saving to: {save_path_2}')    

    tif_1_matched.rio.to_raster(save_path_1)
    tif_2_matched.rio.to_raster(save_path_2)

    return tif_1_matched, tif_2_matched

def plot_array(
    arr, 
    cmap='binary_r',
    vmin=None,
    vmax=None,
    title='',
    ylabel='',
    save_path=''):

    if isinstance(arr,str):
        with rasterio.open(arr) as src:
                arr = src.read(1)

    stats_arr = np.array(arr)[np.array((np.isfinite(arr)))]
    print('Array Stats (excl nodata)')
    print(f'min: {stats_arr.min()}', 
        f'max: {stats_arr.max()}',
        f'mean: {stats_arr.mean()}',
        f'median: {np.percentile(stats_arr, 50)}',
        f'5th percentile: {np.percentile(stats_arr, 5)}',
        f'90th percentile: {np.percentile(stats_arr, 95)}',
        )

    # calculate percentiles if vmin or vmax is set as 'PXX'
    # e.g. vmin = 'P5'
    if vmin:
        if str(vmin)[0].upper() == 'P':
            vmin = np.percentile(stats_arr,int(vmin[1:]))
    if vmax:
        if str(vmax)[0].upper() == 'P':
            vmax = np.percentile(stats_arr,int(vmax[1:]))

    f, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(arr, 
            vmin = vmin, 
            vmax = vmax,
            cmap = cmap)
    plt.title(title)
    f.colorbar(im, ax=ax, label=ylabel)
    if save_path:
        plt.savefig(save_path)