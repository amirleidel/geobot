# -*- coding: utf-8 -*-
"""
Original code is from https://github.com/robolyst/streetview


"""

import re
from datetime import datetime
import requests
import time
import shutil
import itertools
from PIL import Image
from io import BytesIO
import os
import numpy as np

from skimage import io
from scipy.interpolate import griddata

def _panoids_url(lat, lon):
    """
    Builds the URL of the script on Google's servers that returns the closest
    panoramas (ids) to a give GPS coordinate.
    """
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)


def _panoids_data(lat, lon, proxies=None):
    """
    Gets the response of the script on Google's servers that returns the
    closest panoramas (ids) to a give GPS coordinate.
    """
    url = _panoids_url(lat, lon)
    return requests.get(url, proxies=None)


def panoids(lat, lon, closest=False, disp=False, proxies=None):
    """
    Gets the closest panoramas (ids) to the GPS coordinates.
    If the 'closest' boolean parameter is set to true, only the closest panorama
    will be gotten (at all the available dates)
    """

    resp = _panoids_data(lat, lon)

    # Get all the panorama ids and coordinates
    # I think the latest panorama should be the first one. And the previous
    # successive ones ought to be in reverse order from bottom to top. The final
    # images don't seem to correspond to a particular year. So if there is one
    # image per year I expect them to be orded like:
    # 2015
    # XXXX
    # XXXX
    # 2012
    # 2013
    # 2014

    pans = re.findall('\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', resp.text)
    pans = [{
        "panoid": p[0],
        "lat": float(p[1]),
        "lon": float(p[2])} for p in pans]  # Convert to floats

    # Remove duplicate panoramas
    pans = [p for i, p in enumerate(pans) if p not in pans[:i]]

    if disp:
        for pan in pans:
            print(pan)

    # Get all the dates
    # The dates seem to be at the end of the file. They have a strange format but
    # are in the same order as the panoids except that the latest date is last
    # instead of first.
    dates = re.findall('([0-9]?[0-9]?[0-9])?,?\[(20[0-9][0-9]),([0-9]+)\]', resp.text)
    dates = [list(d)[1:] for d in dates]  # Convert to lists and drop the index

    if len(dates) > 0:
        # Convert all values to integers
        dates = [[int(v) for v in d] for d in dates]

        # Make sure the month value is between 1-12
        dates = [d for d in dates if d[1] <= 12 and d[1] >= 1]

        # The last date belongs to the first panorama
        year, month = dates.pop(-1)
        pans[0].update({'year': year, "month": month})

        # The dates then apply in reverse order to the bottom panoramas
        dates.reverse()
        for i, (year, month) in enumerate(dates):
            pans[-1-i].update({'year': year, "month": month})

    # # Make the first value of the dates the index
    # if len(dates) > 0 and dates[-1][0] == '':
    #     dates[-1][0] = '0'
    # dates = [[int(v) for v in d] for d in dates]  # Convert all values to integers
    #
    # # Merge the dates into the panorama dictionaries
    # for i, year, month in dates:
    #     pans[i].update({'year': year, "month": month})

    # Sort the pans array
    def func(x):
        if 'year'in x:
            return datetime(year=x['year'], month=x['month'], day=1)
        else:
            return datetime(year=3000, month=1, day=1)
    pans.sort(key=func)

    if closest:
        return [pans[i] for i in range(len(dates))]
    else:
        return pans


def api_download(panoid, flat_dir, key, width=640, height=640,heading = 0,
                 fov=120, pitch=0, extension='jpg', year=2017, fname=None):
    """
    Download an image using the official API. These are not panoramas.

    Params:
        :panoid: the panorama id
        :heading: the heading of the photo. Each photo is taken with a 360
            camera. You need to specify a direction in degrees as the photo
            will only cover a partial region of the panorama. The recommended
            headings to use are 0, 90, 180, or 270.
        :flat_dir: the direction to save the image to.
        :key: your API key.
        :width: downloaded image width (max 640 for non-premium downloads).
        :height: downloaded image height (max 640 for non-premium downloads).
        :fov: image field-of-view.
        :image_format: desired image format.
        :fname: file name

    You can find instructions to obtain an API key here: https://developers.google.com/maps/documentation/streetview/
    """
    if not fname:
        fname = "%s_%s_%s" % (year, panoid, str(heading))
    image_format = extension if extension != 'jpg' else 'jpeg'

    url = "https://maps.googleapis.com/maps/api/streetview"

    params = {
        # maximum permitted size for free calls
        "size": "%dx%d" % (width, height),
        "fov": fov,
        "pitch": pitch,
        "heading": heading,
        "pano": panoid,
        "key": key
    }


     
    response = requests.get(url, params=params, stream=True)
    
    
    try:
        img = Image.open(BytesIO(response.content))
        filename = f"{flat_dir}/{fname}.{extension}" if flat_dir else f"{fname}.{extension}"
        img.save(filename, image_format)
        
    except:
        print("Image not found")
        filename = None
        

    del response
    return filename


def download_full_pano(panoid, flat_dir, key, zoom = 2, width=160, height=160, extension='jpg', year=2017, fname=None):

    fov_lvls = [180,90,45,22.5,11.25] 
    # fov for every zoom lvl from 0 to 4
    

    
    fov = fov_lvls[zoom]
    if not fname:
        fname = f"{year}_{panoid}"
        
    image_format = extension if extension != 'jpg' else 'jpeg'
    url = "https://maps.googleapis.com/maps/api/streetview"
    
    rows    = int(360 / fov)
    columns = int(180 / fov)
    
    print(rows,"x",columns,"panos")
    
    ''' pano indexing 
    
       i -->
    j  0x0         ...   0xrows-1
    |   .                   .
    |   .                   .
    \/  .                   .
       columns-1x0 ... columns-1xrows-1
    
    '''
        
    for i in range(rows):
        for j in range(columns):
            
            # heading, pitch is centered on pano (?)
            # -> add fov/2
            
            heading = int(fov/2 + i*fov)
            pitch = int(90-(fov/2 + j*fov))
            
            params = {
                # maximum permitted size for free calls
                "size": "%dx%d" % (width, height),
                "fov": fov,
                "pitch": pitch,
                "heading": heading,
                "pano": panoid,
                "key": key
            }

            response = requests.get(url, params=params, stream=True)

            img = Image.open(BytesIO(response.content))
            filename = f"{flat_dir}/{fname}_{i}x{j}.{extension}" if flat_dir else f"{fname}.{extension}"
            img.save(filename, image_format)

            del response
    
def stich_tiles(panoid, directory, final_directory, tile_width=160, tile_height=160, zoom = 2,year=2017):
    """
    Stiches all the tiles of a panorama together. The tiles are located in
    `directory'.
    """

    fov_lvls = [180,90,45,22.5,11.25] 
    fov = fov_lvls[zoom]
    
    rows    = int(360 / fov)
    columns = int(180 / fov)
    
    panorama = Image.new('RGB', (rows*tile_width, columns*tile_height))
    
    for i in range(rows):
        for j in range(columns):
            print(i,j)

            fname = f"{directory}/{year}_{panoid}_{i}x{j}.jpg"

            tile = Image.open(fname)

            panorama.paste(im=tile, box=(i*tile_width, j*tile_height))

            del tile

    panorama.save(f"{final_directory}/{panoid}.jpg")
    del panorama
    

def download_flats(panoid, flat_dir, key, width=400, height=300,
                   fov=120, pitch=0, extension='jpg', year=2017):
    for heading in [0, 90, 180, 270]:
        api_download(panoid, heading, flat_dir, key, width, height, fov, pitch, extension, year)


def download_panoids(lat,lng,key):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    
    ###
    ### No quota is consumed when you request metadata.
    ###
    
    params = { 
            "location": f"{lat},{lng}",
            "radius": 3000,
            "key": key
        }
    
    response = requests.get(url, params=params, stream=True)
    
    return response.json()


import hashlib
import hmac
import base64
import urllib.parse as urlparse

def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()


if __name__ == "__main__":
    input_url = input("URL to Sign: ")
    secret = input("URL signing secret: ")
    print("Signed URL: " + sign_url(input_url, secret))
    

def download_cube_mapping(panoid, flat_dir, key, secret = False, width=160, height=160, extension='jpg', year=2017, fname=None):
    
    if not fname:
        fname = f"{panoid}"
        
    image_format = extension if extension != 'jpg' else 'jpeg'
    url = "https://maps.googleapis.com/maps/api/streetview"
    
    if secret:
        url = sign_url(url, secret)
    

    for i,j in ((0,0),(1,0),(2,0),(3,0),(1,-1),(1,1)):
        
        fov = 90 
            
        heading = i*fov
        pitch = j*90
        
        params = {
            # maximum permitted size for free calls
            "size": "%dx%d" % (width, height),
            "fov": fov,
            "pitch": pitch,
            "heading": heading,
            "pano": panoid,
            "key": key
        }

        response = requests.get(url, params=params, stream=True)

        img = Image.open(BytesIO(response.content))
        filename = f"{flat_dir}/{fname}_{i}x{j}.{extension}" if flat_dir else f"{fname}.{extension}"
        
        rgb_im = img.convert('RGB')
        rgb_im.save(filename, image_format)

        del response
    
   
def cube_stich_tiles(panoid, directory, final_directory, tile_width=160, tile_height=160,year=2017):
    """
    Stiches all the tiles of a panorama together. The tiles are located in
    `directory'.
    """
    
    panorama = Image.new('RGB', (4*tile_width, 3*tile_height))
    
    for i,j in ((0,0),(1,0),(2,0),(3,0),(1,-1),(1,1)):
        #print(i,j)

        fname = f"{directory}/{panoid}_{i}x{j}.jpg"
        #print(fname)
        tile = Image.open(fname)

        panorama.paste(im=tile, box=(i*tile_width, (1-j)*tile_height))

        del tile

    panorama.save(f"{final_directory}/{panoid}.jpg")
    
    return panorama
 
    
    
# TODO:
#
#  -variable res
#  -better perf by sampling less at edges
#  -multiprocessing?



def equatorial_sampling_points(img_index, scale_factor,x_res=1280, y_res=640):
    
    lam = np.linspace(-45+img_index*90,45+img_index*90,y_res//2)  #longitude / heading
    phi = np.linspace(45,-45,y_res//2) #azimuth / pitch

    lam,phi = np.meshgrid(lam,phi)

    lam_ = lam-90*img_index #((lam + 45) % 90) - 45 #90°-periodic lam, -45 < lam_ < 45
    #index = np.array(((lam+45)//90) % 4, dtype=int) #image index along lam
    x = np.tan(lam_*np.pi/180) #x,y in coordinates on cube face orentation given by i x j
    y = np.tan(phi*np.pi/180) / np.cos(lam_*np.pi/180)

    '''orientation:
             /\
             |
             y
    <--- x center

    => convert to image coordinates

      ---> x_px
    |
    |
    \/ y_px

    ''' 

    x_px = x_res/4 * 1/2 * (x+1)
    y_px = y_res/2 * 1/2 * (1-y)
    
    return x_px*scale_factor,y_px*scale_factor

def polar_sampling_points(img_index,scale_factor,polar=True,upper=True, x_res=1280, y_res=640): #img_index is here heading corresponding to equatorial index
    
    lam = np.linspace(-45,360-45,x_res)  #longitude / heading
    
    if img_index == 4: #for sampling pole indices 4,5
        phi = np.linspace(-45,-90,y_res//4) #lat / pitch
    else:
        phi = np.linspace(90,45,y_res//4) #lat / pitch
    
    if not polar: #for indices 0...3
        if upper:
            lam = np.linspace(-45+img_index*90,45+img_index*90,x_res//4)  #longitude / heading

            #phi = np.linspace(45,-45,y_res//2) #lat / pitch (slower)
            phi = np.linspace(45,-1e-10,y_res//4) #lat / pitch
            
            
        else:
            if img_index % 2 == 0: #i dont know why i have to do this, but it works
                lam = np.linspace(+45+img_index*90,-45+img_index*90,x_res//4)  #longitude / heading
            else:
                img_index += 2
                img_index %= 4
                
                lam = np.linspace(+45+img_index*90,-45+img_index*90,x_res//4)  #longitude / heading
            
            #phi = np.linspace(-45,+45,y_res//2) #lat / pitch (slower)
            phi = np.linspace(-1e-10,45,y_res//4) #lat / pitch 
            #should be flipped
    
    lam,phi = np.meshgrid(lam,phi)

    #lam_ = ((lam + 45) % 90) - 45 #90°-periodic lam, -45 < lam_ < 45
    #index = np.array(((lam+45)//90) % 4, dtype=int) #image index along lam
    x = (1 if img_index == 4 else -1)*np.cos(lam*np.pi/180)/np.tan(phi*np.pi/180) #x,y in coordinates on cube face orentation given by i x j
    y = -np.sin(lam*np.pi/180)/np.tan(phi*np.pi/180) 

    '''orientation:
             /\
             |
             y
    <--- x center

    => convert to image coordinates

      ---> x_px
    |
    |
    \/ y_px

    ''' 

    x_px = x_res/4 * 1/2 * (x+1)
    y_px = y_res/2 * 1/2 * (1-y)
    
    return x_px*scale_factor,y_px*scale_factor



def equirectangular_projection(panoid, directory, final_directory, x_res=1280, y_res=640, x_tile_res = 320, y_tile_res = 320 ):
    

    #now load the 6 images to project from
    #year = 2017
    #directory = "test1"
    
    scale_factor = x_tile_res // (x_res // 4) 
    

    data      = np.empty((6,y_tile_res,x_tile_res,3),dtype= int)
    
    projected = np.empty((4,y_res//2,x_res//4,3),dtype= int)   #equatorial indices
    projected2 = np.empty((2,y_res//4,x_res,3),dtype= int) #polar

    for img_index,(i,j) in enumerate(((1,-1),(1,1))): #polar indices
        img_index += 4


        fname = f"{directory}/{panoid}_{i}x{j}.jpg" 

        img = Image.open(fname)
        data[img_index] = np.asarray( img, dtype = int)[:,:,:3]

        del img

        x_px,y_px = polar_sampling_points(img_index, scale_factor, x_res=x_res, y_res=y_res)


        #interpolation method = 'nearest', for appropriate clipping 0...255 and speed
        
        t_r,t_g,t_b = [griddata(np.array(np.meshgrid(np.arange(y_tile_res),np.arange(x_tile_res))).reshape(2,-1).T,
                                np.ndarray.flatten(data[img_index,:,:,color_index]),
                                (x_px,y_px),
                                method='nearest',
                               ) for color_index in (0,1,2)]
        
        
        #print(fname)
        print(img_index, end = " ")

        t = np.stack([t_r,t_g,t_b], axis = 2)


        projected2[(img_index-4)] = t

    for img_index,(i,j) in enumerate(((0,0),(1,0),(2,0),(3,0))): #equatorial indices

        fname = f"{directory}/{panoid}_{i}x{j}.jpg" 

        img = Image.open(fname)
        data[img_index] = np.asarray( img, dtype = int)[:,:,:3]

        del img

        x_px,y_px = equatorial_sampling_points(img_index, scale_factor, x_res=x_res, y_res=y_res)


        #interpolation method = 'nearest', for appropriate clipping 0...255 and speed
        t_r,t_g,t_b = [griddata(np.array(np.meshgrid(np.arange(y_tile_res),np.arange(x_tile_res))).reshape(2,-1).T,
                                np.ndarray.flatten(data[img_index,:,:,color_index]),
                                (x_px,y_px),
                                method='nearest',
                               ) for color_index in (0,1,2)]
        #print(fname)
        print(img_index, end = " ")


        #edges of pano that have to be projected from poles

        x_upper_sides_px,y_upper_sides_px = polar_sampling_points(img_index,scale_factor,polar=False,upper=True, x_res=x_res, y_res=y_res)
        x_lower_sides_px,y_lower_sides_px = polar_sampling_points(img_index,scale_factor,polar=False,upper=False, x_res=x_res, y_res=y_res)
        
        #these have to be sampled from the upper and lower polar img_index
        t_upper_r,t_upper_g,t_upper_b = [griddata(np.array(np.meshgrid(np.arange(y_tile_res),
                                                                       np.arange(x_tile_res))).reshape(2,-1).T,
                                                  np.ndarray.flatten(data[5,:,:,color_index]),
                                                  (x_upper_sides_px,y_upper_sides_px),
                                                  method='nearest',
                                                 ) for color_index in (0,1,2)]

        t_lower_r,t_lower_g,t_lower_b = [griddata(np.array(np.meshgrid(np.arange(y_tile_res),
                                                                       np.arange(x_tile_res))).reshape(2,-1).T,
                                                  np.ndarray.flatten(data[4,:,:,color_index]),
                                                  (x_lower_sides_px,y_lower_sides_px),
                                                  method='nearest',
                                                 ) for color_index in (0,1,2)]

        ''' slower
        t_sides_r = np.vstack([t_upper_r[:y_res//4],t_lower_r[y_res//4:]])
        t_sides_g = np.vstack([t_upper_g[:y_res//4],t_lower_g[y_res//4:]])
        t_sides_b = np.vstack([t_upper_b[:y_res//4],t_lower_b[y_res//4:]])
        '''
        t_sides_r = np.vstack([t_upper_r,t_lower_r])
        t_sides_g = np.vstack([t_upper_g,t_lower_g])
        t_sides_b = np.vstack([t_upper_b,t_lower_b])
       
        
        #t_sides = np.stack([t_sides_r,t_sides_g,t_sides_b], axis = 2)

        t = np.stack([np.where(y_px < 0, t_sides_r , np.where(y_px > y_tile_res, t_sides_r, t_r)),
                      np.where(y_px < 0, t_sides_g , np.where(y_px > y_tile_res, t_sides_g, t_g)),
                      np.where(y_px < 0, t_sides_b , np.where(y_px > y_tile_res, t_sides_b, t_b))
                     ]
                     , axis = 2)

        projected[img_index] = t
    
    #now stich the projections together
    print()
    stiched_projected = np.concatenate((projected[0],projected[1],projected[2],projected[3]),axis = 1) #all equat. images
    full_pano = np.concatenate((projected2[1],stiched_projected,projected2[0]),axis = 0) #add poles
    
    im = Image.fromarray(np.array(full_pano,dtype=np.uint8()))
    im.save(f"{final_directory}/full_pano_{panoid}.png")

    return full_pano

