# Google Street View Panorama Image downloader
A python tool to download google street view panorama images with given location. Modified based on [robolyst's work](https://github.com/robolyst/streetview)
Based on  a fork from [cplusx](https://github.com/cplusx/google-street-view-panorama-download)
### Function added comparing to [original work](https://github.com/robolyst/streetview)
- Faster panorama image retrieve with panoid. Omitting blank blocks and stitching panorama image on the fly.

### Added Panorama stiching for streetview static api images
- Download 6x 90° fov images from a given panoid with ```download_cube_mapping```
&rarr; added function stiching a cube net preview of how the images are arranged ```cube_stich_tiles```
- Project into an equirectangular projection using ``` equirectangular_projection```  

---

### [Basic Usage](demo.ipynb):
```
import streetview
import matplotlib.pyplot as plt

key = "YOUR API KEY"

panoids = streetview.panoids(lat=47.0734667, lon=15.4344101)
panoid = panoids[0]['panoid']

#size of downloaded tiles (max. 640x640)
width, height = 640,640 

#download cube mapping of pano corresponding to panoid from API
streetview.download_cube_mapping(panoid, "target_folder", key, width=width, 
		height=height, extension='jpg', year=2023, fname=None)
```
```
#you may preview the downloaded tiles in a cube net

cube = streetview.cube_stich_tiles(panoid, 
		"source_folder", "target_folder", 
		tile_width = width, tile_height = height)
		
plt.imshow(cube)
```
![](https://i.imgur.com/mG5pILh.png)
```
#size of pano (2:1)
res_x,res_y = 2560,1280 

#project frames and stich them in equirectangular projection
panorama = streetview.equirectangular_projection(panoid,
		"source_folder","target_folder",
		x_res=res_x,y_res=res_y,
		x_tile_res = width, y_tile_res = height )

plt.imshow(panorama)
```
![](https://i.imgur.com/4EMlt6P.png)

