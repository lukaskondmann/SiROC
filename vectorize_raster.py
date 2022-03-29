import gdal
import subprocess
import glob
import os
import geopandas as gpd



## Raster Conversion 
class_dict = {
    1: 'Impervious Surfaces',
    2: 'Agriculture',
    3: 'Forest and Other Vegetation',
    4: 'Wetlands',
    5: 'Soil',
    6: 'Water',
    7: 'Snow'
}

# load image and label from UShelf 
path = '/mnt/ushelf/datasets/landcover_dynamicearth/bundle/labels/1417_3281_13_11N'
outpath = './test_conversion.geojson'
for in_file in sorted(glob.glob(path+'/Raster/*/*.tif')):
    for i in range (1,7):
        out_file = path+'/Vector/'+class_dict[i].replace(" ", "_")+'/11N-117W-33N-L3H-SR/'+in_file.split('/')[-1].split('.')[0]+'.geojson'
        if os.path.exists(out_file):
            os.remove(out_file)
            print("File existed. Deleted to avoid writing conflicts.")       
        cmdline = ['gdal_polygonize.py', in_file,"-b",str(i),"-f", "GeoJSON",out_file]
        subprocess.call(cmdline)
        
        # Adjust class id value from 255 to 1-7
        geo_df = gpd.read_file(out_file)
        geo_df.crs = 32611 
        assert 'CODE' not in geo_df, "CODE Column should not exist yet"
        geo_df['CODE']=0
        geo_df.loc[geo_df['DN'] == 255, 'CODE'] = i

        # Add description column 
        geo_df['CLASS'] = geo_df['CODE'].replace(class_dict)
        geo_df = geo_df.drop(geo_df[geo_df['CODE'] == 0].index)
        geo_df = geo_df.drop('DN',1)

        if geo_df.empty:
            print(f'Skipping {in_file} Band {i} because it is empty. Deleting GeoJson')
            os.remove(out_file)
            continue
        #save again 
        geo_df.to_file(out_file)


# Add name band for final files 