import numpy as np
import os
from torch.utils.data import Dataset
from osgeo import gdal

class MapDataset(Dataset):
    def __init__(self, x_dir, y_dir):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.x_files = os.listdir(x_dir)
        self.y_files = os.listdir(y_dir)
    
    def __len__(self):
        return len(self.x_files)
    
    def __getitem__(self, index) :
        img_path = (os.path.join(self.x_dir, self.x_files[index]),
                    os.path.join(self.y_dir, self.y_files[index]))
        SAR_img = gdal.Open(img_path[0])
        opt_img = gdal.Open(img_path[1])
        band_sar = SAR_img.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        opt_img_rgb = np.dstack(
            (opt_img.GetRasterBand(1).ReadAsArray(),
            opt_img.GetRasterBand(2).ReadAsArray(),
            opt_img.GetRasterBand(3).ReadAsArray()
            )
        )
        opt_img_rgb = (opt_img_rgb - opt_img_rgb.min())/(opt_img_rgb.max() - opt_img_rgb.min())*255
        opt_img_rgb = opt_img_rgb.astype(np.uint8)
        return band_sar, opt_img_rgb
    