import tensorflow as tf
import numpy as np
from osgeo import gdal
import os

# load the pb model
model = tf.keras.models.load_model(r'model_1\saved_model.pb')

# get the files from S1_data and pass it from the model. Save the output in test_1 folder

output_folder = r'test_1'
path_SAR = r'S1_data'
output_img = []
for file in os.listdir(path_SAR):
    SAR_img = gdal.Open(os.path.join(path_SAR,file))
    SAR_img = SAR_img.GetRasterBand(1)
    SAR_img = SAR_img.ReadAsArray()
    SAR_img = SAR_img.reshape(1,10,10,1)
    output = model(SAR_img)
    output_img.append(output)
    np.save(os.path.join(output_folder,file),output)


# Shape of element of output_img is (10, 10, 3) and total elements in output_img are 74980.
# So we will reshape the output_img to 