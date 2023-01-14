import tensorflow as tf
import os
import numpy as np
import cv2
from osgeo import gdal
import sys

class MDN(tf.keras.Model):
  def __init__(self):
    super(MDN, self).__init__()
    self.reshape = tf.keras.layers.Reshape((10,10,1))
    self.conv1 = tf.keras.layers.Conv2D(32, kernel_size= (3, 3), activation='relu')
    self.conv2 = tf.keras.layers.Conv2D(64, kernel_size= (3, 3), activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(6)

  def call(self, SAR_img):
    x = tf.expand_dims(SAR_img, 0)
    x = self.reshape(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.dense(x)
    return x

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder,self).__init__()
    self.reshape = tf.keras.layers.Reshape((1,10,10,3))
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(256, activation='relu')
    self.dense2 = tf.keras.layers.Dense(64, activation='relu')
    self.dense3 = tf.keras.layers.Dense(10, activation='relu')

  def call(self, inputs):
    x,rgb_dist = inputs
    x = tf.expand_dims(x, 0)
    x = self.reshape(x)
    x = self.flatten(x)
    x = tf.concat([x,rgb_dist], axis=1)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x

class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder,self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation='relu')
    self.dense2 = tf.keras.layers.Dense(256, activation='relu')
    self.dense3 = tf.keras.layers.Dense(10*10*3, activation='sigmoid')
    self.reshape = tf.keras.layers.Reshape((256,256,1))

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.dense2(x)
    x = self.dense3(x)
    x = tf.reshape(x, (-1, 10, 10, 3))
    return x

class Model(tf.keras.Model):
  def __init__(self):
    super(Model,self).__init__()
    self.mdn = MDN()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def call(self, SAR_img, opt_img):
    rgb_dist = self.mdn(SAR_img)
    latent = self.encoder([opt_img, rgb_dist])
    recon = self.decoder(latent)
    return recon

path_SAR = 'S1_data'
path_opt = 'S2_data'
def get_data():
  for file1,file2 in zip(os.listdir(path_SAR),os.listdir(path_opt)):
    SAR_img = gdal.Open(os.path.join(path_SAR,file1))
    band_SAR_img = SAR_img.GetRasterBand(1)
    band_SAR_img = band_SAR_img.ReadAsArray()
    opt_img = gdal.Open(os.path.join(path_opt,file2))
    band_opt_img_R = opt_img.GetRasterBand(1)
    band_opt_img_G = opt_img.GetRasterBand(2)
    band_opt_img_B = opt_img.GetRasterBand(3)
    band_opt_img_R = band_opt_img_R.ReadAsArray()
    band_opt_img_G = band_opt_img_G.ReadAsArray()
    band_opt_img_B = band_opt_img_B.ReadAsArray()
    yield band_SAR_img, np.dstack((band_opt_img_R, band_opt_img_G, band_opt_img_B))
  
  
# write the training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model = Model()
save_model_path = os.path.join(os.getcwd(), 'model_1_weights')

def ssim_loss(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def MSE_loss(y_true, y_pred):
  return tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))

num_steps = 10

for epoch in range(2):
  print("Epoch: {}".format(epoch))
  for step in range(num_steps):
    with tf.GradientTape() as tape:
      SAR_img,opt_img = get_data().__next__()
      SAR_img = tf.convert_to_tensor(SAR_img, dtype=tf.float32)
      opt_img = tf.convert_to_tensor(opt_img, dtype=tf.float32)
      recon = model(SAR_img, opt_img)
      # loss = ssim_loss(opt_img, recon)
      loss = MSE_loss(opt_img, recon)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      print("Step: {}, Loss: {}".format(step, loss.numpy()))

tf.saved_model.save(model, save_model_path)

# visualization of the output
import matplotlib.pyplot as plt
SAR_img, opt_img = get_data().__next__()
recon = model(SAR_img, opt_img)
plt.imshow(recon[0])
print(recon[0].shape)
plt.show()

