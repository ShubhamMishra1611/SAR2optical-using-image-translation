import tensorflow as tf
import os
import numpy as np

class MDN(tf.keras.Model):
  def __init__(self):
    super(MDN, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(32, kernel_size= (3, 3), activation='relu')
    self.conv2 = tf.keras.layers.Conv2D(64, kernel_size= (3, 3), activation='relu')
    self.flatten = tf.keras.Layers.Flatten()
    self.dense = tf.keras.layers.Dense(6)

  def call(self, SAR_img):
    x = self.conv1(SAR_img)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.dense(x)
    return x

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder,self).__init__()
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(256, activation='relu')
    self.dense2 = tf.keras.layers.Dense(64, activation='relu')
    self.dense3 = tf.keras.layers.Dense(10, activation='relu')

  def call(self, inputs):
    x,rgb_dist = inputs
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
    self.mdn = MDN()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, SAR_img, opt_img):
    rgb_dist = self.mdn(SAR_img)
    latent = self.encoder([opt_img, rgb_dist])
    recon = self.decoder(latent)
    return recon

path_SAR = 'S1_data'
path_opt = 'S2_data'
def get_data():
  for file1,file2 in zip(os.listdir(path_SAR),os.listdir(path_opt)):
    SAR_img = np.load(os.path.join(path_SAR,file1))
    opt_img = np.load(os.path.join(path_opt,file2))
    yield SAR_img, opt_img
  
# write the training loop
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model = Model()
loss_fn = tf.keras.losses.MeanSquaredError()

num_steps = 100

for epoch in range(100):
  print("Epoch: {}".format(epoch))
  for step in range(num_steps):
    with tf.GradientTape() as tape:
      SAR_img, opt_img = get_data()
      recon = model(SAR_img, opt_img)
      loss = loss_fn(recon, opt_img)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      print("Step: {}, Loss: {}".format(step, loss.numpy()))

# visualization of the output
import matplotlib.pyplot as plt
SAR_img, opt_img = get_data()
recon = model(SAR_img, opt_img)
plt.imshow(recon[0])
plt.show()

