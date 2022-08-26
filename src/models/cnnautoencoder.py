import tensorflow as tf

class CNNAutoencoder(tf.keras.models.Model):

  def __init__(self,latent_dim,):
    super(CNNAutoencoder, self).__init__()
    self.latent_dim = latent_dim   
    
    # encoder
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, padding="same", activation="relu"),
      tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same", activation="relu"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim, activation="relu")
    ])

    # decoder
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(8 * 8 * latent_dim , activation='relu'),
      tf.keras.layers.Reshape((8, 8, latent_dim)), # output (8,8,latent_dim)
      tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(2,2), strides=2, activation="relu"), # output (16,16,16)
      tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(2,2), strides=2, activation="relu"),  # output (32,32,8)
      tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(2,2), strides=2, activation="relu")   # output (64,64,1) 
    ])

  def call(self, x):

    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded