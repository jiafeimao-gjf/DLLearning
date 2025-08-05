import tensorflow as tf

print("tf version",tf.__version__)

print("tf gpu",tf.config.list_physical_devices('GPU'))
