import platform
import os
import sys

print("==============================================")
print(os.system("pip list | grep tensor"))
print("==============================================")


import tensorflow as tf
import tensorflow.keras

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")

# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     print(gpu)

print(tf.config.list_physical_devices('GPU'))