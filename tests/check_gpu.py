import tensorflow as tf, torch
print(tf.config.list_physical_devices('GPU'))
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))