import task1 as t1
import numpy as np

# reshape train mnist into 1-D array
train_images_reshape = t1.train_images.reshape(t1.train_images.shape[0], -1)
print(train_images_reshape.shape)

# reshape test mnist into 1-D array
test_images_reshape = t1.test_images.reshape(t1.test_images.shape[0], -1)
print(test_images_reshape.shape)

#######################################################################################

# reshape train fashion-mnist into 1-D array
f_train_images_reshape = t1.f_train_images.reshape(t1.f_train_images.shape[0], -1)
print(f_train_images_reshape.shape)

# reshape test fashion-mnist into 1-D array
f_test_images_reshape = t1.f_test_images.reshape(t1.f_test_images.shape[0], -1)
print(f_test_images_reshape.shape)