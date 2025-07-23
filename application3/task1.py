import idx2numpy
import matplotlib.pyplot as plt

# MNIST Dataset
# path to train files
train_image_file = 'mnist/train/train-images-idx3-ubyte'
train_label_file = 'mnist/train/train-labels-idx1-ubyte'

# path to test files
test_image_file = 'mnist/test/t10k-images-idx3-ubyte'
test_label_file = 'mnist/test/t10k-labels-idx1-ubyte'

# import train files into numpy arrays
train_images = idx2numpy.convert_from_file(train_image_file)
train_labels = idx2numpy.convert_from_file(train_label_file)

# import test files into numpy arrays
test_images = idx2numpy.convert_from_file(test_image_file)
test_labels = idx2numpy.convert_from_file(test_label_file)

# plot training
plt.imshow(train_images[0])  # The first image is plotted
plt.title("MNIST Train Image")
plt.show()

# plot test
plt.imshow(test_images[0])  # The second image is plotted
plt.title("MNIST Test Image")
plt.show()

####################################################################################

# Fasion-MNIST Dataset
# path to train files
f_train_image_file = 'fashion-mnist/train/train-images-idx3-ubyte'
f_train_label_file = 'fashion-mnist/train/train-labels-idx1-ubyte'

# path to test files
f_test_image_file = 'fashion-mnist/test/t10k-images-idx3-ubyte'
f_test_label_file = 'fashion-mnist/test/t10k-labels-idx1-ubyte'

# import files into numpy arrays
f_train_images = idx2numpy.convert_from_file(f_train_image_file)
f_train_labels = idx2numpy.convert_from_file(f_train_label_file)

# import test files into numpy arrays
f_test_images = idx2numpy.convert_from_file(f_test_image_file)
f_test_labels = idx2numpy.convert_from_file(f_test_label_file)

# plot training
plt.imshow(f_train_images[0])  # The first image is plotted
plt.title("Fashion-MNIST Train Image")
plt.show()

# plot test
plt.imshow(f_test_images[0])  # The second image is plotted
plt.title("Fashion-MNIST Test Image")
plt.show()
