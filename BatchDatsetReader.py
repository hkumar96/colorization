"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
from scipy import misc
from skimage import color
import cv2


class BatchDatset:
    files = []
    images = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of files to read -
        :param image_options: Dictionary of options to modify the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image
        color=LAB, RGB, HSV
        """
        print("Initializing Batch Dataset Reader...")
        self.pixel_distr = np.load('soft_encoder.npy')
        self.rebalancing = np.load('weights.npy')
        self.rebalancing = np.reshape(self.rebalancing,(16,16))
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()
        self.Q = self.image_options['Q'];

    def _read_images(self):
        # print(self.files)
        self.images = np.array([self._transform(filename) for filename in self.files])
        self.big_images = np.array([self._transform_r(resize_image) for resize_image in self.images])
        np.save('prob_img',self.big_images)
        self.rebal_weights = np.array([self._transform_rb(resize_image) for resize_image in self.images])
        print (self.images.shape, self.big_images.shape)

    def _transform(self, filename):
        # print(filename)
        try:
            image = misc.imread(filename)
            if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
                image = np.array([image for i in range(3)])

            if self.image_options.get("resize", False) and self.image_options["resize"]:
                resize_size = int(self.image_options["resize_size"])
                resize_image = cv2.resize(image, dsize = (resize_size, resize_size))
            else:
                resize_image = image

            # print(resize_image)
            if self.image_options.get("color", False):
                option = self.image_options['color']
                if option == "LAB":
                    resize_image = cv2.cvtColor(resize_image, cv2.COLOR_RGB2LAB)
                elif option == "HSV":
                    resize_image = color.rgb2hsv(resize_image)
        except:
            print ("Error reading file: %s of shape %s" % (filename, str(image.shape)))
            raise

        return np.array(resize_image)

    def _transform_r(self, resize_image):
        # print(resize_image)
        a_vector = resize_image[:, :, 1].flatten()
        b_vector = resize_image[:, :, 2].flatten()
        # print(a_vector,b_vector)
        prob_image = self.pixel_distr[a_vector, b_vector, :]

        prob_image = np.reshape(prob_image,(resize_image.shape[0],
                                 resize_image.shape[1], 256))

        return np.array(prob_image)

    def _transform_rb(self, resize_image):
        # print(resize_image)
        a_vector = (resize_image[:, :, 1].flatten().astype(float)*16/256).astype(np.uint8);
        b_vector = (resize_image[:, :, 2].flatten().astype(float)*16/256).astype(np.uint8);
        # print(a_vector,b_vector)
        re_image = self.rebalancing[a_vector, b_vector]

        re_image = np.reshape(re_image,(resize_image.shape[0],
                                 resize_image.shape[1]))

        print("reimage shape is:",re_image.shape)
        return np.array(re_image)

    def get_records(self):
        return self.images

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        images = self.images[start:end]
        prob_images = self.big_images[start:end]
        rebal_w = self.rebal_weights[start:end]
        return np.expand_dims(images[:, :, :, 0], axis=3), prob_images,rebal_w

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        images = self.images[indexes]
        prob_images = self.big_images[indexes]
        rebal_w = self.rebal_weights[indexes]
        return np.expand_dims(images[:, :, :, 0], axis=3), prob_images,rebal_w, images
