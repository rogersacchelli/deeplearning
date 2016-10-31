from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


#train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
#test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

train_filename = 'notMNIST_large.tar.gz'
test_filename = 'notMNIST_small.tar.gz'

num_classes = 10
np.random.seed(133)


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

"""
-------------------- Problem 2 -------------
Let's verify that the data still looks good.
Displaying a sample of the labels and images from the ndarray.
Hint: you can use matplotlib.pyplot.

REMARKS: Image is normalized and pickled, it's needed to unplicle
and load the image to check if it's correct.
--------------------------------------------
"""


for p_train in train_datasets:
    with open(p_train,mode='rb') as f:
        letter_set = pickle.load(f)
        print ('size of train letter_set %s - %i' % (p_train,len(letter_set)))
        sample_idx = np.random.randint(len(letter_set))
        sample_img = letter_set[sample_idx,:,:]
        #plt.figure()
        #plt.imshow(sample_img)
        #plt.show()
        f.close()

for p_test in test_datasets:
    with open(p_test,mode='rb') as f:
        letter_set = pickle.load(f)
        print ('size of test letter_set %s - %i' % (p_train,len(letter_set)))
        sample_idx = np.random.randint(len(letter_set))
        sample_img = letter_set[sample_idx,:,:]
        plt.figure()
        #plt.imshow(sample_img)
        #plt.show()
        f.close()

'''
---------------- Problem 3 ----------------
Merge and prune the training data as needed.
Depending on your computer setup, you might
not be able to fit it all in memory, and you can tune
`train_size` as needed.

The labels will be stored into a separate array of integers 0 through 9.
Also create a validation dataset for hyperparameter tuning.
-------------------------------------------
'''

## definig the new size of single pickle file
# training and validation can be also be generated by cross validation functions from sklearn
train_size = 200000
validation_size = 100000
test_size = 10000


def make_arrays(nb_rows, img_size_w,img_size_h):

    """ Used to allocate memory for n_images (nb_rows) according to image size.
        It returns two np arrays, dataset and label."""

    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size_w, img_size_h), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(train_datasets,train_size,test_datasets,test_size):

    valid_dataset,valid_labels = make_arrays(validation_size,image_size,image_size)
    train_dataset,train_labels = make_arrays(train_size,image_size,image_size)











