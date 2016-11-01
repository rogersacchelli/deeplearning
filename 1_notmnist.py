from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import time
import hashlib
from IPython.display import display, Image
from scipy import ndimage, random
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

## pickle files for traininin and test
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

def sanity_check(train,test):


    for p_train in train:
        with open(p_train,mode='rb') as f:
            letter_set = pickle.load(f)
            print ('size of train letter_set %s - %i' % (p_train,len(letter_set)))
            sample_idx = np.random.randint(len(letter_set))
            sample_img = letter_set[sample_idx,:,:]
            plt.figure()
            plt.imshow(sample_img)
            plt.show()
            f.close()

    for p_test in test:
        with open(p_test,mode='rb') as f:
            letter_set = pickle.load(f)
            print ('size of test letter_set %s - %i' % (p_train,len(letter_set)))
            sample_idx = np.random.randint(len(letter_set))
            sample_img = letter_set[sample_idx,:,:]
            plt.figure()
            plt.imshow(sample_img)
            plt.show()
            f.close()

#sanity_check(train_datasets,test_datasets)

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
valid_size = 10000
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

def merge_datasets(pickle_files,train_size,valid_size=0):

    valid_dataset,valid_labels = make_arrays(valid_size,image_size,image_size)
    train_dataset,train_labels = make_arrays(train_size,image_size,image_size)

    num_classes = len(train_datasets)

    n_train_per_class = train_size/num_classes
    n_valid_per_class = valid_size/num_classes

    start_train,start_val = 0,0
    end_train,end_val = n_train_per_class,n_valid_per_class
    end_limit = n_train_per_class + n_valid_per_class

    for label,pickle_file in enumerate(pickle_files):
        with open(pickle_file,mode='rb') as f:
            letter_set = pickle.load(f)
            np.random.shuffle(letter_set) ## why do we do it?
            if valid_dataset is not None:
                letter_batch = letter_set[:n_valid_per_class,:,:]
                valid_dataset[start_val:end_val,:,:] = letter_batch
                valid_labels[start_val:end_val] = label

            letter_batch = letter_set[n_valid_per_class:end_limit,:,:]
            train_dataset[start_train:end_train] = letter_batch
            train_labels[start_train:end_train] = label

            f.close()

        start_train,start_val = end_train,end_val
        end_train,end_val = end_train + n_train_per_class, end_val + n_valid_per_class


    return valid_dataset,valid_labels,train_dataset,train_labels


valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)

_,_,test_dataset, test_labels = merge_datasets(test_datasets,test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

'''
Next, we'll randomize the data.
It's important to have the labels well shuffled for the training and test distributions to match.
'''
## randomization is done with numpy method permutation

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


'''
Problem 4
--------------------
Convince yourself that the data is still good after shuffling!
--------------------
'''
# show 10 random images and print their classes
check_data = False

if check_data:
    for i in range(10):
        rand = random.randint(0,len(train_dataset))
        plt.imshow(train_dataset[rand,:,:])
        print ('class: %i' % train_labels[rand])
        plt.show()



pickle_file = 'notMNIST.pickle'

if not os.path.exists(pickle_file):
    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

'''
---------------------------------- Problem 5 -----------------------------------------
By construction, this dataset might contain a lot of overlapping samples,
including training data that's also contained in the validation and test set!

Overlap between training and test can skew the results if you expect to use your
model in an environment where there is never an overlap, but are actually ok if you
expect to see training samples recur when you use it.

Measure how much overlap there is between training, validation and test samples.
Optional questions: - What about near duplicates between datasets?
(images that are almost identical) - Create a sanitized validation and test set
, and compare your accuracy on those in subsequent assignments.
--------------------------------------------------------------------------------------

'''

def check_overlapp():

    t1 = time.time()

    train_hashes = [hashlib.sha1(x).digest() for x in train_dataset]
    valid_hashes = [hashlib.sha1(x).digest() for x in valid_dataset]
    test_hashes = [hashlib.sha1(x).digest() for x in test_dataset]

    valid_in_train = np.in1d(valid_hashes, train_hashes)
    test_in_train = np.in1d(test_hashes, train_hashes)
    test_in_valid = np.in1d(test_hashes, valid_hashes)

    valid_keep = ~valid_in_train
    test_keep = ~(test_in_train | test_in_valid)

    valid_dataset_clean = valid_dataset[valid_keep]
    valid_labels_clean = valid_labels[valid_keep]

    test_dataset_clean = test_dataset[test_keep]
    test_labels_clean = test_labels[test_keep]

    t2 = time.time()

    print("Time: %0.2fs" % (t2 - t1))
    print("valid -> train overlap: %d samples" % valid_in_train.sum())
    print("test  -> train overlap: %d samples" % test_in_train.sum())
    print("test  -> valid overlap: %d samples" % test_in_valid.sum())

check_over = False
if check_over:
    check_overlapp()

'''
------------------------------- Problem 6 ------------------------------
Let's get an idea of what an off-the-shelf classifier can give you on this data.
It's always good to check that there is something to learn, and that it's a problem
that is not so trivial that a canned solution solves it. Train a simple model on this
data using 50, 100, 1000 and 5000 training samples. Hint: you can use the LogisticRegression
model from sklearn.linear_model. Optional question: train an off-the-shelf model on all the data!
 -----------------------------------------------------------------------

'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression()

for samples in [50,100,1000,5000]:

    train_dataset_reshape = train_dataset.reshape(train_dataset.shape[0],train_dataset.shape[1]*train_dataset.shape[2])[:samples,:]
    test_dataset_reshape = test_dataset.reshape(test_dataset.shape[0],test_dataset.shape[1]*test_dataset.shape[2])[:samples,:]

    clf.fit(X=train_dataset_reshape
        ,y=train_labels[:samples])

    pred = clf.predict(test_dataset_reshape)

    acc = accuracy_score(test_labels[:samples],pred)

    print ('accuracy for %i is %f' % (samples,acc))