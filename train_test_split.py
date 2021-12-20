import csv
import os
import random
import shutil

img_path = r'./root/morphed_images'

train, test, train_temp, test_temp = [], [], [], []

images = os.listdir(img_path)
print(images)


def splitter_bologna(path):
    return path.replace('O-', '')


def splitter(path):
    return path.replace('.png', '').split('-vs-')


def any_value_exists(common, train_image):
    for i in common:
        train_items = splitter(train_image)
        if i in train_items:
            return train_image
    return None


def randomize(image_list):
    return random.sample(image_list, len(image_list))


# randomize the image files so that each time the test train split will have a distinct images
images = randomize(images)
print(images)
# first image
train.append(images[0])
train_temp.extend(splitter(images[0]))

for image in images[1:]:
    left, right = splitter(image)
    if (left not in train_temp) and (right not in train_temp):
        if (left in test_temp) or (right in test_temp):
            test_temp.extend([left, right])
            test.append(image)
        else:
            split_ratio = len(test) / len(train)
            if split_ratio < 2 / 3:
                # test is smaller
                test_temp.extend([left, right])
                test.append(image)
            else:
                # train is smaller
                train_temp.extend([left, right])
                train.append(image)
    else:
        train_temp.extend([left, right])
        train.append(image)

print(train_temp, '\n', test_temp)
# common issue
commons = list(set(train_temp).intersection((set(test_temp))))
common_test = []
common_train = []
for i in train:
    exists = any_value_exists(commons, i)
    if exists:
        common_train.append(exists)
        # train_temp.extend(splitter(exists))

for i in test:
    exists = any_value_exists(commons, i)
    if exists:
        common_test.append(exists)
        # train_temp.extend(splitter(exists))

train.extend(common_test)
# train_temp.ex
for i in common_test:
    test.remove(i)
    # test_temp.remove(splitter(i))


# open two csv files
f1 = open('train_data.csv', 'w', encoding='UTF8', newline='')
train_data_save = csv.writer(f1)
f2 = open('test_data.csv', 'w', encoding='UTF8', newline='')
test_data_save = csv.writer(f2)
for train_value in train:
    train_data_save.writerow([train_value])
for test_value in test:
    test_data_save.writerow([test_value])

# save the image in destination to test and train
# before saving clear the destination folder
if os.path.exists(os.path.join(os.getcwd(), 'temp')):
    shutil.rmtree('train_test')
os.mkdir('train_test')

train_destination = 'train_test\/train\/morph'
os.mkdir(train_destination)
test_destination = 'train_test\/test\/morph'
os.mkdir(train_destination)


# if the image is in train, save it in the train folder
for img in os.listdir(img_path):
    if img in train:
        print(img)
        shutil.copy((os.path.join(img_path, img)), train_destination)


# If the image is test, save it in the test folder
for img in os.listdir(img_path):
    if img in test:
        print(img)
        shutil.copy((os.path.join(img_path, img)), test_destination)

# For the bona fide image, check the file name and check if its in train set or test set
bonafide_img_path = r'.\/root\/bonafide'
train_bonafide_destination = 'train_test\/train\/bona'
os.mkdir(train_bonafide_destination)
test_bonafide_destination = 'train_test\/test\/bona'
os.mkdir(test_bonafide_destination)
bonafide_image = os.listdir(bonafide_img_path)

for bona in bonafide_image:
    for i in train:
        # print("this is i",i)
        l, r = splitter(i)
        b = bona.replace('.JPG', '')
        # print(l, r)
        if b in [l, r]:
            # print(b)
            print(bona)
            # check if the file is already there
            if bona not in train_bonafide_destination:
                shutil.copy((os.path.join(bonafide_img_path, bona)), train_bonafide_destination)

# For the bona fide image, check the file name and check if its in train set or test set
for bona in bonafide_image:
    for i in test:
        # print("this is i",i)
        l, r = splitter(i)
        b = bona.replace('.JPG', '')
        # print(l, r)
        if b in [l, r]:
            # print(b)
            print(bona)
            # check if the file is already there
            if bona not in test_bonafide_destination:
                shutil.copy((os.path.join(bonafide_img_path, bona)), test_bonafide_destination)


