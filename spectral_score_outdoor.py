import csv
import glob
import os
from pathlib import Path
from shutil import rmtree, copy

from deepface import DeepFace
from deepface.basemodels import Facenet512
from embeddings import get_embeddings
import numpy as np


def distance_(embeddings):
    # Distance based on cosine similarity
    cos_similarity = np.dot(embeddings, embeddings.T)
    cos_similarity = cos_similarity.clip(min=0, max=1)

    return cos_similarity[0][1]


f = open('spectral_score_outdoor.csv', 'w', encoding='UTF8', newline='')
csv_save = csv.writer(f)
header = ['subject1', 'subject2', 'score']
csv_save.writerow(header)
# folder_name = r'C:\Users\admin\OneDrive\Desktop\For comparison\images'
folder_name = r'/media/user1/Himali/generated_images-subject1vs2-MIPGAN-I'
folder_name_2 = r'/media/user1/KIRAN/spectral-morphing-attack-detection/Outdoor/'


# Check if the temp file exist or not, if exist delete it and create a new temp folder
if os.path.exists(os.path.join(os.getcwd(), 'temp')):
    rmtree('temp')
os.mkdir('temp')


for index, file in enumerate(os.listdir(folder_name)):
    source = file
    after_split = source.split('-vs')[0]
    for file2 in os.listdir(folder_name_2):
        if file2.startswith('subject-'):
            source2 = file2
            # print(source2)
            after_split_subject = source2.split('subject-')[1]
            # print(after_split_subject)
            if after_split_subject == after_split:
                # Create a new folder inside the temp folder
                root = 'temp/' + str(index)
                os.mkdir(root)
                f1 = 'temp/' + str(index) + '/cat'
                os.mkdir(f1)

                # Assign the after split subject value to subject for the comparison
                subject = after_split_subject
                imga2_path = os.path.join(folder_name_2, 'subject-' + after_split_subject)
                print(imga2_path)
                temp = ["530"]
                # temp = ["530", "590", "650", "710", "770", "830", "890", "950", "1000", "whole light"]
                for i in range(len(temp)):
                    temp2 = [r"/non_spec", "/spec"]
                    # print(temp[i])
                    for j in range(len(temp2)):
                        image2_path = os.path.join(imga2_path, temp[i] + temp2[j])
                        # print(image2_path)
                        for path in os.listdir(image2_path):
                            if path.endswith('.bmp'):
                                image_name = path
                                img1_path = os.path.join(folder_name, source)
                                # print(img1_path)
                                img2_path = os.path.join(image2_path, image_name)
                                # put the images to the folder and find the embeddings of each images
                                copy(img1_path, f1)
                                copy(img2_path, f1)
                                input_size = [112, 112]

                                embeddings = get_embeddings(
                                    data_root=root,
                                    model_root="checkpoint/backbone_ir50_ms1m_epoch120.pth",
                                    input_size=input_size,
                                )
                                # Find the distance of each images
                                # ipdb.set_trace()
                                data = [source, source2, distance_(embeddings)]
                                print(data)
                                csv_save.writerow(data)
                break