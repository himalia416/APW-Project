import csv
import glob
import os
from pathlib import Path
from shutil import rmtree, copy
import numpy as np
import math
import ipdb
from embeddings import get_embeddings

def distance_(embeddings0):
    # Distance based on cosine similarity
    cos_similarity = np.dot(embeddings, embeddings.T)
    cos_similarity = cos_similarity.clip(min=0, max=1)

    return cos_similarity[0][1]


f = open('comparison_score_imposter_v1.csv',  'w',  encoding='UTF8', newline='')
csv_save = csv.writer(f)
header = ['subject1', 'subject2', 'score']
csv_save.writerow(header)

folder_name = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\MIPGAN1vs2'
folder_name_2 = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\subject1'
folder_name_3 = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\subject2'

if os.path.exists(os.path.join(os.getcwd(), 'temp')):
    rmtree('temp')
os.mkdir('temp')
for index, file in enumerate(os.listdir(folder_name)):
    source = file
    # print(source)
    after_split = source.replace('.jpg', '').split('-vs-')
    print(after_split[0], after_split[1])
    for file2 in os.listdir(folder_name_2):
        source2 = file2
        after_split_subject = source2.split('.JPG')[0]
        if after_split_subject == after_split[0]:
            subject_1 = source2
            # make directory for each images
            root = 'temp\/' + str(index)
            os.mkdir(root)
            f1 = 'temp\/' + str(index) + '\cat'
            os.mkdir(f1)
            img1_path = os.path.join(folder_name_2, subject_1)
            print(img1_path)
            img2_path = os.path.join(folder_name_3, after_split[1]+'.JPG')
            print(img2_path)
            copy(img1_path, f1)
            copy(img2_path, f1)
            input_size = [112, 112]

            embeddings = get_embeddings(
                data_root=root,
                model_root="checkpoint/backbone_ir50_ms1m_epoch120.pth",
                input_size=input_size,
            )

            # obj = DeepFace.verify(img1_path, img2_path, model_name='ArcFace', detector_backend='dlib', enforce_detection=False)
            # score = obj['distance']
            data = [after_split[0]+'.JPG', after_split[1]+'.JPG', distance_(embeddings)]
            print(data)
            csv_save.writerow(data)