# split the images file name and compare with bonafide -> correct
import csv
import glob
import os
from pathlib import Path
from shutil import rmtree, copy
import numpy as np
import math
import ipdb

from deepface import DeepFace
# from deepface.basemodels import Facenet512
from embeddings import get_embeddings


def distance_(embeddings):
    # Distance based on cosine similarity
    cos_similarity = np.dot(embeddings, embeddings.T)
    cos_similarity = cos_similarity.clip(min=0, max=1)

    return cos_similarity[0][1]


f = open('comparison_score_morphed2.csv', 'w', encoding='UTF8', newline='')
csv_save = csv.writer(f)
header = ['subject1', 'subject2', 'score']
csv_save.writerow(header)
folder_name = './root/morphed'  # path to the morphed images. The morphed images should be named like this: 1-vs-2.png
folder_name_2 = './root/bona'  # path to the bona fide images. The bona fide images should be name like this: 1.JPG
# folder_name = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\bologna2vs1'
# folder_name_2 = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\subject2'


if os.path.exists(os.path.join(os.getcwd(), 'temp')):
    rmtree('temp')
os.mkdir('temp')
for index, file in enumerate(os.listdir(folder_name)):
    source = file
    # print(source)
    morph_split = source.split('-vs')[0]
    print(morph_split)
    for file2 in os.listdir(folder_name_2):
        source2 = file2
        # file2 = file2.split('O-')[1]
        bona_split = file2.split('.JPG')[0]
        if morph_split == bona_split:
            root = 'temp\/' + str(index)
            os.mkdir(root)
            f1 = 'temp\/' + str(index) + '\similarity'
            os.mkdir(f1)
            img1_path = os.path.join(folder_name, source)
            print(img1_path)
            img2_path = os.path.join(folder_name_2, source2)
            print(img2_path)
            # obj = DeepFace.verify(img1_path, img2_path, model_name='ArcFace',
            #                       detector_backend='dlib', enforce_detection=False)
            # score = obj['distance']
            # data = [source, source2, score]
            # print(data)
            # csv_save.writerow(data)
            copy(img1_path, f1)
            copy(img2_path, f1)
            input_size = [112, 112]

            embeddings = get_embeddings(
                data_root=root,
                model_root="checkpoint/backbone_ir50_ms1m_epoch120.pth",
                input_size=input_size,
            )

            # ipdb.set_trace()
            data = [source, source2, distance_(embeddings)]
            print(data)
            csv_save.writerow(data)
