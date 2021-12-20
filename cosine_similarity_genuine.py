import os
import csv
import numpy as np
import math
from shutil import copy, rmtree
import torch

import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from backbone import Backbone
from tqdm import tqdm
# from pathlib import Path
from deepface import DeepFace

from embeddings import get_embeddings


def distance_(embeddings0):
    # Distance based on cosine similarity
    cos_similarity = np.dot(embeddings, embeddings.T)
    cos_similarity = cos_similarity.clip(min=0, max=1)

    return cos_similarity[0][1]


f = open('comparison_score_genuine.csv',  'w',  encoding='UTF8', newline='')
csv_save = csv.writer(f)
header = ['subject1', 'subject2', 'score']
csv_save.writerow(header)

folder_name = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\subject1'
folder_name_2 = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\subject2'

if os.path.exists(os.path.join(os.getcwd(), 'temp')):
    rmtree('temp')
os.mkdir('temp')

for index, file in enumerate(os.listdir(folder_name)):
    source_1 = file
    for file2 in os.listdir(folder_name_2):
        source_2 = file2.split('O-')[1]
        if source_2 == source_1:
            root = 'temp\/' + str(index)
            os.mkdir(root)
            f1='temp\/' + str(index)+ '\cat'
            f2 = 'temp\/' + str(index)+ '\dog'
            os.mkdir(f1)
            os.mkdir(f2)

            # print(source_2, source_1)
            img1_path = os.path.join(folder_name, source_1)
            img2_path = os.path.join(folder_name_2, file2)
            print(img1_path)
            print(img2_path)
            copy(img1_path, f1)
            copy(img2_path, f2)
            input_size = [112, 112]

            embeddings = get_embeddings(
                data_root=root,
                model_root="checkpoint/backbone_ir50_ms1m_epoch120.pth",
                input_size=input_size,
            )
            data = [source_1, source_2, distance_(embeddings)]
            print(data)
            # print(data)
            csv_save.writerow(data)
            # obj = DeepFace.verify(img1_path, img2_path, model_name='ArcFace',
            #                       detector_backend='dlib', enforce_detection=False)
            # score = obj['distance']
            # data = [source_1, source_2, score]
            # print(data)
            # csv_save.writerow(data)