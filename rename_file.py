import os
import csv

source_path = './root/morphed'
destination_path = './root/renamed morphed'
# path = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\subject2'
# tmp_path = r'C:\Users\admin\PycharmProjects\pythonProject2\data\new_aligned\subject2'

# provide a csv file which should have the old_image_name in first column and new_image_name in second column

with open('rename_subject2.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        old_name = os.path.join(source_path, row[0])
        if os.path.exists(old_name):
            new_name = os.path.join(destination_path, row[1])
            os.rename(old_name, new_name)
