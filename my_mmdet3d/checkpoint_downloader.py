import os
import glob
import mim
import yaml
from yaml import load, Loader
from termcolor import colored
import requests
import wget
import csv

file_list=glob.glob("./configs/*/metafile.yml")
my_path="./configs/"

checkpoints=[]
if not os.path.isdir(my_path):
    os.mkdir(my_path)
for file in file_list:
    stream = open(file, 'r')
    dictionary = yaml.load(stream, Loader=Loader)
    for i in range(len(dictionary["Models"])):
        dest = my_path+file.split("/")[2]
        try:
            filename = wget.download(dictionary["Models"][i]["Weights"], out=dest)
            file_path = dest+"/"+filename
            checkpoints.append([dictionary["Models"][i]["In Collection"],dictionary["Models"][i]["Name"],file_path])
        except: 
            print("\n"+colored(("Can not find:"), "light_red"), colored(dictionary["Models"][i]["Name"],"light_red"))

with open('checkpoints.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["Collection", "Name", "Checkpoint"])
    write.writerows(checkpoints)
    
mim download mmdet3d --config dgcnn_4xb32-cosine-100e_s3dis-seg_test-area2.py --dest .