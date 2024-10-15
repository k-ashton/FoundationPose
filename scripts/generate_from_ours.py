import cv2
import argparse
import os
import numpy as np

import shutil
import glob
import json

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()

    root_path = 'demo_data/'+args.dataset_name + '/'
    os.makedirs('demo_data', exist_ok=True)
    os.makedirs(root_path, exist_ok=True)
    for fd in ['rgb','depth']:
        os.makedirs(root_path+fd, exist_ok=True)

    for fn in glob.glob(args.data_folder+'images/*'):
        shutil.copy(fn, root_path+'rgb/')

    for fn in glob.glob(args.data_folder+'monodepths/*'):
        shutil.copy(fn, root_path+'depth/')

    with open(args.data_folder+'transforms.json', 'r') as f:
        traj_data = json.load(f)

    with open(root_path + "cam_K.txt", "w") as f:
        f.write(f"{traj_data['fl_x']} 0 {traj_data['cx']}\n")
        f.write(f"0 {traj_data['fl_y']} {traj_data['cy']}\n")
        f.write(f"0 0 1\n")