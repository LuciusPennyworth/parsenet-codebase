import os
import h5py
import numpy as np

root = "/home/zhuhan/Code/ProjectMarch/dataset/hpnet/"
filename = 'val_data.txt'
# save_path =
data_path = open(os.path.join(root, filename), 'r')

data_list = [item.strip() for item in data_path.readlines()]

all_points = []
all_normals = []
all_labels = []
all_primitives = []
all_params = []

for i, idx in enumerate(data_list):
    if i % 120 == 0:
        print(i, "/", len(data_list))
    data_file = os.path.join(root, idx + '.h5')
    with h5py.File(data_file, 'r') as hf:
        points = np.array(hf.get("points"))
        labels = np.array(hf.get("labels"))
        normals = np.array(hf.get("normals"))
        primitives = np.array(hf.get("prim"))
        primitive_param = np.array(hf.get("T_param"))

    all_points.append(points)
    all_labels.append(labels)
    all_normals.append(normals)
    all_primitives.append(primitives)
    all_params.append(primitive_param)

with h5py.File(os.path.join("/home/zhuhan/Code/ProjectMarch/dataset/new_parsenet/val_data.h5"), 'w') as wf:
    wf.create_dataset('labels', data=np.stack(all_labels, 0))
    wf.create_dataset('prim', data=np.stack(all_primitives, 0))
    wf.create_dataset('points', data=np.stack(all_points, 0))
    wf.create_dataset('normals', data=np.stack(all_normals, 0))
    wf.create_dataset('param', data=np.stack(all_params, 0))
