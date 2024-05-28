import random

from helper_tool import DataProcessing as DP
from helper_tool import ConfigSCANNETV2 as cfg
from os.path import join
import numpy as np
import os, pickle
import torch.utils.data as torch_data
import torch
from pathlib import Path
from helper_ply import read_ply


class Scannet(torch_data.Dataset):
    def __init__(self, mode):
        self.name = 'scannet'
        self.dataset_path = '/data/home/scv3159/run/scannet/'
        self.label_to_names = {0: "unannotated",
                               1: "wall",
                               2: "floor",
                               3: "chair",
                               4: "table",
                               5: "desk",
                               6: "bed",
                               7: "bookshelf",
                               8: "sofa",
                               9: "sink",
                               10: "bathtub",
                               11: "toilet",
                               12: "curtain",
                               13: "counter",
                               14: "door",
                               15: "window",
                               16: "shower curtain",
                               17: "refridgerator",
                               18: "picture",
                               19: "cabinet",
                               20: "otherfurniture"}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}

        self.loop = 10
        self.file_list = list(Path(self.dataset_path + 'input_0.040/').glob(f'*.ply'))
        self.mode = mode

        if mode == 'training':
            self.scene_id = [l.rstrip() for l in open(self.dataset_path + 'scannetv2_train.txt')]
            self.data_list = [f for f in self.file_list if f.stem[:12] in self.scene_id]
        elif mode == 'val':
            self.scene_id = [l.rstrip() for l in open(self.dataset_path + 'scannetv2_val.txt')]
            self.data_list = [f for f in self.file_list if f.stem[:12] in self.scene_id]

        self.data_idx = np.arange(len(self.data_list))
     #   print(sorted(self.data_list))
        self.possibility = []
        self.min_possibility = []

        for test_file_name in self.data_list:
            data = read_ply(test_file_name)
            feature = np.vstack((
                data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'], data['class']
            )).T
            self.possibility += [np.random.rand(feature.shape[0]) * 1e-3]
            # self.min_possibility += [float(np.min(self.possibility[-1]))]
        #
        # cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        cfg.class_weights = DP.get_class_weights_scannetv2()

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, item):

        selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind,selected_pc_aug,selected_colors_aug = self.spatially_regular_gen(item)
        return selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind,selected_pc_aug,selected_colors_aug

    def label_remap(self,label,percentage = '0.1%'):
        """

        :param label:n
        :return:
        """
        # label remap to add 0 for unlabeled data
        # label_map = label + 1

        for i in range(1,21):
            class_index = np.where(label == i)[0]
            num_of_class = len(class_index)
            if num_of_class > 0:
                if "%" in percentage:
                    ratio = float(percentage[:-1]) / 100
                    num_selected = max(int(num_of_class * ratio),1)
                else:
                    num_selected = int(percentage)
                label_index = list(range(num_of_class))
                random.shuffle((label_index))
                noselect_labels_indx = label_index[num_selected:]
                select_labels_indx = label_index[:num_selected]
                ind_class_noselect = class_index[noselect_labels_indx]
                ind_class_select = class_index[select_labels_indx]
                label[ind_class_noselect] = 0


        return label






    def spatially_regular_gen(self, item):
        # Generator loop

        cloud_idx = self.data_idx[item % len(self.data_idx)]
        pc_path = self.data_list[cloud_idx]
        sub_pc, sub_color, sub_tree, sub_labels = self.get_data(pc_path)
        pick_idx = np.argmin(self.possibility[cloud_idx]) if self.mode == 'training' else np.random.choice(len(sub_pc),1)
        selected_pc, selected_colors, selected_labels, selected_idx = self.crop_pc(sub_pc, sub_color, sub_tree,
                                                                                   sub_labels, pick_idx)
        dists = np.sum(np.square((selected_pc - sub_pc[pick_idx]).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[cloud_idx][selected_idx] += delta

        selected_pc_aug = self.augment(selected_pc,['noise','flip','rotate','scale'])
        selected_colors_aug = self.drop_color(selected_colors,droprate=0.2)


        # selected_pc_aug = selected_pc
        # selected_colors_aug = selected_colors
        # aug_num = np.random.choice([1,2,3],1,replace=False)
        # aug_type = np.random.choice([1,2,3],aug_num,replace=False)
        # if 1 in aug_type and 2 in aug_type:
        #     selected_pc_aug = self.augment(selected_pc, [ 'noise'])
        #     selected_pc_aug = self.augment(selected_pc_aug, ['flip'])
        # if 2 in aug_type and 1 not in aug_type:
        #     selected_pc_aug = self.augment(selected_pc, ['flip'])
        # if 1 in aug_type and 2 not in aug_type:
        #     selected_pc_aug = self.augment(selected_pc, ['noise'])
        # if 3 in aug_type:
        #     selected_colors_aug = self.drop_color(selected_colors,droprate=0.2)


       # un,counts = np.unique(selected_labels,return_counts=True)
#        print(un)
 #       print(counts)

        return selected_pc.astype(np.float32), selected_colors.astype(np.float32), \
           selected_labels.astype(np.int32), selected_idx.astype(np.int32), np.array([cloud_idx], dtype=np.int32),\
           selected_pc_aug.astype(np.float32), selected_colors_aug.astype(np.float32)


    def get_data(self, file_path):
        cloud_name = file_path.stem
        kd_tree_file = Path(self.dataset_path) / 'input_0.040' / '{:s}_KDTree.pkl'.format(cloud_name)
        sub_ply_file = Path(self.dataset_path) / 'input_0.040' / '{:s}.ply'.format(cloud_name)
        data = read_ply(sub_ply_file)
        feature = np.vstack((
            data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'], data['class']
        )).T

        sub_pc = feature[:, :3]
        sub_colors = feature[:, 3:6]
        sub_labels = feature[:, -1].copy()

        if self.mode == 'training':
            sub_labels = self.label_remap(sub_labels,percentage= '0.1%')
        else:
            sub_labels = sub_labels


        with open(kd_tree_file, 'rb') as f:
            search_tree = pickle.load(f)

        return sub_pc, sub_colors, search_tree, sub_labels

    @staticmethod
    def crop_pc(points, colors, search_tree, labels, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        center_point = center_point + noise.astype(center_point.dtype)

        if len(points) >= cfg.num_points:
            selected_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
        else:
            selected_idx = search_tree.query(center_point, k=len(points))[1][0]
        selected_idx = DP.shuffle_idx(selected_idx)
        selected_points = points[selected_idx]
        selected_colors = colors[selected_idx]
        selected_labels = labels[selected_idx]

        selected_points = selected_points - center_point

        if len(points) < cfg.num_points:
            selected_points, selected_colors, selected_idx, selected_labels = DP.data_aug(
                selected_points, selected_colors, selected_labels, selected_idx, cfg.num_points
            )

        return selected_points, selected_colors, selected_labels, selected_idx

    def tf_map(self, batch_pc, batch_colors, batch_label, batch_pc_idx, batch_cloud_idx,batch_pc_aug,batch_color_aug):

        features = np.concatenate((batch_pc, batch_colors,batch_pc_aug,batch_color_aug), axis=2)

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    def collate_fn(self, batch):

        selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind,selected_pc_aug,selected_colors_aug = [], [], [], [], [],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_colors.append(batch[i][1])
            selected_labels.append(batch[i][2])
            selected_idx.append(batch[i][3])
            cloud_ind.append(batch[i][4])
            selected_pc_aug.append(batch[i][5])
            selected_colors_aug.append(batch[i][6])

        selected_pc = np.stack(selected_pc)
        selected_colors = np.stack(selected_colors)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)
        selected_pc_aug = np.stack(selected_pc_aug)
        selected_colors_aug = np.stack(selected_colors_aug)

        flat_inputs = self.tf_map(selected_pc, selected_colors, selected_labels, selected_idx, cloud_ind,selected_pc_aug,selected_colors_aug)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())         #b,40960,k --b 10240,k--b,2560,k---b,640,k----6,160,k
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs

    @staticmethod
    def augment(xyz, methods):
        if 'rotate' in methods:
            angle = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(angle), np.sin(angle)
            R = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], R)

        if 'flip' in methods:
            direction = np.random.choice(4, 1)
            if direction == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif direction == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif direction == 3:
                xyz[:, :2] = -xyz[:, :2]

        if 'scale' in methods:
            s = np.random.uniform(0.95, 1.05)
            xyz[:, :3] = s * xyz[:, :3]

        if 'nbscale' in methods:
            scale = np.array([0.9, 1.1])
            scale_min, scale_max = np.min(scale), np.max(scale)
            scale = np.random.rand(3) * (scale_max - scale_min) + scale_min
            # symmetric = np.random.rand(3) * 2 - 1
            xyz[:, :3] = xyz[:, :3] * scale

        if 'noise' in methods:
            noise = np.array([np.random.normal(0, 0.1, 1),
                              np.random.normal(0, 0.1, 1),
                              np.random.normal(0, 0.1, 1)]).T
            xyz[:, :3] += noise
        if 'floorcenter' in methods:
            xyz -= np.mean(xyz, axis=0, keepdims=True)
            xyz[:, 2] -= np.min(xyz[:, 2])

        return xyz

    @staticmethod
    def drop_color(color, droprate=0.2):
        drop = np.random.rand(1) < droprate
        if drop:
            color[:, :3] = 0
        return color

    @staticmethod
    def color_autocontrast(color, factor=0.5):
        if np.random.rand(1) < 0.2:
            if color.mean() <= 0.1:
                return color
            lo = color.min(1, keepdims=True)[0]
            hi = color.max(1, keepdims=True)[0]
            scale = 1 / (hi - lo + 1e-8)
            contrast_feats = (color - lo + 1e-8) * scale
            color = (1 - factor) * color + factor * contrast_feats
        return color

    @staticmethod
    def color_normalize(color):
        mean = [0.5136457, 0.49523646, 0.44921124]
        std = [0.18308958, 0.18415008, 0.19252081]
        color = (color - np.array(mean)) / np.array(std)
        return color

if __name__ =="__main__":
    from torch.utils.data import DataLoader
    dataset = S3DIS(mode='training',test_id=5)
    dataloader = DataLoader(dataset,batch_size=10,collate_fn=dataset.collate_fn)
    for i,data in enumerate(dataloader):
        print(type(data))
        neigh_idx = data['features']
        print(neigh_idx.shape)
        # print(len(neigh_idx))
        # for j in range(len(neigh_idx)):
        #     print(neigh_idx[j].shape)
    # print(type(out))
