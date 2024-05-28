# Common
import os
import yaml
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# my module
from helper_tool import DataProcessing as DP
from helper_tool import ConfigSCANNETV2 as cfg
from scantest import scannettest
from RandLANet import Network
import pickle
from pathlib import Path


np.random.seed(0)
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--infer_type', default='all', type=str, choices=['all', 'sub'], help='Infer ALL or just infer Subsample')
parser.add_argument('--checkpoint_path', default='./log/2022-08-15-22-26randla_cos100epoch_possidataset/checkpoint.tar')
parser.add_argument('--test_mode', default='val', type=str, help='Predicted sequence id [default: 08]')
parser.add_argument('--result_dir', default='results/randla_possi', help='Dump dir to save prediction [default: result/]')
parser.add_argument('--yaml_config', default='utils/semantic-kitti.yaml', help='semantic-kitti.yaml path')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 30]')
parser.add_argument('--index_to_label', action='store_true',
                    help='Set index-to-label flag when inference / Do not set it on seq 08')
FLAGS = parser.parse_args()


label_to_names = {0: "unannotated",
                               1: "wall",
                               2: "floor",
                               3: "cabinet",
                               4: "bed",
                               5: "chair",
                               6: "sofa",
                               7: "table",
                               8: "door",
                               9: "window",
                               10: "bookshelf",
                               11: "picture",
                               12: "counter",
                               13: "desk",
                               14: "curtain",
                               15: "refridgerator",
                               16: "shower curtain",
                               17: "toilet",
                               18: "sink",
                               19: "bathtub",
                               20: "otherfurniture"}

names_to_label = {
"wall":1 ,
"floor":2,
"cabinet":3,
"bed":4,
"chair":5,
"sofa":6,
"table":7,
"door":8,
"window":9,
"bookshelf":10,
"picture":11,
"counter":12,
"desk":14,
"curtain":16,
"refridgerator":24,
"shower curtain":28,
"toilet":33,
"sink":34,
"bathtub":36,
"otherfurniture":39
}

(152, 223, 138),  # floor
(174, 199, 232),  # wall
(31, 119, 180),  # cabinet
(255, 187, 120),  # bed
(188, 189, 34),  # chair
(140, 86, 75),  # sofa
(255, 152, 150),  # table
(214, 39, 40),  # door
(197, 176, 213),  # window
(148, 103, 189),  # bookshelf
(196, 156, 148),  # picture
(23, 190, 207),  # counter
(247, 182, 210),  # desk
(219, 219, 141),  # curtain
(255, 127, 14),  # refrigerator
(227, 119, 194),  # bathtub
(158, 218, 229),  # shower curtain
(44, 160, 44),  # toilet
(112, 128, 144),  # sink
(82, 84, 163),  # otherfurn



class Tester:
    def __init__(self):
        # Init Logging
        os.makedirs(FLAGS.result_dir, exist_ok=True)
        log_fname = os.path.join(FLAGS.result_dir, 'log_test.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        self.logger = logging.getLogger("Tester")

        # load yaml file
        # self.remap_lut = self.load_yaml(FLAGS.yaml_config)

        # get_dataset & dataloader
        test_dataset = scannettest(mode = FLAGS.test_mode, batch_size=FLAGS.batch_size)

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=None,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            num_workers=0
        )
        # Network & Optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Network(cfg)
        self.net.to(device)

        # Load module
        print('begin loading training parameter########################3')
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None :
            print('loading training parameter!!!!!!!!!!!!!!!!!!!!1')
            checkpoint = torch.load(CHECKPOINT_PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'])

        # Multiple GPU Testing
        if torch.cuda.device_count() > 1:
            self.logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = nn.DataParallel(self.net)

        self.test_dataset = test_dataset
        # Initialize testing probability
        self.test_dataset.init_prob()
        self.test_probs = self.init_prob()
        self.test_smooth = 0.98


    def load_yaml(self, path):
        DATA = yaml.safe_load(open(path, 'r'))
        # get number of interest classes, and the label mappings
        remapdict = DATA["learning_map_inv"]
        # make lookup table for mapping
        maxkey = max(remapdict.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        return remap_lut

    def init_prob(self):
        probs = []
        for item in self.test_dataset.possibility:
            prob = np.zeros(shape=[len(item), self.test_dataset.num_classes], dtype=np.float32)
            probs.append(prob)
        return probs

    def test(self):
        self.logger.info("Start Testing")
        self.rolling_predict()
        # Merge Probability
        self.merge_and_store()

    def rolling_predict(self):
        self.net.eval()  # set model to eval mode (for bn and dp)

        iter_loader = iter(self.test_loader)
        with torch.no_grad():
            min_possibility = self.test_dataset.min_possibility #x选取最小的possi，当作当前测试的frame
            while np.min(min_possibility) <= 0.5:
              #  print('min possi is -- ',np.min(min_possibility))
                batch_data, input_inds, cloud_inds, min_possibility = next(iter_loader)
                for i in range(len(cloud_inds)):
                    self.logger.info('cloud indx is: %d'%cloud_inds[i])
                    self.logger.info('min_poss is : %f'%np.min(min_possibility))
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)
                # Forward pass
        #        torch.cuda.synchronize()
                end_points = self.net(batch_data)
                end_points['logits'] = end_points['logits'].transpose(1, 2)  # b,n,c
                # update prediction (multi-thread)
                self.update_predict(end_points, batch_data, input_inds, cloud_inds)

    def update_predict(self, end_points, batch_data, input_inds, cloud_inds):
        # Store logits into list
        B = end_points['logits'].size(0)
        end_points['logits'] = end_points['logits'].cpu().numpy()
        for j in range(B):
            probs = end_points['logits'][j]
            inds = input_inds[j]
            c_i = cloud_inds[j][0]
            self.test_probs[c_i][inds] = \
                self.test_smooth * self.test_probs[c_i][inds] + (1 - self.test_smooth) * probs

    def merge_and_store(self):
        # initialize result directory
        root_dir = os.path.join(FLAGS.result_dir, str(self.test_dataset.test_mode), 'predictions')
        os.makedirs(root_dir, exist_ok=True)
        self.logger.info(f'mkdir {root_dir}')
        N = len(self.test_probs)
        for j in tqdm(range(N)):
            if FLAGS.infer_type == 'all':
                proj_file_name = self.test_dataset.data_list[j].split('/')[-1].split('.')[0]
                if FLAGS.test_mode == 'val':
                    proj_file = Path(self.test_dataset.dataset_path+'input_0.040/') / '{:s}_proj.pkl'.format(proj_file_name)
                else:
                    proj_file = Path(self.test_dataset.dataset_path + 'input_test_0.040/') / '{:s}_proj.pkl'.format(
                        proj_file_name)
                if os.path.isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds,alllabels = pickle.load(f)
                probs = self.test_probs[j][proj_inds, :]
                pred = np.argmax(probs, 1).astype(np.uint32)
            elif FLAGS.infer_type == 'sub':
                pred = np.argmax(self.test_probs[j], 1).astype(np.uint32)
            else:
                raise TypeError("Choose what you want to infer")
            if FLAGS.test_mode == 'val':
                name = self.test_dataset.data_list[j].split('/')[-1].split('.')[0] + '_pred.npy'
            else:
                name = self.test_dataset.data_list[j].split('/')[-1].split('.')[0][:-11] + '.txt'
            # pred_file = Path(self.test_dataset.dataset_path) / '{:s}_pred.pkl'.format(proj_file_name)
            # name = name + '_pred.npy'
            output_path = os.path.join(root_dir, name)
            if FLAGS.test_mode == 'val':
                np.save(output_path, pred)
            else:
                namelist = [label_to_names[p] for p in pred]
                labellist = [names_to_label[n] for n in namelist]
                labelnumpy = np.asarray(labellist)
                # np.set_printoptions(suppress=True)
                np.savetxt(output_path,labelnumpy,fmt='%d')

    def remap(self, label):
        upper_half = label >> 16      # get upper half for instances
        lower_half = label & 0xFFFF   # get lower half for semantics
        lower_half = self.remap_lut[lower_half]  # do the remapping of semantics
        label = (upper_half << 16) + lower_half   # reconstruct full label
        label = label.astype(np.uint32)
        return label


def main():
    tester = Tester()
    tester.test()


if __name__ == '__main__':
    main()
