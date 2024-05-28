import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix
from pointnet2_utils import index_points
from projector import *

class Network(nn.Module):

    def __init__(self, config,is_aug = True,project_size = 256):
        super().__init__()
        self.config = config
        self.is_aug = is_aug
        self.class_weights = DP.get_class_weights_s3dis()


        self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)


        self.decoder_blocks = nn.ModuleList()

        for j in range(self.config.num_layers):
            if j < 4:
                d_in = d_out + 2 * self.config.d_out[-j - 2]
                d_out = 2 * self.config.d_out[-j - 2]
            else:
                d_in = 4 * self.config.d_out[-5]
                d_out = 2 * self.config.d_out[-5]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1), bn=False, activation=None)

        #projection head
        self.project_head = ProjectionV1(32, 32)



    def forward(self, end_points):
        if self.training:
            if self.is_aug:
                features = end_points['features'][:,6:,:]  # training set: student:strong aug
            else:
                features = end_points['features'][:,:6,:]  # traing set : teacher: no aug
        else:
            features = end_points['features'][:, :6, :]  # val set: no dataaug
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])
        # project_features = self.project_head(features)

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################
        #   project features1
        # project_features = features
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        # project features2
        project_features_m = features
        project_features_p = self.project_head(project_features_m)
        features = self.fc3(features)
        f_out = features.squeeze(3)
        nei_idx = end_points['neigh_idx'][0] # b,40960,k

        # end_points['logits'] = f_out
        return project_features_p.squeeze(-1),project_features_m.squeeze(-1),f_out,end_points['labels'],nei_idx

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features



def compute_acc(valid_logits,valid_labels):

    logits = valid_logits
    labels = valid_labels
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])

    return acc

def compute_acc_list(valid_logits_list,valid_labels_list):
    acc = 0
    for i in range(len(valid_labels_list)):
        logits = valid_logits_list[i]
        labels = valid_labels_list[i]
        logits = logits.max(dim=1)[1]
        acc = (logits == labels).sum().float() / float(labels.shape[0]) + acc

    return acc / len(valid_labels_list)


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, valid_logits,valid_labels):
        logits = valid_logits
        labels = valid_labels
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, labels = np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list

class IoUCalculator_list:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, valid_logits_list,valid_labels_list):
        for i in range(len(valid_labels_list)):
            logits = valid_logits_list[i]
            labels = valid_labels_list[i]
            pred = logits.max(dim=1)[1]
            pred_valid = pred.detach().cpu().numpy()
            labels_valid = labels.detach().cpu().numpy()

            val_total_correct = 0
            val_total_seen = 0

            correct = np.sum(pred_valid == labels_valid)
            val_total_correct += correct
            val_total_seen += len(labels_valid)

            conf_matrix = confusion_matrix(labels_valid, pred_valid, labels = np.arange(0, self.cfg.num_classes, 1))
            self.gt_classes += np.sum(conf_matrix, axis=1)
            self.positive_classes += np.sum(conf_matrix, axis=0)
            self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list

class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = torch.softmax(p,dim=-1)
        q = torch.softmax(q,dim=-1)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        m_inf = m.isinf()
        if m_inf.sum() > 0:
            print('m is inf--------------------------------')
        p_log = p.log().isinf()
        if p_log.sum() > 0:
            print('p_log is inf !!!!!!!!!!!!!!!!!!!!!!!!!')
        q_log = q.log().isinf()
        if q_log.sum() > 0:
            print('q_log is inf ******************************')
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def consistency_loss_KL(logits_s,logits_t,b,n,c):
    #test batchmean and mean
    critrion = nn.KLDivLoss(reduction='batchmean')
    logits_s = logits_s.log_softmax(-1)
    logits_t = logits_t.softmax(-1)

    # logits_s = logits_s.view(b,-1,c).contiguous(
    # logits_t = logits_t.view(b,-1,c).contiguous()
    loss = critrion(logits_s,logits_t)
    return loss

def consistency_loss_KL_list(logits_s_list,logits_t_list,b,n,c):
    #test batchmean and mean
    critrion = nn.KLDivLoss(reduction='batchmean')
    logits_s_log_list = [logits_s.log_softmax(-1) for logits_s in logits_s_list]
    logits_t_soft_list = [logits_t.softmax(-1) for logits_t in logits_t_list]
    loss = 0
    for i in range(b):
        loss  = loss + critrion(logits_s_log_list[i],logits_t_soft_list[i])

    return loss / n


def consistency_loss_JSD_list(logits_s_list,logits_t_list,b,n,c):
    #test batchmean
    critrion = JSD()
    loss = 0
    for i in range(b):
        loss  = loss + critrion(logits_s_list[i],logits_t_list[i])

    return loss / b


def consistency_loss_JSD_list_neisimi(simi_s_list,simi_t_list,logits_s_list,logits_t_list,b,n,c):
    #test batchmean and mean
    critrion = JSD()

    loss = 0
    for i in range(b):

        # loss  = loss + critrion(logits_s_list[i],logits_t_list[i])

        loss = loss + critrion(simi_s_list[i],simi_t_list[i])

    return loss / b

def consistency_loss_JSD_list_gaussian(gaussian_s_list,gaussian_t_list,logits_s_list,logits_t_list,b,n,c):
    #test batchmean and mean
    critrion = JSD()

    loss = 0
    for i in range(b):

        # loss  = loss + critrion(logits_s_list[i],logits_t_list[i])

        loss = loss + critrion(gaussian_s_list[i],gaussian_t_list[i])

    return loss / b

def consistency_loss_MSE_list_gaussian(gaussian_s_list,gaussian_t_list,logits_s_list,logits_t_list,b,n,c):
    #test batchmean and mean
    critrion = nn.MSELoss()

    loss = 0
    for i in range(b):

        # loss  = loss + critrion(logits_s_list[i],logits_t_list[i])

        loss = loss + critrion(gaussian_s_list[i],gaussian_t_list[i])

    return loss / b


def consistency_loss_L1_list_gaussian(gaussian_s_list,gaussian_t_list,logits_s_list,logits_t_list,b,n,c):
    #test batchmean and mean
    critrion = nn.L1Loss()

    loss = 0
    for i in range(b):

        # loss  = loss + critrion(logits_s_list[i],logits_t_list[i])

        loss = loss + critrion(gaussian_s_list[i],gaussian_t_list[i])

    return loss / b

def consistency_loss_KL_list_neisimi(simi_s_list,simi_t_list,logits_s_list,logits_t_list,b,n,c):
    #test batchmean and mean
    critrion = nn.KLDivLoss(reduction='batchmean')
    logits_s_log_list = [logits_s.log_softmax(-1) for logits_s in logits_s_list]
    logits_t_soft_list = [logits_t.softmax(-1) for logits_t in logits_t_list]

    simi_s_log_list = [simi_s.log_softmax(-1) for simi_s in simi_s_list]  # [[n,k].....]b
    simi_t_soft_list = [simi_t.softmax(-1) for simi_t in simi_t_list]

    loss = 0
    for i in range(b):
        # loss  = loss + critrion(logits_s_log_list[i],logits_t_soft_list[i])
        loss = loss + critrion(simi_s_log_list[i],simi_t_soft_list[i])

    return loss / b


def consistency_loss_KL_list_gaussian(gaussian_s_list,gaussian_t_list,logits_s_list,logits_t_list,b,n,c):
    #test batchmean and mean
    critrion = nn.KLDivLoss(reduction='batchmean')
    logits_s_log_list = [logits_s.log_softmax(-1) for logits_s in logits_s_list]
    logits_t_soft_list = [logits_t.softmax(-1) for logits_t in logits_t_list]

    gaussian_s_log_list = [simi_s.log_softmax(-1) for simi_s in gaussian_s_list]  # [[n,k,c].....]b
    gaussian_t_soft_list = [simi_t.softmax(-1) for simi_t in gaussian_t_list]

    loss = 0
    for i in range(b):
        # loss  = loss + critrion(logits_s_log_list[i],logits_t_soft_list[i])
        loss = loss + critrion(gaussian_s_log_list[i],gaussian_t_soft_list[i])

    return loss / b

def compute_neigh_meanandstd_s3dis(logits,logits_t,nei_idx):
    """

    :param logits: b,c,n
    :param logits_t: b,c,n
    :param nei_idx: b,n,k
    :return:
    """
    neig_s = index_points(logits.permute(0,2,1),nei_idx) # b,n,k,c
    neig_t = index_points(logits_t.permute(0,2,1),nei_idx) # b,n,k,c
    b,n,k,c = neig_t.shape
    #local mean
    neig_s_mean = torch.mean(neig_s,dim=-2,keepdim=True) # b,n,1,c
    neig_t_mean = torch.mean(neig_t,dim=-2,keepdim=True) # b,n,1,c
    #local std
    # nei_s_std = torch.std((neig_s - neig_s_mean).reshape(b,n,-1),dim=-1,keepdim=True)   # b,n,k,c--b,n,c--b,n,k,1
    # nei_t_std = torch.std((neig_t - neig_t_mean).reshape(b,n,-1),dim=-1,keepdim=True)
    # global std
    nei_s_std = torch.std((neig_s - neig_s_mean).reshape(b,  -1), dim=-1, keepdim=True)  # b,n,k,c--b,n,c--b,n,k,1
    nei_t_std = torch.std((neig_t - neig_t_mean).reshape(b,  -1), dim=-1, keepdim=True)

    criterion = torch.nn.MSELoss()
    loss = criterion(nei_s_std,nei_t_std) + criterion(neig_s_mean,neig_t_mean)
    return loss

def compute_neigh_simi_s3dis(logits,logits_t,nei_idx):
    """

    :param logits: b,c,n
    :param logits_t: b,c,n
    :param nei_idx: b,n,k
    :return:
    """
    logits_t = logits_t.detach()
    neig_s = index_points(logits.permute(0,2,1),nei_idx) # b,n,k,c
    neig_t = index_points(logits_t.permute(0,2,1),nei_idx) # b,n,k,c
    b,n,k,c = neig_t.shape

    neig_s_nor = F.normalize(neig_s,dim=-1)  # b,n,k,c
    neig_t_nor = F.normalize(neig_t,dim=-1)  # b,n,k,c

    logits_nor = F.normalize(logits.permute(0,2,1),dim=-1) # b,n,c
    logits_t_nor = F.normalize(logits_t.permute(0,2,1),dim=-1) # b,n,c

    nei_s_simi = torch.matmul(neig_s_nor,logits_nor.unsqueeze(-1)).squeeze(-1)  # b,n,k
    nei_t_simi = torch.matmul(neig_t_nor, logits_t_nor.unsqueeze(-1)).squeeze(-1)  # b,n,k


    return nei_s_simi,nei_t_simi


def compute_neigh_gaussian_s3dis(logits,logits_t,nei_idx):
    """

    :param logits: b,c,n
    :param logits_t: b,c,n
    :param nei_idx: b,n,k
    :return:
    """
    device = logits.device
    logits_t = logits_t.detach()
    neig_s = index_points(logits.permute(0,2,1),nei_idx) # b,n,k,c
    neig_t = index_points(logits_t.permute(0,2,1),nei_idx) # b,n,k,c
    b,n,k,c = neig_t.shape
#method 1
    # neig_s = F.normalize(neig_s,p=2,dim=-1)
    # neig_t = F.normalize(neig_t,p=2,dim=-1)

    # neig_s_mean = torch.mean(neig_s,dim=-2,keepdim=True)  # b,n,1,c
    # neig_t_mean = torch.mean(neig_t,dim=-2,keepdim=True)  # b,n,1,c

    neig_s_mean = logits.permute(0,2,1).unsqueeze(-2)  # b,n,1,c
    neig_t_mean = logits_t.permute(0,2,1).unsqueeze(-2)  # b,n,1,c

  # #for logits
  #   neig_s_std = torch.std((neig_s-neig_s_mean).reshape(b,n,-1),dim=-1,keepdim=True) # b,n,1
  #   neig_t_std = torch.std((neig_t-neig_t_mean).reshape(b,n,-1),dim=-1,keepdim=True)
# for features
    neig_s_std = torch.std((neig_s-neig_s_mean),dim=-2,keepdim=True) # b,n,1,c
    neig_t_std = torch.std((neig_t - neig_t_mean),dim=-2,keepdim=True)

    nei_s_distri = (neig_s - neig_s_mean) / (neig_s_std + 1e-6) # b,n,k,c
    nei_t_distri = (neig_t - neig_t_mean) / (neig_t_std + 1e-6)  # b,n,k,c
  # for logits
  #   nei_s_distri = (neig_s - neig_s_mean) / (neig_s_std.unsqueeze(-1) + 1e-6)  # b,n,k,c
  #   nei_t_distri = (neig_t - neig_t_mean) / (neig_t_std.unsqueeze(-1) + 1e-6)  # b,n,k,c
   # for features
    s_std_range = torch.std(neig_s_std.reshape(b,n, -1), dim=-1, keepdim=True)  # b,n,1
    t_std_range = torch.std(neig_t_std.reshape(b,n, -1), dim=-1, keepdim=True)  # b,n,1

    s_mean_range = torch.std(neig_s_mean.reshape(b,n,-1), dim=-1, keepdim=True)  # b,n,1
    t_mean_range = torch.std(neig_t_mean.reshape(b,n,-1), dim=-1, keepdim=True)  # b,n,1


# for features
    neig_s_mean_new = neig_s_mean + torch.sigmoid(s_mean_range).unsqueeze(-1) * s_mean_range.unsqueeze(
        -1)  # b,n,1,c
    neig_t_mean_new = neig_t_mean + torch.sigmoid(t_mean_range).unsqueeze(-1) * t_mean_range.unsqueeze(
        -1)  # b,n,1,c
  # the same
    neig_s_std_new = neig_s_std + torch.sigmoid(s_std_range).unsqueeze(-1) * s_std_range.unsqueeze(-1) # b,n,1,c
    neig_t_std_new = neig_t_std + torch.sigmoid(t_std_range).unsqueeze(-1) * t_std_range.unsqueeze(-1)  # b,n,1,c
# for featuers
    nei_s_distri = neig_s_std_new * nei_s_distri + neig_s_mean_new  # b,n,k,c
    nei_t_distri = neig_t_std_new * nei_t_distri + neig_t_mean_new

    nei_s_distri = torch.nn.functional.normalize(nei_s_distri,p=2,dim=-2)
    nei_t_distri = torch.nn.functional.normalize(nei_t_distri, p=2, dim=-2)


    return nei_s_distri,nei_t_distri



def compute_loss_s3dis_list(logits_s,labels, cfg, logits_t, only_unlabeled=True,only_ce=True,evalmode = False):
    """

    :param logits_s: b,c,n
    :param labels: b,n
    :param cfg:
    :param logits_t: b,c,n
    :param only_unlabeled:
    :param only_ce:
    :param evalmode:
    :return:
    """
    logits_t = logits_t.detach()
    logits = logits_s
    b,c,n = logits.shape
    labels = labels
    logits_T = logits_t


    logits = logits.transpose(1,2)  # b,n,c
    logits_T = logits_T.transpose(1,2)  # b,n,c

    ignored_bool = labels == 13
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)  # b,n

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits_list = [logits[i][valid_idx_i,:] for i,valid_idx_i in enumerate(valid_idx) ]
    valid_logits_T_list = [logits_T[i][valid_idx_i,:] for i,valid_idx_i in enumerate(valid_idx) ]
    valid_labels_init = [labels[i][valid_idx_i] for i,valid_idx_i in enumerate(valid_idx)]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes+1).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels_list = [torch.gather(reducing_list, 0, indexi) for indexi in valid_labels_init ]
    if evalmode:
        return valid_logits_list,valid_labels_list,valid_logits_T_list


    loss = get_loss_list(valid_logits_list, valid_labels_list, cfg.class_weights)
    # print('student loss -----', loss.grad_fn)
#    loss = get_smoothloss(logits,labels,cfg.class_weights)

    unlabeled_idx = ignored_bool == 1
    unlabeled_logits_S_list = [logits[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]
    unlabeled_logits_T_list = [logits_T[i][valid_idx_i,:] for i,valid_idx_i in enumerate(unlabeled_idx) ]
    if only_unlabeled:
        # consistency_loss = consistency_loss_KL_list(unlabeled_logits_S_list,unlabeled_logits_T_list,b,n,c)
        consistency_loss = consistency_loss_JSD_list(unlabeled_logits_S_list,unlabeled_logits_T_list,b,n,c)
    else:
        # consistency_loss = consistency_loss_KL_list(logits,logits_T,b,n,c)
        consistency_loss = consistency_loss_JSD_list(logits, logits_T, b, n, c)


    if only_ce:
        total_loss = loss
    else:
        total_loss = loss + consistency_loss

    return total_loss, valid_logits_list,valid_labels_list,valid_logits_T_list

def compute_loss_s3dis_list_and_nei_simi_and_gaussian(nei_s_gaussian,nei_t_gaussian,nei_s_simi,nei_t_simi,logits_s,labels, cfg, logits_t, only_unlabeled=True,only_ce=True,evalmode = False):
    logits_t = logits_t.detach()
    logits = logits_s
    b, c, n = logits.shape
    labels = labels
    logits_T = logits_t

    logits = logits.transpose(1, 2)  # b,n,c
    logits_T = logits_T.transpose(1, 2)  # b,n,c

    ignored_bool = labels == 13
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)  # b,n

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits_list = [logits[i][valid_idx_i, :] for i, valid_idx_i in enumerate(valid_idx)]
    valid_logits_T_list = [logits_T[i][valid_idx_i, :] for i, valid_idx_i in enumerate(valid_idx)]
    valid_labels_init = [labels[i][valid_idx_i] for i, valid_idx_i in enumerate(valid_idx)]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes + 1).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels_list = [torch.gather(reducing_list, 0, indexi) for indexi in valid_labels_init]
    if evalmode:
        return valid_logits_list, valid_labels_list, valid_logits_T_list

    loss = get_loss_list(valid_logits_list, valid_labels_list, cfg.class_weights)
    # print('student loss -----', loss.grad_fn)
    #    loss = get_smoothloss(logits,labels,cfg.class_weights)
    if only_unlabeled:
        unlabeled_idx = ignored_bool == 1
        unlabeled_logits_S_list = [logits[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]
        unlabeled_logits_T_list = [logits_T[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]

        # unlabeled_logits_s_simi_list = [nei_s_simi[i][valid_idx_i, :] for i, valid_idx_i in
        #                                 enumerate(unlabeled_idx)]  # [[n,c,k]....]
        # unlabeled_logits_t_simi_list = [nei_t_simi[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]

        consistency_loss_simi = consistency_loss_JSD_list_neisimi(nei_s_simi, nei_t_simi,
                                                            unlabeled_logits_S_list, unlabeled_logits_T_list, b, n, c)
        cosistency_loss_gaussian = consistency_loss_JSD_list_gaussian(nei_s_gaussian,nei_t_gaussian,
                                                             unlabeled_logits_S_list, unlabeled_logits_T_list, b, n, c)
        consistency_loss = 1.0 * consistency_loss_simi + 1.0 * cosistency_loss_gaussian
    else:
        consistency_loss_simi = consistency_loss_JSD_list_neisimi(nei_s_simi, nei_t_simi, logits, logits_T, b, n, c)
        cosistency_loss_gaussian = consistency_loss_JSD_list_gaussian(nei_s_gaussian, nei_t_gaussian,
                                                                     logits, logits_T,b, n, c)
        consistency_loss = 1.0 * consistency_loss_simi + 1.0 * cosistency_loss_gaussian

    if only_ce:
        total_loss = loss
    else:
        total_loss = loss + consistency_loss

    return total_loss, valid_logits_list, valid_labels_list, valid_logits_T_list


def compute_loss_s3dis_list_and_nei_simi(nei_s_simi,nei_t_simi,logits_s,labels, cfg, logits_t, only_unlabeled=True,only_ce=True,evalmode = False):
    """
nei_s_simi:b,n,c,k
    :param logits_s: b,c,n
    :param labels: b,n
    :param cfg:
    :param logits_t: b,c,n
    :param only_unlabeled:
    :param only_ce:
    :param evalmode:
    :return:
    """
    logits_t = logits_t.detach()
    logits = logits_s
    b,c,n = logits.shape
    labels = labels
    logits_T = logits_t


    logits = logits.transpose(1,2)  # b,n,c
    logits_T = logits_T.transpose(1,2)  # b,n,c

    ignored_bool = labels == 13
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)  # b,n

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits_list = [logits[i][valid_idx_i,:] for i,valid_idx_i in enumerate(valid_idx) ]
    valid_logits_T_list = [logits_T[i][valid_idx_i,:] for i,valid_idx_i in enumerate(valid_idx) ]
    valid_labels_init = [labels[i][valid_idx_i] for i,valid_idx_i in enumerate(valid_idx)]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes+1).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels_list = [torch.gather(reducing_list, 0, indexi) for indexi in valid_labels_init ]
    if evalmode:
        return valid_logits_list,valid_labels_list,valid_logits_T_list

    loss = get_loss_list(valid_logits_list, valid_labels_list, cfg.class_weights)
    # print('student loss -----', loss.grad_fn)
#    loss = get_smoothloss(logits,labels,cfg.class_weights)
    if only_unlabeled:
        unlabeled_idx = ignored_bool == 1
        unlabeled_logits_S_list = [logits[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]
        unlabeled_logits_T_list = [logits_T[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]

        unlabeled_logits_s_simi_list = [nei_s_simi[i][valid_idx_i, :] for i, valid_idx_i in
                                        enumerate(unlabeled_idx)]  # [[n,c,k]....]
        unlabeled_logits_t_simi_list = [nei_t_simi[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]

        consistency_loss = consistency_loss_JSD_list_gaussian(unlabeled_logits_s_simi_list,unlabeled_logits_t_simi_list,
                                                    unlabeled_logits_S_list,unlabeled_logits_T_list,b,n,c)
    else:
        consistency_loss = consistency_loss_JSD_list_gaussian(nei_s_simi,nei_t_simi,logits,logits_T,b,n,c)


    if only_ce:
        total_loss = loss
    else:
        total_loss = loss + 0.5 * consistency_loss

    return total_loss, valid_logits_list,valid_labels_list,valid_logits_T_list

def compute_loss_s3dis_list_and_nei_simi_and_logits(s_logits,t_logits,nei_s_simi,nei_t_simi,logits_s,labels, cfg, logits_t, only_unlabeled=True,only_ce=True,evalmode = False):
    """
nei_s_simi:b,n,c,k
    :param logits_s: b,c,n
    :param labels: b,n
    :param cfg:
    :param logits_t: b,c,n
    :param only_unlabeled:
    :param only_ce:
    :param evalmode:
    :return:
    """
    logits_t = logits_t.detach()
    logits = logits_s
    b,c,n = logits.shape
    labels = labels
    logits_T = logits_t


    logits = logits.transpose(1,2)  # b,n,c
    logits_T = logits_T.transpose(1,2)  # b,n,c

    ignored_bool = labels == 13
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)  # b,n

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits_list = [logits[i][valid_idx_i,:] for i,valid_idx_i in enumerate(valid_idx) ]
    valid_logits_T_list = [logits_T[i][valid_idx_i,:] for i,valid_idx_i in enumerate(valid_idx) ]
    valid_labels_init = [labels[i][valid_idx_i] for i,valid_idx_i in enumerate(valid_idx)]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes+1).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels_list = [torch.gather(reducing_list, 0, indexi) for indexi in valid_labels_init ]
    if evalmode:
        return valid_logits_list,valid_labels_list,valid_logits_T_list

    loss = get_loss_list(valid_logits_list, valid_labels_list, cfg.class_weights)
    # print('student loss -----', loss.grad_fn)
#    loss = get_smoothloss(logits,labels,cfg.class_weights)
    if only_unlabeled:
        unlabeled_idx = ignored_bool == 1
        unlabeled_logits_S_list = [logits[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]
        unlabeled_logits_T_list = [logits_T[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]

        unlabeled_logits_s_simi_list = [nei_s_simi[i][valid_idx_i, :] for i, valid_idx_i in
                                        enumerate(unlabeled_idx)]  # [[n,c,k]....]
        unlabeled_logits_t_simi_list = [nei_t_simi[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]

        unlabeled_logits_s_simi_list_1 = [s_logits[i][valid_idx_i, :] for i, valid_idx_i in
                                        enumerate(unlabeled_idx)]  # [[n,c,k]....]
        unlabeled_logits_t_simi_list_1 = [t_logits[i][valid_idx_i, :] for i, valid_idx_i in enumerate(unlabeled_idx)]


        consistency_loss = consistency_loss_JSD_list_gaussian(unlabeled_logits_s_simi_list,unlabeled_logits_t_simi_list,
       unlabeled_logits_S_list,unlabeled_logits_T_list,b,n,c) + consistency_loss_JSD_list_gaussian(unlabeled_logits_s_simi_list_1,unlabeled_logits_t_simi_list_1,
                         unlabeled_logits_S_list,unlabeled_logits_T_list,b,n,c)


    else:
        consistency_loss = consistency_loss_JSD_list_gaussian(nei_s_simi,nei_t_simi,logits,logits_T,b,n,c)


    if only_ce:
        total_loss = loss
    else:
        total_loss = loss + 0.25 * consistency_loss

    return total_loss, valid_logits_list,valid_labels_list,valid_logits_T_list
from lovasz import lovasz_softmax as laso
def get_loss(logits, labels, pre_cal_weights):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss

def get_loss_list(logits_list, labels_list, pre_cal_weights):
    class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    output_loss = 0
    for i in range(len(logits_list)):
        output_loss = criterion(logits_list[i],labels_list[i]) + output_loss
    # output_loss = output_loss.mean()
    return output_loss / len(logits_list)
def get_smoothloss(logits,labels,precal_weights):
    """

    :param logits: b*n,c
    :param labels: b*n
    :param precal_weights:
    :return:
    """
    n_class = logits.size(1)
    eps = 0.2
    one_hot = torch.zeros_like(logits).scatter(1,labels.view(-1,1),1)

    one_hot = one_hot * (1-eps) + (1- one_hot) * eps / (n_class - 1)
    log_pro = F.log_softmax(logits,1)
    loss = -(one_hot * log_pro).sum(dim=1).mean()
    return loss



class featurememory:
    def __init__(self,memory_per_class = 1024, n_classes=13,feature_size=None):
        self.memory_per_class = memory_per_class
        self.n_classes = n_classes
        self.feature_size = feature_size
        self.memory = [None] * n_classes

    def add_feature_memory_bank(self,features,labels):
        """

        :param features: b,n,c
        :param labels: b,n
        :param batch_size:b
        :return:
        """
        b,n,c = features.shape
        features = features.reshape(-1,c)  # b*n,c
        labels = labels.reshape(-1)        # b*n
        features = features.detach()
        labels = labels.detach().cpu().numpy()
        ignore_bool = labels == 13
        valid_idx = ignore_bool == 0
        valid_labels = labels[valid_idx]
        for c in range(self.n_classes):
            mask_c = labels == c  #当前batch里有多少个这个类的point
            features_c = features[mask_c]
            if features_c.shape[0] > 0:
                new_features = features_c.cpu().numpy()
                if self.memory[c] is None:  # empty first element
                    self.memory[c] = new_features
                else:
                    self.memory[c] = np.concatenate((new_features,self.memory[c]),axis=0)[:self.memory_per_class,:]

def contrastive_loss_between_memorybank(logits,features,labels,nei_idx,memorybank,threhold,num_class,temperature,banksize,anchor_ratio):
    """

    :param logits: model output logits:b,n,c_class
    :param features:model decoder features: b,n,c
    :param labels :b,n
    :param nei_idx:b,n,k
    :param memorybank: list[(2048,32),...]
    :param threhold: 0.9 hard threhold to select high confidence point
    :param num_class: 13 for s3dis
    :return:
    """
    loss = 0
    # features_k = index_points(features,nei_idx) # b,n,k,c
    # features_k = features_k.reshape(-1,features_k.shape(-2),features_k.shape(-1))
    logits = logits.reshape(-1,logits.shape[-1])  #b*n,13
    features = features.reshape(-1,features.shape[-1])  # b*n,c
    labels = labels.reshape(-1)  #b*n
    #for s3dis unlabeled idx is 13
    unlabeled_idx = labels == 13       #U
    labeled_idx = unlabeled_idx == 0  #L

    unlabeled_logits = logits[unlabeled_idx]  # U,13
    unlabeled_features = features[unlabeled_idx]  #U,c
    # criterian = JSD()

    for c in range(num_class):
        #首先生成pseudo label,显然只对unlabeld的point
        #pseudo_label = torch.argmax(unlabeled_logits,dim=-1) #U [0,4,3,12.....]
        confidence = torch.softmax(unlabeled_logits,dim=-1)
        confidence,pseudo_label = torch.max(confidence,dim=-1)  # 0: confidence  1:index
        confidence,ind = torch.sort(confidence,dim=0,descending=True) # U_c  降序
        mask_c_confi = confidence > threhold
        mask_c_pseudolabel = pseudo_label[ind] == c
        mask_c = torch.logical_and(mask_c_confi , mask_c_pseudolabel)
        if mask_c.sum() > 1:
            #根据mask找到那些unlabeld的features,     现在按照比例选一部分
            features_c = unlabeled_features[mask_c]  #U_c, c
            num_c = torch.ceil(torch.tensor(features_c.shape[0] * anchor_ratio)).long().cuda()
            features_c = features_c[:num_c,:]

            features_c_nor = F.normalize(features_c,dim=-1) #U_c,c
            u_c,_ = features_c_nor.shape

            memory = np.array(memorybank.memory)
            memory = torch.from_numpy(memory).cuda()  # cls,2048,c
            memory_nor = F.normalize(memory,dim=-1)
            memory_c_nor = memory_nor[c] # 2048,c

            # add center calculation
            memory_c_nor = torch.mean(memory_c_nor,dim=0,keepdim=True) #1,c

            memory_not_c_nor = torch.cat((memory_nor[0:c,:,:],memory_nor[c+1:,:,:]),dim=0) # cls-1,2048,c
            # add center calculation
            memory_not_c_nor = torch.mean(memory_not_c_nor,dim=1,keepdim=True) #cls-1, 1, c

            memory_not_c_nor = torch.cat(torch.unbind(memory_not_c_nor,dim=1),dim=0) # cls-1,2048,c--->n,c

            anchor_dot_constrast = torch.matmul(features_c_nor,memory_c_nor.T) # u_c, 2048  positive logits
            anchor_dot_constrast = torch.div(anchor_dot_constrast,temperature)

           # for numerical stability
            logits_max,_ = torch.max(anchor_dot_constrast,dim=1,keepdim=True) #u_c,1
            logits = anchor_dot_constrast - logits_max.detach()  # u_c, 2048

            anchor_dot_cons_negative = torch.matmul(features_c_nor,memory_not_c_nor.T) # u_c , cls-1*2048
            anchor_dot_cons_negative = torch.div(anchor_dot_cons_negative,temperature)
            logits_negative_max,_ = torch.max(anchor_dot_cons_negative,dim=1,keepdim=True) #u_c,1
            logits_negative = anchor_dot_cons_negative - logits_negative_max.detach() # u_c, cls-1*2048

            exp_positve = torch.exp(logits)
            exp_negative = torch.exp(logits_negative)

            constrativeloss = (-1) * logits.sum(1) + torch.log(exp_positve.sum(1) + exp_negative.sum(1) + 1e-6)

            loss = loss + constrativeloss.mean()








            # memory_c = torch.from_numpy(memorybank.memory[c]).cuda() # 2048,c
            # memory_c_nor = F.normalize(memory_c,dim=-1) # 2048,c
            # memory_c_mean = torch.mean(memory_c,dim=0,keepdim=True) #1,c
            #
            # #l1
            # memory_c_std = torch.std(memory_c,dim=0,keepdim=True)

            #JSD
            # memory_c_std = torch.std(memory_c - memory_c_mean,dim=0,keepdim=True)
            # map into gaussian distribution with a little pertubation
            # memory_c_gau = (memory_c - memory_c_mean) / (memory_c_std + 1e-6)  # 2048,c
            # memory_mean_range = torch.std(memory_c_mean.reshape(-1), dim=-1)  # 1
            # memory_std_range = torch.std(memory_c_std.reshape(-1), dim=-1)  # 1
            # memory_mean_c_new = memory_c_mean + torch.sigmoid(memory_mean_range).unsqueeze(-1) * memory_c_mean  # 1,c
            # memory_std_c_new = memory_c_std + torch.sigmoid(memory_std_range).unsqueeze(-1) * memory_c_std  # 1,c
            # memory_c_gau = memory_std_c_new * memory_c_gau + memory_mean_c_new # 2048,c
            #
            # size_memory = memory_c_gau.shape[0]
            # size_feas = features_c_gau.shape[0]

            # if size_feas >= size_memory:
            #     select_idx = np.random.choice(size_feas,size_memory,replace=False)
            # else:
            #     select_idx1 = np.random.choice(size_feas,size_memory-size_feas,replace=True)
            #     select_idx2 = np.arange(0,size_feas)
            #     select_idx = np.concatenate((select_idx1,select_idx2),axis=0)
            #
            # select_idx = torch.from_numpy(select_idx).cuda()
            # select_features_c_gua = features_c_gau[select_idx]



            # l1 loss
            # loss = loss + F.l1_loss(mean_c,memory_c_mean) + F.l1_loss(std_c,memory_c_std) + torch.mean(std_c)
            #Mahalanobis Distance
            # loss = loss + mahalanobis_distance(features_c,memory_c_mean,memory_c_std)
            #mse loss
            # loss = loss + F.mse_loss(mean_c,memory_c_mean) + F.mse_loss(std_c,memory_c_std)
            #JSD
            # loss = loss + criterian(select_features_c_gua,memory_c_gau)
            #MLS
            # loss = loss + MLS(mean_c.squeeze(0),memory_c_mean.squeeze(0),std_c.squeeze(0),memory_c_std.squeeze(0))

            #Bhattacharyya Distance


    return loss / num_class

def mahalanobis_distance(a,mean,std):
    """

    :param mean_a: 1,c
    :param mean_b: 1,c
    :param std_a: 1,c
    :param std_b: 1,c
    :return:
    """
    loss = torch.sqrt(torch.mean( (a-mean) * (1 / (std + 1e-6)) * (a-mean)))
    return loss

def bhattach_distance(mean_a,mean_b,std_a,std_b):
    """

    :param mean_a:1,c
    :param mean_b: 1,c
    :param std_a: 1,c
    :param std_b: 1,c
    :return:
    """


def MLS(mean_a,mean_b,std_a,std_b):
    """
    :param mean_a : 1,c
    :param mean_b: 1,c
    :param std_a: 1,c
    :param std_b: 1,c
    :return: 1
    """
    dimension = mean_a.shape()
    loss = 0
    for i in range(dimension):
        loss = loss + (mean_a[i] - mean_b[i]) ** 2 / (std_a[i] + std_b[i]) + torch.log(std_a[i] + std_b[i])
    loss = -0.5 * loss -0.5 * dimension * torch.log(2 * torch.pi)
    return loss


def adjust_memory_size(memorybank,size):
    """

    :param memorybank:[[...],[...],[...]]
    :return:
    """
    each_size = [len(i) for i in memorybank]#[6,3,4,6....]
    offset = [size - i for i in each_size]  #[0,3,2,0...]
    select_idx = [np.random.choice(each_size[i],offset[i],replace=True) for i in range(len(each_size))]
    adjust = [np.concatenate((np.array(memorybank[i]),np.array(memorybank[i])[select_idx[i]]),axis=0) for i in range(len(each_size))]

    return np.array(adjust)






















if __name__ == '__main__':
    from pointnet2_utils import index_points
    import torch
    data = torch.rand(2,5,2) # b,n,c
    #neiidx  b,n,k
    idx = torch.randint(0,5,(2,5,2))
    out = index_points(data,idx)
    print(data)
    print(idx)
    print(out)

