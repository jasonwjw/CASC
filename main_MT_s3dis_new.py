from helper_tool import ConfigS3DIS as cfg
from RandLANet_MT_S3DIS import *
# from scfnet import Network, compute_loss_s3dis, compute_acc, IoUCalculator
#from s3dis_dataset import S3DIS
# from weakly_possi_s3disdataset import S3DIS
from MT_s3dis_dataset import S3DIS
import numpy as np
import os, argparse
import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import shutil
from torch.nn.utils import clip_grad_norm_


parser = argparse.ArgumentParser()
now = str(datetime.now().strftime('%Y-%m-%d-%H-%M'))
parser.add_argument('--checkpoint_path', default='log/s3dis/'+now+'-motivation-kl/checkpoint.tar')
parser.add_argument('--log_dir', default='log/s3dis/'+now+'-motivation-kl/')
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--onlyce',  default= False)
parser.add_argument('--onlyunlabled',  default= True)
parser.add_argument('--threhold', type=float, default=0.95)
parser.add_argument('--memory_epoch', type=int, default= 0)
parser.add_argument('--memory_loss',type=int,  default= 10)
parser.add_argument('--banksize', type=int, default=64)
parser.add_argument('--t', type=float, default=0.1)
parser.add_argument('--loss_c', type=float, default=0.01)
FLAGS = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
#################################################   log   #################################################
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


#################################################   dataset   #################################################
# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
TRAIN_DATASET = S3DIS(mode='training',test_id=5)
TEST_DATASET = S3DIS(mode='test',test_id=5)

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=TRAIN_DATASET.collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=TEST_DATASET.collate_fn)


print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
TOTAL_INTERATION = len(TRAIN_DATALOADER) * FLAGS.max_epoch
trainfile = os.path.abspath(__file__)
shutil.copy(trainfile, str(LOG_DIR))
shutil.copy('./RandLANet_MT_S3DIS.py',str(LOG_DIR))   #RandLANet.py
shutil.copy('./MT_s3dis_dataset.py',str(LOG_DIR))  #s3dis_dataset.py

#################################################   network   #################################################
def create_ema_model(model, net_class,cfg):
    ema_model = net_class(cfg,is_aug=False)
    for param in ema_model.parameters():
        param.detach_()       #不参与参数更新
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model
def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration*10 + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_S = Network(cfg,is_aug=True)  # student model : strong aug
net_S.to(device)

memorybank = featurememory(memory_per_class=FLAGS.banksize,n_classes=13,feature_size=None)

# Load the Adam optimizer
optimizer = optim.Adam(net_S.parameters(), lr=cfg.learning_rate)
#clip_grad_norm_(net_S.paramters(),10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=FLAGS.max_epoch,eta_min = cfg.learning_rate / 100)
# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
CHECKPOINT_PATH = FLAGS.checkpoint_path


# if torch.cuda.device_count() > 1:
#     log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     net = nn.DataParallel(net)




#################################################   training functions   ###########################################



def train(start_epoch):
    global EPOCH_CNT
    global evalmiou
    BESTMIOU = 0
    # ALPHA = FLAGS.alpha
    # loss = 0
    iter = 0
    net_T = create_ema_model(net_S, net_class=Network, cfg=cfg)
    net_T.to(device)
    for epoch in range(start_epoch, FLAGS.max_epoch):
        EPOCH_CNT = epoch
        evalmiou = 0.001
        log_string('**** EPOCH %03d ****' % (epoch))

        log_string(str(datetime.now()))

        np.random.seed()

        stat_dict_S = {}  # collect statistics
        stat_dict_T = {}

        net_S.train()  # set model to training mode
        net_T.train()
        iou_calc_S = IoUCalculator_list(cfg)
        iou_calc_T = IoUCalculator_list(cfg)

        for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)

            # Forward pass
            optimizer.zero_grad()
            with torch.no_grad():
                features_t_p,features_t_m,logits_t,_ ,_= net_T(batch_data)

            features_s_p,features_s_m,logits_s,labels,nei_idx = net_S(batch_data)

      #method 1   只拉近feature的距离
            # loss_S, valid_logits_s,valid_labels,valid_logits_T= compute_loss_s3dis_list(logits_s, labels, cfg, logits_t,
            #                         only_unlabeled=FLAGS.onlyunlabled,only_ce=FLAGS.onlyce, evalmode=False)
#method2 拉近both simi和gaussian的距离
            # neig_s_gaussian,neig_t_gaussian = compute_neigh_gaussian_s3dis(logits_s,logits_t,nei_idx)
            # neig_s_simi,neig_t_simi = compute_neigh_simi_s3dis(logits_s,logits_t,nei_idx)
            # loss_S, valid_logits_s,valid_labels,valid_logits_T = compute_loss_s3dis_list_and_nei_simi_and_gaussian(neig_s_gaussian,
            #                         neig_t_gaussian,neig_s_simi,neig_t_simi,
            #                         logits_s,labels, cfg,logits_t,
            #                      only_unlabeled=FLAGS.onlyunlabled,only_ce=FLAGS.onlyce,
            #                     evalmode=False)
            #method 3 只拉近simi的距离
            # neig_s_simi, neig_t_simi = compute_neigh_simi_s3dis(logits_s, logits_t, nei_idx)
            # loss_S, valid_logits_s, valid_labels, valid_logits_T = compute_loss_s3dis_list_and_nei_simi(
            #      neig_s_simi, neig_t_simi,
            #     logits_s, labels, cfg, logits_t,
            #     only_unlabeled=FLAGS.onlyunlabled, only_ce=FLAGS.onlyce,
            #     evalmode=False)
            #method4 只拉近gaussian的距离
            # neig_s_simi, neig_t_simi = compute_neigh_gaussian_s3dis(logits_s, logits_t, nei_idx)

            #construct memory bank when epoch bigger than 2
            if epoch  > FLAGS.memory_epoch:
                memorybank.add_feature_memory_bank(features_t_p.permute(0,2,1),labels)
                # memorybank.add_feature_memory_bank(logits_t.permute(0, 2, 1), labels)


            neig_s_simi, neig_t_simi = compute_neigh_gaussian_s3dis(features_s_m, features_t_m, nei_idx)
            # neig_s_simi_logits, neig_t_simi_logits = compute_neigh_gaussian_s3dis(logits_s, logits_t, nei_idx)
            # neig_s_simi, neig_t_simi = logits_s, logits_t
            loss_S, valid_logits_s, valid_labels, valid_logits_T = compute_loss_s3dis_list_and_nei_simi(
                neig_s_simi, neig_t_simi,
                logits_s, labels, cfg, logits_t,
                only_unlabeled=FLAGS.onlyunlabled, only_ce=FLAGS.onlyce,
                evalmode=False)
            if epoch > FLAGS.memory_loss:
                # #logits
                # loss_con = contrastive_loss_between_memorybank(logits_s.permute(0, 2, 1), features_s.permute(0, 2, 1),
                #                                                labels, nei_idx, memorybank, threhold=FLAGS.threhold,
                #                                                num_class=13)
                #features
                loss_con = contrastive_loss_between_memorybank(logits_s.permute(0,2,1),features_s_p.permute(0,2,1),labels,nei_idx,memorybank,
                                                               threhold=FLAGS.threhold,num_class=13,temperature=FLAGS.t,banksize=FLAGS.banksize,anchor_ratio= epoch / FLAGS.max_epoch)
                loss_S = FLAGS.loss_c * loss_con + loss_S

            loss_S.backward()

            # clip_grad_norm_(net_S.parameters(),10)
            optimizer.step()
            #updata teacher's parameters
#            ALPHA = min(1-1/(EPOCH_CNT+1),ALPHA)
#            ALPHA = 1 - (1 - 0.995) * (math.cos(math.pi * iter / TOTAL_INTERATION) + 1) / 2
            ALPHA = 0.995
            net_T = update_ema_variables(ema_model=net_T, model= net_S, alpha_teacher=ALPHA, iteration=iter)
            iter = iter + 1
            # for tp,sp in zip(net_T.parameters(),net_S.parameters()):
            #     tp.data.mul_(ALPHA).add_(1-ALPHA,sp.data)


            acc_S = compute_acc_list(valid_logits_s,valid_labels)
            iou_calc_S.add_data(valid_logits_s,valid_labels)

            acc_T = compute_acc_list(valid_logits_T,valid_labels)
            iou_calc_T.add_data(valid_logits_T,valid_labels)


            if 'loss' not in stat_dict_S: stat_dict_S['loss'] = 0
            if 'acc' not in stat_dict_S:stat_dict_S['acc'] = 0
            stat_dict_S['loss'] += loss_S.item()
            stat_dict_S['acc'] += acc_S.item()


            # if 'loss' not in stat_dict_T: stat_dict_T['loss'] = 0
            if 'acc' not in stat_dict_T:stat_dict_T['acc'] = 0
            # stat_dict_T['loss'] += loss_S.item()
            stat_dict_T['acc'] += acc_T.item()

            batch_interval = 50
            if (batch_idx) % batch_interval == 0:
                log_string(' ---- batch: %03d ----' % (batch_idx + 1))

                for key in sorted(stat_dict_S.keys()):
                    log_string('student --- mean %s: %f' % (key, stat_dict_S[key] / batch_interval))
                    stat_dict_S[key] = 0

                for key in sorted(stat_dict_T.keys()):
                    log_string('teacher --- mean %s: %f' % (key, stat_dict_T[key] / batch_interval))
                    stat_dict_T[key] = 0

        mean_iou_S, iou_list_S = iou_calc_S.compute_iou()
        log_string('student --- training mean IoU:{:.1f}'.format(mean_iou_S * 100))
        s = 'IoU:'
        for iou_tmp in iou_list_S:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_string(s)

        mean_iou_T, iou_list_T = iou_calc_T.compute_iou()
        log_string('teacher -- training mean IoU:{:.1f}'.format(mean_iou_T * 100))
        s = 'IoU:'
        for iou_tmp in iou_list_T:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_string(s)

        scheduler.step()
        # train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 2 == 0:  # Eval every 10 epochs
            log_string(
                '******************************************* EVAL EPOCH %03d START***********************************' % (
                    epoch))
            # evalmiou = 0
            stat_dict_eval_S = {}  # collect statistics
            stat_dict_eval_T = {}
            net_S.eval()  # set model to eval mode (for bn and dp)
            net_T.eval()

            iou_cal_eval_S = IoUCalculator_list(cfg)
            iou_cal_eval_T = IoUCalculator_list(cfg)
            for batch_idx, batch_data in enumerate(TEST_DATALOADER):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].cuda()
                    else:
                        batch_data[key] = batch_data[key].cuda()

                # Forward pass
                with torch.no_grad():
                    features_t_c,features_t_m,logits_t, _,_ = net_T(batch_data)
                    features_s_c,features_s_m,logits_s,labels,nei_idx = net_S(batch_data)

                valid_logits_s, valid_labels, valid_logits_T = compute_loss_s3dis_list(logits_s, labels, cfg,
                                                 logits_t, only_unlabeled=FLAGS.onlyunlabled,only_ce=FLAGS.onlyce,
                                                 evalmode=True)

                acc_S = compute_acc_list(valid_logits_s, valid_labels)
                iou_cal_eval_S.add_data(valid_logits_s, valid_labels)

                acc_T = compute_acc_list(valid_logits_T, valid_labels)
                iou_cal_eval_T.add_data(valid_logits_T, valid_labels)


                # Accumulate statistics and print out
                # if 'loss' not in stat_dict_eval_S: stat_dict_eval_S['loss'] = 0
                if 'acc' not in stat_dict_eval_S: stat_dict_eval_S['acc'] = 0
                # stat_dict_eval_S['loss'] += loss_S.item()
                stat_dict_eval_S['acc'] += acc_S.item()

                # if 'loss' not in stat_dict_T: stat_dict_T['loss'] = 0
                if 'acc' not in stat_dict_eval_T: stat_dict_eval_T['acc'] = 0
                # stat_dict_T['loss'] += loss_S.item()
                stat_dict_eval_T['acc'] += acc_T.item()


                batch_interval = 50
                if (batch_idx + 1) % batch_interval == 0:
                    log_string(' ---- batch: %03d ----' % (batch_idx + 1))

            for key in sorted(stat_dict_eval_S.keys()):
                log_string('student----- eval mean %s: %f' % (key, stat_dict_eval_S[key] / (float(batch_idx + 1))))
            mean_iou_S, iou_list_S = iou_cal_eval_S.compute_iou()
            log_string('stduent --- evaling mean IoU:{:.1f}'.format(mean_iou_S * 100))
            s = 'IoU:'
            for iou_tmp in iou_list_S:
                s += '{:5.2f} '.format(100 * iou_tmp)
            log_string(s)

            for key in sorted(stat_dict_eval_T.keys()):
                log_string('teacher eval mean %s: %f' % (key, stat_dict_eval_T[key] / (float(batch_idx + 1))))
            mean_iou_T, iou_list_T = iou_cal_eval_T.compute_iou()
            log_string('teacher --- evaling mean IoU:{:.1f}'.format(mean_iou_T * 100))
            s = 'IoU:'
            for iou_tmp in iou_list_T:
                s += '{:5.2f} '.format(100 * iou_tmp)
            log_string(s)


            evalmiou = mean_iou_T
            print('miou is ------', mean_iou_T)
            print('evalmiou is-------', evalmiou)
            # log_string('---------------------------eval miou is  %3d ------------------' % (evalmiou))
            log_string(
                '********************************************* EVAL EPOCH %03d END************************************' % (
                    epoch))
            # Save checkpoint
            if BESTMIOU <= evalmiou:
                save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss': loss_S,
                             }
                try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                    save_dict['model_state_dict'] = net_T.module.state_dict()
                except:
                    save_dict['model_state_dict'] = net_T.state_dict()
                torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
                BESTMIOU = mean_iou_T
                log_string('saving checkpoint!!!!!!! and miou is %3f  !!!!!!!' % (BESTMIOU))
            log_string('---------------------------BEST MIOU is  %3f ------------------' % (BESTMIOU))

    log_string('!!!!!!!!!!!!! training finished !!!!!!!!!!!!!!!')
if __name__ == '__main__':

    train(start_epoch)

