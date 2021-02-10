# coding: utf-8
# Heavily modified from https://github.com/minzwon/sota-music-tagging-models/
import math
import os
import numpy as np
from sklearn import metrics
import datetime
import time
import pickle as pkl
import librosa
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from frontend import Frontend_mine, Frontend_won
from backend import Backend
from data_loader import get_DataLoader

# define here all the parameters
main_dict = {"frontend_dict":
             {"list_out_channels":[128,128,128,256,256,256], 
              "list_kernel_sizes":[(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)],
              "list_pool_sizes":  [(3,2),(2,2),(2,2),(2,2),(2,2),(2,1)], 
              "list_avgpool_flags":[False,False,False,False,False,True]},
             
             "backend_dict":
             {"n_class":50,
              "bert_config":None, 
              "recurrent_units":2}, #  pass None to deactivate
             
             "dataset":'msd',
             "architecture":'crnnsa',
             "n_epochs":1000,
             
             "data_loader_dict":
             {"path_to_repo":'~/dl4am/',
              "batch_size":32,
              "input_length":15, # [s]
              "spec_path":'/import/c4dm-datasets/rmri_self_att/msd',
              "audio_path":'/import/c4dm-03/Databases/songs/',
              "mode":'train', 
              "num_workers":20}}

def get_auc(predictions, groundtruths):
    predictions = np.array(predictions)
    groundtruths = np.array(groundtruths)

    roc_aucs  = metrics.roc_auc_score(groundtruths, predictions, average='macro')
    pr_aucs = metrics.average_precision_score(groundtruths, predictions, average='macro')
    print('roc_auc: %.4f' % roc_aucs)
    print('pr_auc: %.4f' % pr_aucs)
    return roc_aucs, pr_aucs

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class CRNNSA(nn.Module):
    
    """
    TODO: explore whether "rec_unit -> self_att -> rec_unit"
    would have worked better.
    """
    
    # BERT-based Convolutional Recurrent Neural Network
    # Code adopted from https://github.com/minzwon/sota-music-tagging-models/
    def __init__(self, main_dict=None, backend=None, frontend=None):
        super(CRNNSA, self).__init__()
        

        if main_dict is not None:
            self.frontend = Frontend_mine(main_dict["frontend_dict"])
            self.backend = Backend(main_dict)
        else:
            self.frontend = frontend
            self.backend = backend


    def forward(self, spec):
        
        x = self.backend(self.frontend(spec))
        
        return x

class Solver(object):
    def __init__(self, main_dict):
        
        # Data loader
        self.path_to_repo = os.path.expanduser(main_dict["data_loader_dict"]["path_to_repo"])

        self.data_loader_train = get_DataLoader(main_dict["data_loader_dict"]["batch_size"],
                                                main_dict["data_loader_dict"]["input_length"],
                                                main_dict["data_loader_dict"]["spec_path"],
                                                main_dict["data_loader_dict"]["audio_path"],
                                                main_dict["data_loader_dict"]["path_to_repo"],
                                                main_dict["data_loader_dict"]["mode"], 
                                                main_dict["data_loader_dict"]["num_workers"])
        
        self.data_loader_val   = get_DataLoader(1,
                                                main_dict["data_loader_dict"]["input_length"],
                                                main_dict["data_loader_dict"]["spec_path"],
                                                main_dict["data_loader_dict"]["audio_path"],
                                                main_dict["data_loader_dict"]["path_to_repo"],
                                                "valid", 
                                                main_dict["data_loader_dict"]["num_workers"])
        
        # Preprocessing
        self.fs = 16000
        self.window = 512
        self.hop = 256
        self.mel = 96
        
        # Batch size
        self.batch_size = main_dict["data_loader_dict"]["batch_size"]
        # Epochs
        self.n_epochs = main_dict["n_epochs"]
        # Learning rate
        self.initial_lr = 1e-3
        # Model
        self.model = CRNNSA(main_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.initial_lr)       
        # Loss
        self.criterion = nn.BCELoss()
        # Save path
        self.model_save_path = os.path.join(self.path_to_repo,"models")
        os.makedirs(self.model_save_path, exist_ok=True)
    
        # Tensorboard
        now = datetime.datetime.now()
        log_dir = os.path.join("./","logs",
                               main_dict["dataset"],
                               main_dict["architecture"],
                               now.strftime("%m:%d:%H:%M"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"tensorboard --logdir '{log_dir}' --port ")
        self.writer = SummaryWriter(log_dir)

    def train(self):

        # Start training
        start_t = time.time()
        current_optimizer = 'adam'
        best_roc_auc = 0
        drop_counter = 0
        
        for epoch in range(self.n_epochs):
            # train
            ctr = 0
            drop_counter += 1
            n_iters_per_epoch = math.ceil(float(len(self.data_loader_train))/self.batch_size)
            self.model.train()
            for x, y in self.data_loader_train:
                ctr += 1

                # Forward
                x = to_var(x)
                y = to_var(y)
                out = self.model(x)

                # Backward
                loss = self.criterion(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                if (ctr % 100) == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             epoch+1, self.n_epochs, ctr, len(self.data_loader_train), loss.item(),
                             datetime.timedelta(seconds=time.time()-start_t)))
                    self.writer.add_scalar("train/step_loss", loss.item(), epoch*n_iters_per_epoch + ctr)

            # validation
            roc_auc, pr_auc = self.get_validation_auc()

            self.writer.add_scalar("val/epoch_rocauc", roc_auc, epoch + 1)
            self.writer.add_scalar("val/epoch_prauc", pr_auc, epoch + 1)
                
            # save model
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc
                torch.save(self.model.state_dict(),
                        os.path.join(self.model_save_path, 'best_model.pth'))
            # change optimizer
            if current_optimizer == 'adam' and drop_counter == 60:
                self.load = os.path.join(self.model_save_path, 'best_model.pth')
                self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                                 0.001, 
                                                 momentum=0.9, 
                                                 weight_decay=0.0001, 
                                                 nesterov=True)
                current_optimizer = 'sgd_1'
                drop_counter = 0
                print('sgd 1e-3')
            # first drop
            if current_optimizer == 'sgd_1' and drop_counter == 20:
                self.load = os.path.join(self.model_save_path, 'best_model.pth')
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.0001
                current_optimizer = 'sgd_2'
                drop_counter = 0
                print('sgd 1e-4')
            # second drop
            if current_optimizer == 'sgd_2' and drop_counter == 20:
                self.load = os.path.join(self.model_save_path, 'best_model.pth')
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.00001
                current_optimizer = 'sgd_3'
                print('sgd 1e-5')

        print("[%s] Train finished. Elapsed: %s"
              % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                 datetime.timedelta(seconds=time.time() - start_t)))

    def get_validation_auc(self):
        self.model.eval()
        predictions = []
        groundtruths = []
        for x,y in self.data_loader_val:

            # forward
            x = to_var(x)
            out = self.model(x)
            out = out.detach().cpu()

            predictions.append(out)

            groundtruths.append(y)

            if _i % 1000 == 0:
                print("[%s] Valid Iter [%d/%d] " %
                      (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       _i, len(self.data_loader_val)))

        roc_auc, pr_auc = get_auc(predictions, groundtruths)
        return roc_auc, pr_auc
    
if __name__ == '__main__':
    solver = Solver(main_dict)
    solver.train()