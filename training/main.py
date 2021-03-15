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
from backend import Backend, Backend2
from data_loader import get_DataLoader

#os.environ["NVIDIA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# define here all the parameters
main_dict = {"frontend_dict":
             {"list_out_channels":[128,128,256,256,256,256], 
              "list_kernel_sizes":[(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)],
              "list_pool_sizes":  [(3,2),(2,2),(2,2),(2,1),(2,1),(2,1)], 
              "list_avgpool_flags":[False,False,False,False,False,True]},
             
             "backend_dict":
             {"n_class":50,
              "bert_config":None, 
              "recurrent_units":2, 
              "bidirectional":True}, #  pass recurrent_units = None to deactivate
             
             "training_dict":
             {"dataset":'msd',
              "architecture":'conv_before_encoder_5s',
              "n_epochs":1000,
              "learning_rate":1e-4},
             
             "data_loader_dict":
             {"path_to_repo":'~/dl4am/',
              "batch_size":128,
              "input_length":5, # [s]
              "spec_path":'/import/c4dm-datasets/rmri_self_att/msd',
              "audio_path":'/import/c4dm-03/Databases/songs/',
              "mode":'train', 
              "num_workers":20}}

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class AssembleModel(nn.Module):
    
    def __init__(self, main_dict=None, backend=None, frontend=None):
        super(AssembleModel, self).__init__()
        

        if main_dict is not None:
            self.frontend = Frontend_mine(main_dict["frontend_dict"]) #Frontend_won() #
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

        # Training 
        self.batch_size = main_dict["data_loader_dict"]["batch_size"]
        self.n_epochs = main_dict["training_dict"]["n_epochs"]
        self.initial_lr = main_dict["training_dict"]["learning_rate"]
        
        # Model
        self.model = AssembleModel(main_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)
        
        # Optimizer
        """see https://www.fast.ai/2018/07/02/adam-weight-decay/"""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), self.initial_lr)
        
        # Loss
        self.criterion = nn.BCELoss()
        # Model save path
        self.model_save_path = os.path.join(self.path_to_repo,"models")
        os.makedirs(self.model_save_path, exist_ok=True)
    
        # Tensorboard
        now = datetime.datetime.now()
        log_dir = os.path.join("./","logs",
                               main_dict["training_dict"]["dataset"],
                               main_dict["training_dict"]["architecture"],
                               now.strftime("%m:%d:%H:%M"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"tensorboard --logdir '{log_dir}' --port ")
        self.writer = SummaryWriter(log_dir)

    def load_parameters(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def train(self, drop_counter=0, best_roc_auc=0):

        start_t = time.time()
        current_optimizer = 'adam'
                
        for epoch in range(self.n_epochs):

            # change optimizer
            if current_optimizer == 'adam' and drop_counter == 40:
                self.load_parameters(os.path.join(self.model_save_path, 'best_model.pth'))
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
                self.load_parameters(os.path.join(self.model_save_path, 'best_model.pth'))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.0001
                current_optimizer = 'sgd_2'
                drop_counter = 0
                print('sgd 1e-4')
            # second drop
            if current_optimizer == 'sgd_2' and drop_counter == 20:
                self.load_parameters(os.path.join(self.model_save_path, 'best_model.pth'))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = 0.00001
                current_optimizer = 'sgd_3'
                print('sgd 1e-5')
            
            # train
            ctr = 0
            n_iters_per_epoch = len(self.data_loader_train)
            self.model.train()
            for x, y in self.data_loader_train:
                ctr+=1

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
                if ctr % 100 == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             epoch+1, self.n_epochs, ctr, n_iters_per_epoch, loss.item(),
                             datetime.timedelta(seconds=time.time()-start_t)))
                    self.writer.add_scalar("train/step_loss", loss.item(), epoch*n_iters_per_epoch + ctr)

            # validation
            roc_auc, pr_auc = self.validate()
            self.writer.add_scalar("val/epoch_rocauc", roc_auc, epoch + 1)
            self.writer.add_scalar("val/epoch_prauc", pr_auc, epoch + 1)
                
            # save model
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
            
            # update drop_counter
            drop_counter += 1


    def validate(self):
        self.model.eval()
        y_score = []
        y_true = []
        ctr = 0
        for x,y in self.data_loader_val:
            ctr+=1
            
            # NB: in validation mode the output of the DataLoader
            # has a shape of (1,n_chunks,F,T), where n_chunks = total time frames // input_length
            x = x.permute(1,0,2,3) 
            # by permuting it here we are treating n_chunks as the batch_size
            
            # forward
            x = to_var(x)
            out = self.model(x)
            out = out.detach().cpu()

            y_score.append(out.numpy().mean(axis=0))

            y_true.append(y.detach().numpy())

            if ctr % 1000 == 0:
                print("[%s] Valid Iter [%d/%d] " %
                      (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       ctr, len(self.data_loader_val)))
        
        y_score = np.array(y_score).squeeze()
        y_true = np.array(y_true).squeeze().astype(int)
        
        roc_auc  = metrics.roc_auc_score(y_true, y_score, average='macro')
        pr_auc = metrics.average_precision_score(y_true, y_score, average='macro')
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)
        return roc_auc, pr_auc
    
if __name__ == '__main__':
    solver = Solver(main_dict)
    #solver.train(drop_counter=40, best_roc_auc=0.8736) # pass nothing to start a new session
    solver.train()
