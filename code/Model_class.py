import numpy as np
import pandas as pd
import json
import sys
import pickle
from datetime import datetime,timedelta
import os
import re
import torch
from Rnn import Reg_Rnn_160_2, Binary_Rnn_160_2, Reg_Rnn_160_1, Binary_Rnn_160_1
import traceback
from collections import Counter



class One_Model():
    def __init__(self, model_dir_name, start_day, version, available_day, pick_high = True):
        '''
        normal_stock_list : directory of stocks used to be predicted. The format of the file should be in mins
        candidate_stock_list : directory of stocks used to be predicted. The format of the file should be in 3mins interval
        major_stock_files: directory of stocks used to train model (should be similar to the stock used to be predicted)
        minor_stock_files: directory of stocks used to train model (these stock may not be used to make prediction, be considered as supplement to training data)
        '''
        
    
        self.version = version
        
        self.model_dir_name = model_dir_name
        
        
        
        
        self.start_day = start_day
        self.model_start_day_list = self.get_model_start_day_list()
        self.available_day = available_day
        self.pick_high = pick_high
        #pickle.dump(self.model_data,open('../temp/model_data','wb'))
        
        
        if self.start_day < self.model_start_day_list[0]:
            print('Error : start_day out of range')
            exit()
        elif self.start_day < self.available_day[self.delta_t]:
            print('Error : start_day out of range, the earliest preditable day is : ',self.available_day[self.delta_t])
            exit()
            
    
    def get_model_start_day_list(self):
        files = [file for file in os.listdir(self.model_dir_name) if re.findall('^[A-Za-z]+_Rnn_160_[1,2]_'+str(self.version)+'_+[0-9]+-[0-9]+-[0-9]+$',file)]
        model_start_day_list = [name.split('_')[-1] for name in files]
        model_start_day_list = set(model_start_day_list)
        model_start_day_list = sorted(model_start_day_list)
        self.delta_t = torch.load(self.model_dir_name + files[0]).delta_t
        return model_start_day_list
    
    
    def get_model_date(self, current_day):
        self.current_day = current_day
        for model_start_day in self.model_start_day_list[::-1]:
            if self.current_day >= model_start_day:
                model_date = model_start_day
                break
        model_file_names = [file_name for file_name in os.listdir(self.model_dir_name) if re.findall('^[A-Za-z]+_Rnn_160_[1,2]_'+str(self.version)+'_'+ model_date +'$', file_name)]
        assert len(model_file_names) == 1, 'Error: can not find or find multiple desired model: ' + str(model_file_names)
        self.model_path = self.model_dir_name + model_file_names[0]
        self.model = torch.load(self.model_path)
        
        
    def pred(self, stock_name_list, train):
        print(train.shape, self.model_path)
        sys.stdout.flush()
        self.model.eval()
        
        pred = self.model(data = train)
        pred = list(pred.cpu().detach().numpy().astype('float64'))
        
        candidate_stock_name = []
        
        for i in sorted(zip(stock_name_list,pred),key=lambda x : x[1], reverse=self.pick_high):
            candidate_stock_name.append(i[0])
        rank = dict([(j,i**2) for i,j in enumerate(candidate_stock_name)])
        return Counter(rank)
            
        
    
        


        
