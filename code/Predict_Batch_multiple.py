import numpy as np
import pandas as pd
import json
import sys
import pickle
from datetime import datetime,timedelta
import os
import re
import torch
from Model_class import One_Model
import traceback
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Predict_Batch():
    def __init__(self,candidate_stock_list, model_dir_name_list, daily_info_dir, start_day = '2014-01-06',end_day = '2017-12-29', version = 0,
                interval = 5,num_total_position = 100, offset = 0.01, top = 40, pick_high = True):
        
        
        self.candidate_stock_list,unvalid_list = self.unvalid_file_detection(candidate_stock_list)
        print('invalid candidate_stock_list (empty or non-exist): ' ,unvalid_list)
        
        
        
        
        self.interval = interval
        self.num_total_position = num_total_position
        self.offset = offset
        self.top = top
        self.all_backtest = pickle.load(open(daily_info_dir,'rb'))
        self.current_position = {}
        self.last_sell_price = {}
        self.num_day_pass = 0
        self.total_num_bought = 0
        self.num_of_buy_time = 0
        self.pick_high = pick_high
        
        
        
        self.start_day = start_day
        self.end_day = end_day
        
        self.model_data,self.available_day  = self.generate_model_data()
        
            
        try:
            start_day_index = self.available_day.index(self.start_day)
            self.current_day_index = start_day_index
        except ValueError:
            for i in range(len(self.available_day) - 1):
                if (self.available_day[i] < self.start_day) and (self.available_day[i+1] > self.start_day):
                    self.current_day_index = i + 1
                    print('Warning : Start day is not a business day, move the following business day')
                    break
        
        self.current_day = self.available_day[self.current_day_index] 
        
        
        self.model_list = []
        for i in range(len(model_dir_name_list)):
            self.model_list.append(One_Model(model_dir_name_list[i], self.start_day,
                                        version[i], self.available_day, self.pick_high))
        delta_t_list = [model.delta_t for model in self.model_list]
        
        self.max_delta_t = max(delta_t_list)
        print('delta_t_list: ',delta_t_list)
        
    def unvalid_file_detection(self,file_lists):
        valid_list = []
        unvalid_list = []
        for file in file_lists:
            if (os.path.isfile(file)) and (os.stat(file).st_size > 0):
                valid_list.append(file)
            else:
                unvalid_list.append(file)
        return valid_list,unvalid_list
    
    
    def generate_model_data(self):
        print('Generating predicting data')
        sys.stdout.flush()
        model_data = {}
        available_day = set()
        start_day = (datetime.strptime(self.start_day,'%Y-%m-%d') - timedelta(days=200)).strftime('%Y-%m-%d')
        for stock_path in self.candidate_stock_list:
            data = pd.read_csv(stock_path, header= None, index_col = [0])
            data = data.dropna()
            model_data[stock_path.split('/')[-1]] = data.loc[start_day:]
            available_day = available_day.union(set(model_data[stock_path.split('/')[-1]].index))
        return model_data,sorted(available_day)
    
    
    def get_training_data(self):
        stock_name_list = []
        model_input = None
        #print(self.current_day)
        for stock_path in self.candidate_stock_list:
            stock_name = stock_path.split('/')[-1]
            #print(stock_name)
            temp_stock = self.model_data[stock_name]
            try:
                end_index = temp_stock.index.get_loc(self.current_day)
            except KeyError:
                continue
            start_index = end_index - self.max_delta_t + 1
            if start_index < 5:
                continue
            
            stock_name_list.append(stock_name)
            temp_stock = temp_stock.iloc[start_index:end_index+1,:]
            #print(temp_stock)
            temp_stock_tensor = torch.from_numpy(temp_stock.iloc[:,:160].as_matrix()).float()
            temp_stock_tensor = torch.unsqueeze(temp_stock_tensor,0)
            if model_input is None:
                model_input = temp_stock_tensor.clone()
            else:
                model_input = torch.cat([model_input,temp_stock_tensor],dim=0)
        #print(current_day,model_input.shape)
        
        
        train = model_input
        train = train[:,:,:160]
    
        print(self.current_day,train.shape)
        sys.stdout.flush()
        self.train = train.cuda()
        self.stock_name_list= stock_name_list
    
    
    
    def virtual_buy(self):
        #find which stock is not able to be sold on that day
        self.return_rate = 0
        for stock_name in list(self.last_sell_price.keys()):
            buy_price = self.last_sell_price[stock_name]
            self.return_rate -= buy_price * (1/(self.num_total_position*buy_price))
            self.current_position[stock_name] = buy_price
        self.last_sell_price = {}
        
    def sell_and_buy(self):  
        final_result = list(self.total_pred_result.items())
        
        candidate_stock_name = []
        for i in sorted(final_result,key=lambda x : x[1],reverse=False):
            candidate_stock_name.append(i[0])
            
            
        if len(self.current_position.keys()) == 0:
            need_buy = candidate_stock_name[:self.num_total_position]
        else:
            need_buy = []
            top_candidate_stock = candidate_stock_name[:self.top]
            for stock_name in top_candidate_stock:
                if stock_name not in self.current_position:
                    need_buy.append(stock_name)
                    
        need_sell = []
        if len(need_buy) > 0:
            for stock_name in candidate_stock_name[::-1]:
                if (stock_name in self.current_position.keys()):
                    need_sell.append(stock_name)
                    if len(need_sell) == len(need_buy):
                        break
                        
        if (len(need_buy) != len(need_sell)) and ((len(need_buy) - len(need_sell)) != self.num_total_position):
            need_buy = need_buy[:len(need_sell)]
        
                
        self.total_num_bought += len(need_buy)
        self.num_of_buy_time += 1
        
        print(self.current_day,'need-buy: ',len(need_buy), 'need-sell: ',len(need_sell))

        for stock_name in need_sell:
            sell_price = self.all_backtest[stock_name].loc[self.current_day,'median']
            self.return_rate += sell_price*(1/(self.num_total_position*self.current_position[stock_name]))*(1-self.offset)
            self.current_position.pop(stock_name)
        for stock_name in need_buy:
            buy_price = self.all_backtest[stock_name].loc[self.current_day,'median']
            self.return_rate -= buy_price*(1/(self.num_total_position*buy_price))*(1 + self.offset)
            self.current_position[stock_name] = buy_price
            
            
    def virtual_sell(self):
        for stock_name,buy_price in self.current_position.items():
            if stock_name in self.stock_name_list:
                sell_price = self.all_backtest[stock_name].loc[self.current_day,'close']
            else:
                sell_price = buy_price
            self.last_sell_price[stock_name] = sell_price 
            self.return_rate += sell_price * (1/(self.num_total_position*buy_price))
        self.current_position = {}
        
        
        
    def simulate(self):
        self.acutual_return = []
        while True:
            if (self.current_day < self.end_day) and (self.current_day_index <= len(self.available_day) - 1):
                self.get_training_data()
                self.virtual_buy()
                if self.num_day_pass%self.interval == 0:
                    self.total_pred_result = Counter()
                    for model in self.model_list:
                        model.get_model_date(self.current_day)
                        temp_result = model.pred(self.stock_name_list,self.train[:,self.max_delta_t - model.delta_t:,:])
                        self.total_pred_result = self.total_pred_result + temp_result
                    self.sell_and_buy()
                self.virtual_sell()
                self.acutual_return.append(self.return_rate)
                self.current_day_index += 1
                if self.current_day_index > len(self.available_day) - 1:
                    break
                self.current_day = self.available_day[self.current_day_index]
                self.num_day_pass += 1
            else:
                break
    
    


