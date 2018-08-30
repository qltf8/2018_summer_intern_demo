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
    
    

def process_dir(dir_name):
    dir_name = dir_name.replace('\\','/')
    if dir_name[-1] != '/':
        dir_name = dir_name + '/'
    return dir_name

if __name__ == '__main__':
    
    #print(sys.argv)
    if len(sys.argv) < 2:
        #major_stock_files = [file for file in os.listdir('../500_3mins_wv_positive/') if re.findall("^[a-zA-Z0-9]+.+\.csv$",file)]
        #major_stock_files = ['../500_3mins_wv_positive/'+file for file in major_stock_files]
        #minor_stock_files = [file for file in os.listdir('../300_3mins_wv_positive/') if re.findall("^[a-zA-Z0-9]+.+\.csv$",file)]
        #minor_stock_files = ['../300_3mins_wv_positive/'+file for file in minor_stock_files]
        
        candidate_stock_list = candidate_stock_list = pickle.load(open('/home/datacenter/deepLearn/test/temp_data/major_stock_files.dump','rb'))
        
        paras_dict = {'candidate_stock_list' : candidate_stock_list,
         'model_dir_name' : ['/home/datacenter/deepLearn/test/model/10/',
                             '/home/datacenter/deepLearn/models/670790a8-718e-4e96-8f90-bd474b27ec0a@Model/'],
         'start_day' : '2014-01-01',
         'end_day' : '2017-12-30', 
         'version' : 30,
         'interval' : 1,
         'num_total_position' : 100,
         'offset': 0.0015,
         'top' : 20,
         'daily_info_dir' : '/home/datacenter/deepLearn/test/temp_data/500_daily_info',
         'benchmark_file_name': '/home/datacenter/deepLearn/test/temp_data/benchmark_500.csv'}
        
        
        model_version = 10
        
    else:
        try:
            paras = sys.argv[1]
            paras = paras.replace('\\','')
            paras_dict = json.loads(paras)
        except Exception as e:
            print('Error: ', e)
    try:
        
        
        
        candidate_stock_list = paras_dict['candidate_stock_list']
        model_dir_name_list = paras_dict['model_dir_name']
        if isinstance(model_dir_name_list,str):
            model_dir_name_list = [model_dir_name_list]
        elif isinstance(model_dir_name_list,list):
            pass
        else:
            assert False, 'Error: bad input for model_dir_name, ' + str(model_dir_name_list)
        
        model_dir_name_list = [process_dir(path) for path in model_dir_name_list]
            
        start_day = paras_dict['start_day']
        end_day = paras_dict['end_day']
        version = paras_dict.get('version',15)
        interval = paras_dict['interval']
        num_total_position = paras_dict['num_total_position']
        offset = paras_dict['offset']
        top = paras_dict['top']
        daily_info_dir = paras_dict['daily_info_dir']
        benchmark_file_name = paras_dict['benchmark_file_name']
        pick_high = paras_dict.get('pick_high', True)
            
    except Exception as e:
        print('Error: ', e)


    try:
        
        print('candidate_stock_list: ',len(candidate_stock_list),
              '\n start_day: ', start_day,
              '\n end_day: ', end_day,
              '\n model_dir_name_list: ',model_dir_name_list,
              '\n version: ',version)
        for version in [[60,60]]:
            
            benchmark = pd.read_csv(benchmark_file_name,header=0,index_col=[0])
            benchmark = benchmark.loc[start_day:end_day]
            
            
            pred = Predict_Batch(candidate_stock_list = candidate_stock_list, model_dir_name_list = model_dir_name_list, 
                        start_day=start_day, end_day = end_day,  version = version, interval = interval, num_total_position = num_total_position, offset = offset,  
                        top = top, daily_info_dir = daily_info_dir, pick_high = pick_high)
            
            acutual_return = pred.simulate()
            actural = [0]
            #start_actural = 1
            for i in acutual_return:
                actural.append(actural[-1] + i)
            
            benchmark_return = list(benchmark.iloc[:,-1])
            #benchmark_start = 1
            benchmark = [0]
            for i in benchmark_return:
                #benchmark_start *= (1+i)
                benchmark.append(benchmark[-1] + i)
                
                
            f, (ax1, ax2) = plt.subplots(2, sharex=True)
            f.set_figheight(14)
            f.set_figwidth(30)
            ax1.set_title(start_day+'   '+end_day + ' average_trancation: ' + str(pred.total_num_bought/pred.num_of_buy_time) + '  return: ' +  str(actural[-1]) +  ' offset: '  + str(offset) + ' version: '+ str(version) )
            ax1.plot(np.array(actural) - np.array(benchmark),label='difference, exchange_interval:' + str(interval))
            major_ticks = ax1.get_xticks()
            minor_ticks = np.arange(major_ticks.min(),major_ticks.max(),5)
            ax1.set_xticks(minor_ticks, minor=True)
            ax1.grid(True)
            ax1.grid(which='minor', alpha=0.2)
            ax1.legend()
            ax2.plot(benchmark,label = 'benchmark')
            ax2.plot(actural,label = 'actural')
            ax2.grid(True)
            ax2.grid(which='minor', alpha=0.2)
            ax2.legend()
            f.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            fig_name = '500_' + start_day + '_' + end_day + "_" + 'v' + str(version)
            f.savefig('/home/datacenter/deepLearn/test/figure/'+str(model_version)+'/'+fig_name)
            
    except Exception as e:
        print('Error: ', e)
        print(traceback.format_exc())
