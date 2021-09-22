# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression




def Read_Excel():
    
    Der = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 1)
    # print(Der)

    Fund = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 2)
    # print(Fund)

    Bond = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 3)
    # print(Bond)

    Eq = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 4)
    # print(Eq)
    
    # print(Der.shape, Fund.shape, Bond.shape, Eq.shape)

    return Der,Fund,Bond,Eq
    


def Clean_Data(Der,Fund,Bond,Eq):
    
    # Delete unnamed columns
    
    Der.drop(Der.columns[Der.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    Fund.drop(Fund.columns[Fund.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    Bond.drop(Bond.columns[Bond.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    Eq.drop(Eq.columns[Eq.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    # print(Der.shape, Fund.shape, Bond.shape, Eq.shape)
    
    # Delete duplicate columns
    
    Der = Der.T.drop_duplicates().T
    Fund = Fund.T.drop_duplicates().T
    Bond = Bond.T.drop_duplicates().T
    Eq = Eq.T.drop_duplicates().T  
    
def Merge_tables(Fund,Bond,Eq):
    # Create a datraframe list
    # Instrument_list = [Der,Eq,Bond,Fund]
    Instrument_list = [Eq,Bond,Fund]
    Instrument_df = pd.concat(Instrument_list)
    
    return Instrument_df
    
def Read_Excel_Time():
    
    Der_data = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 8)

    Fund_data = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 9)

    Bond_data = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 10)

    Eq_data = pd.read_excel("O:\AFOstockstoscreeneclectic.xlsm" , sheet_name = 11)
    
    # print(Der_data.shape, Fund_data.shape, Bond_data.shape, Eq_data.shape)

    return Der_data,Fund_data,Bond_data,Eq_data

def Clean_Data_Time(Der_data,Fund_data,Bond_data,Eq_data):
    
    # Delete unnamed columns
    
    Der_data.drop(Der_data.columns[Der_data.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    Fund_data.drop(Fund_data.columns[Fund_data.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    Bond_data.drop(Bond_data.columns[Bond_data.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    Eq_data.drop(Eq_data.columns[Eq_data.columns.str.contains('unnamed',case =False)],axis=1,inplace=True)
    
    # print(Der_data.shape, Fund_data.shape, Bond_data.shape, Eq_data.shape)
    
    # Fill NaN values
    # Backwards fill
    Der_data.bfill(inplace=True)
    Fund_data.bfill(inplace=True)
    Bond_data.bfill(inplace=True)
    Eq_data.bfill(inplace=True)
    
    # Forwards fill
    Der_data.ffill(inplace=True)
    Fund_data.ffill(inplace=True)
    Bond_data.ffill(inplace=True)
    Eq_data.ffill(inplace=True)

def Merge_tables_time(Fund_data,Bond_data,Eq_data):
    # Create a datraframe list
    
    # Instrument_list_time = [Der_data,Eq_data,Bond_data,Fund_data]
    # Without derivatives 
    Instrument_list_time = [Eq_data,Bond_data,Fund_data]
    Instrument_df_time = pd.concat(Instrument_list_time, axis=1)
    
    return Instrument_df_time

# Calling functions

# Characteristics
def Get_data():
    Der,Fund,Bond,Eq = Read_Excel()
    Clean_Data(Der,Fund,Bond,Eq)
    Instrument_df = Merge_tables(Fund,Bond,Eq)
    
    # Time Series
    Der_data, Fund_data, Bond_data, Eq_data = Read_Excel_Time()
    Clean_Data_Time(Der_data,Fund_data,Bond_data,Eq_data)
    Instrument_df_time = Merge_tables_time(Fund_data,Bond_data,Eq_data)
    
    
    # Clear Instrument df (Want to only have floats)
    Instrument_df_time.drop([0,1], inplace = True)
    Instrument_df_time.drop(['Date'], axis = 1, inplace = True )
    # Instrument_df_time.reset_index(drop = True)
    
    Der_data.drop([0,1], inplace = True)
    Der_data.drop(['Date'], axis = 1, inplace = True )
    # Eq_data.drop('Date',axis=1,inplace=True)
    # Eq_data.drop([0,1], inplace = True)
    
    return Der_data , Instrument_df_time

Der_data , Instrument_df_time = Get_data()

def get_weights():
    
    Weights_csv = pd.read_excel("O:\Eqfull1.xlsm" , sheet_name = 0)
    Weights = Weights_csv[['Name','ISIN','W8','RIC']]
    Weights.dropna(inplace=True)
    return Weights

Weights = get_weights()


#  Problem Solved ???

# # Key_list = Instrument_df[['ISIN','RIC','NAME']]


def Beta_matrix_linear_func(start_date, end_date):
    
    # Setting an array of Zeros (Correct size)
    # No_eq = len(Eq_data.columns)
    
    No_inst = len(Instrument_df_time.columns)
    No_der = len(Der_data.columns)

    Beta_list = np.zeros(shape = (No_inst, No_der))
    
    for j in range (1,No_inst + 1):
        for i in range (1,No_der + 1):
            # Getting Time-Series data for Equities and Instruments
            Inst_1 = Instrument_df_time.iloc[start_date:end_date,j-1]
            Inst_2 = Der_data.iloc[start_date:end_date,i-1]
            
            New = pd.concat([Inst_1, Inst_2],axis=1)
            # New_new = New.drop([0,1])
            price_change = New.pct_change()
            data_b = price_change.dropna()

            # Set x and y values to conduct Linear Regression
            x = np.array(data_b.iloc[:,1]).reshape((-1,1))
            y = np.array(data_b.iloc[:,0]).reshape((-1,1))
            # Run Linear Regression
            model = LinearRegression().fit(x,y)
            # Compatiblity issues at [i,i] points
            if len(model.coef_) != 1:
                Beta_list[j-1][i-1] = 1
            else:
                Beta_list[j-1][i-1] = (model.coef_)
    
    # Truning the array to a dataframe
    Beta_matrix = pd.DataFrame(Beta_list,index = Instrument_df_time.columns.to_list(),columns = Der_data.columns.to_list())
    
    return Beta_matrix

# Beta_matrix = Beta_matrix_linear_func()

def Large_beta():
    
    arr = Beta_matrix.values
    index_names = Beta_matrix.index
    col_names = Beta_matrix.columns
    
    R,C = np.where(arr>3)
    
    out_arr = np.column_stack((index_names[R],col_names[C],arr[R,C]))
    Large_beta = pd.DataFrame(out_arr,columns=[['Equity','Instrument','value']])
    return Large_beta

# Large_beta = Large_beta()


def Beta_matrix_mult_func(start_date , end_date):
    
    # Setting a correct size array
    R2_mat = []
    No_inst = len(Instrument_df_time.columns)
    No_der = len(Der_data.columns)
    
    Beta_list_mult = np.zeros(shape = (No_inst, No_der))
    
    for i in range (1 , No_inst + 1):
    
        # Split into dependent and independent variables
        x = Der_data.iloc[start_date:end_date]

        y = Instrument_df_time.iloc[start_date:end_date,i-1]
        
        x = x.pct_change()
        y = y.pct_change()
        
        x.dropna(inplace = True)
        y.dropna(inplace = True)
        
        
        # Split data set into traning and test data
        
        x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=0 )
        
        # Train the model on the dataset
        
        model = LinearRegression()
        model.fit(x_train,y_train)
        
        y_pred = model.predict(x_test)
        R2_mat.append(r2_score(y_test,y_pred))
        
        # Get p-values
        # print("coef_pval: ", stats.coef_pval(model,x_train,y_train))
        
        for j in range(1,No_der+1):
            
            Beta_list_mult[i-1][j-1] = model.coef_[j-1]
        
        # for j in range(1,No_inst-1):
        #     if j != i:
        #         Beta_list_mult[i-1][j-1] = model.coef_[j]
        #     else:
        #         Beta_list_mult[i-1][j] = model.coef_[j]
        
    # Truning the array to a dataframe
    Beta_matrix = pd.DataFrame(Beta_list_mult, index = Instrument_df_time.columns.to_list(),columns = Der_data.columns.to_list())
    
   
    return Beta_matrix , pd.DataFrame(R2_mat)
 

def Large_beta_mult():
    
    arr = Beta_matrix_mult.values
    index_names = Beta_matrix_mult.index
    col_names = Beta_matrix_mult.columns
    
    R,C = np.where(arr>10)
    
    out_arr = np.column_stack((index_names[R],col_names[C],arr[R,C]))
    Large_beta = pd.DataFrame(out_arr,columns=[['Instrument_1','Instrument_2','value']])
    
    return Large_beta

# Large_beta_mult = Large_beta_mult()

# Beta_matrix_mult_1 = Beta_matrix_mult_func(0,200)

# Get portofolio betas
def Find_common_el(Beta_matrix_mult_1):
    # Lists to find common elements
    list_ind = np.array(Beta_matrix_mult_1.index.tolist())
    weights_list = np.array(Weights.iloc[:,3])
    Port_beta = pd.DataFrame()
    Port_beta_adj = pd.DataFrame(())
    # If common  elements exist, append their betas to another df
    for i in range(0,len(list_ind)):
        if list_ind[i] in weights_list:
            entry = Beta_matrix_mult_1.iloc[i,:]
            Port_beta_adj = Port_beta.append(entry)
            Port_beta = Port_beta.append(entry)
    return Port_beta, Port_beta_adj

# Get portofolio time series  
def Clear_port_time(start_date,end_date, Beta_matrix_mult_1):
    
    # Lists to find common elements
    list_ind = np.array(Beta_matrix_mult_1.index.tolist())
    weights_list = np.array(Weights.iloc[:,3])
    Port_time = pd.DataFrame()
    Port_time_adj = pd.DataFrame()
    
    # If common  elements exist, append them to another df
    for i in range(0,len(list_ind)):
        if list_ind[i] in weights_list:
            entry = Instrument_df_time.iloc[start_date:end_date,i]
            Port_time_adj = Port_time.append(entry)
            Port_time = Port_time.append(entry)

            Port_time_adj = Port_time_adj.pct_change(axis = 1)
            
    return Port_time.transpose(), Port_time_adj.transpose()

# Get portofolio weights
def Clear_weights_port(Port_beta):
    port_list = np.array(Port_beta.index.tolist())
    weights_list = np.array(Weights.iloc[:,3])
    Port_weights = pd.DataFrame()
    for i in range(0,len(weights_list)):
        if weights_list[i] in port_list:
            entry = Weights.iloc[i,2:4]
            Port_weights = Port_weights.append(entry)
    return Port_weights
    

# #  Get weight adjusted portofolio
def Port_w(Beta_matrix_mult_1,Port_weights,Port_beta_adj):
    list_ind = np.array(Beta_matrix_mult_1.index.tolist())
    for i in range(0,len(Port_weights)):
        if Port_weights.iloc[i,0] in list_ind:
            # Time Series weighted
            # Port_time_adj[Port_weights.iloc[i,0]] = Port_time_adj[Port_weights.iloc[i,0]] * Port_weights.iloc[i,1]
            # Beta weighted
            Port_beta_adj.loc[Port_weights.iloc[i,0],:] = Port_beta_adj.loc[Port_weights.iloc[i,0],:] * Port_weights.iloc[i,1]
   
            
    return Port_beta_adj

# Port_time_adj, Port_beta_adj = Port_w()

# Get plots for portofolio returns

# Port_return = np.array(Port_time_adj.sum(axis=1))

# Der_data_adj = Der_data.copy()
# Index_list = Port_beta_sum_adj.index.tolist()

def Index_data_adj(start_date,end_date,Port_beta_sum_adj):
   
    Der_data_adj = Der_data.pct_change()
    Der_data_adj.dropna(inplace = True)
    
    Index_list = Port_beta_sum_adj.index.tolist()
    
    # Multiply the correct indices by the correct betas
    for i in range(0,len(Port_beta_sum_adj)):
        if Index_list[i] in Der_data_adj:
            Der_data_adj[Index_list[i]] = Der_data_adj[Index_list[i]] * Port_beta_sum_adj[i]
    
    # Get beta adjusted portofolio return
    Port_index_return_adj_sum = Der_data_adj.sum(axis = 1)
    
    # Get dates we are interested in
    Port_index_return_adj_sum = Port_index_return_adj_sum.iloc[start_date:end_date]
    
    return Port_index_return_adj_sum

# Port_index_return_adj = Index_data_adj()


def Get_beta_series(start_date, end_date):
    
    start_date = start_date
    end_date = end_date
    
    # Get Multi Regression Betas
    Beta_matrix_mult_1 , R2 = Beta_matrix_mult_func(start_date , end_date)
    
    #  Get portofolio betas i.e. Find common elements between csv file and portofolio
    Port_beta, Port_beta_adj = Find_common_el( Beta_matrix_mult_1)
    
    
    # Get portofolio weights
    Port_weights = Clear_weights_port(Port_beta)
    
    # Get weight adjusted beta 
    Port_beta_adj = Port_w(Beta_matrix_mult_1,Port_weights,Port_beta_adj)
    
    # Get portofolio beta sum by index
    Port_beta_sum_adj = Port_beta_adj.sum(axis=0)
            
    return Port_beta_sum_adj , Port_weights , Beta_matrix_mult_1 , R2




def Get_port_return(start_date, end_date):
    # Get time series for dates wanted 
    Port_time, Port_time_adj = Clear_port_time(start_date, end_date, Beta_matrix_mult_1)
    Port_return_plot = Port_time_adj.dropna(inplace = True)
    # Get weight adjusted time series for those dates
    list_ind = np.array(Beta_matrix_mult_1.index.tolist())
    for i in range(0,len(Port_weights)):
        if Port_weights.iloc[i,0] in list_ind:
            # Time Series weighted
            Port_time_adj[Port_weights.iloc[i,0]] = Port_time_adj[Port_weights.iloc[i,0]] * Port_weights.iloc[i,1]
    
    Port_return = pd.DataFrame(np.array(Port_time_adj.sum(axis=1)))
    
    return Port_return

# Port_return = Port_return.reset_index()
# Port_return.drop(['index'], axis =1 , inplace = True)
# Get index returns
def Get_index_data(start_date,end_date):
    Der_data_adj = Der_data.pct_change()
    Der_data_adj.dropna(inplace = True)
    Der_data_adj = Der_data_adj.iloc[start_date:end_date]
    Der_data_adj = Der_data_adj.reset_index()
    Der_data_adj.drop(['index'], axis =1 , inplace = True)

    return Der_data_adj

# Hedge = pd.DataFrame(np.dot(Der_data_adj,Port_beta_table_140.T))
# Hedge = (Der_data_adj.mul(Port_beta_table_140))


x = 0
i = 0
No = 10

time_window = 10

end_date = 1001 - No 
start_date = end_date - time_window

Port_beta_table = pd.DataFrame()
R2_table = pd.DataFrame()

# Der_data = Der_data[['ATF' , 'SPX' , 'STOXX50E' ]] 
# Der_data_adj = Der_data[['ATF' , 'SPX' , 'STOXX50E' ]] 

Beta_matrix_1 , R2 = Beta_matrix_mult_func(0,1001)

while i < No :
    Port_beta_sum_adj , Port_weights, Beta_matrix_mult_1 , R2 = Get_beta_series(start_date + i, end_date + i)

    
    Port_beta_table = Port_beta_table.append(Port_beta_sum_adj, ignore_index = True)
    # R2_table = pd.concat((R2, ignore_index = True)
    i = i + 1
 
# Port_beta_table , Port_weights, Beta_matrix_mult_1 , R2 = Get_beta_series(97, 297)

Port_return = Get_port_return(1001 - No , 1001)
Der_data_adj = Get_index_data(1001 - No , 1001)

Index_der = Der_data_adj.T.index.to_list()
Index_beta = Port_beta_table.T.index.tolist()

New = pd.DataFrame()

for i in range (0,len(Der_data.T)):
    for j in range(0,len(Der_data.T)):
        if Index_der[i] == Index_beta[j]:
            New = New.append(Der_data_adj[Index_der[i]] * Port_beta_table[Index_beta[j]] )
  
            
# test=pd.DataFrame(np.sum((Port_beta_table.values*Der_data_adj.values),axis=1) )

Returns = New.sum()

Def Export_to_excel(Port_beta_table, Der_data_adj,  Returns, Port_return ):

with pd.ExcelWriter('Results_test_tw_250.xlsx') as writer:
    
    Port_beta_table.to_excel(writer, sheet_name = 'Port_beta_table')

    Der_data_adj.to_excel(writer, sheet_name = 'Der_data_adj')
    
    Returns.to_excel(writer, sheet_name = 'Hedge')

    Port_return.to_excel(writer, sheet_name = 'Port_returns')


Export_to_excel(Port_beta_table, Der_data_adj,  Returns, Port_return )


