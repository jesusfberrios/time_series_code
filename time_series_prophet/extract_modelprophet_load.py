import pandas as pd
import time
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
import os
from prophet import Prophet
import itertools
import numpy as np
import pandas as pd
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics



engine_str = 'xxxx'
engine_db = create_engine(engine_str)

csv_list = [ 'xxxx']

db_names_sources=csv_list
print('Starting Extraction process')
df_all_list = []
bus = []
xls_sheets = []

for i in db_names_sources:

    ## Connect to db and extract info in df
    pass
        
df_all_list = [df_jcl_21,df_j_21,...]
table_names = ['xx']





# For prophet
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'weekly_seasonality': [False]
}
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
################################################################


# k=0
# df_iter = df_all_list[0]
# j='general'
# m='general'
# n='mean_nps'


for k, df_iter in enumerate(df_all_list):
    engine_str = 'XXXX'
    engine_db = create_engine(engine_str)
    
    col_names = []
    for i in df_iter.columns:
        try:
            df_iter[i].astype(int)
            if (df_iter[i].unique().size<=11) and (df_iter[i].max()<=10):
                col_names.append(i)
        except:
            pass
    
    df_metrics_flat_all = pd.DataFrame()
    
    
    for j in ['general','store','gender_sex','customer_sex','gse']:
        
        df_aux = df_iter[['transaction_date']+col_names].copy()
        
        try:
            df_aux['group_id'] = j
            df_aux['group_name'] = df_iter[j] if j!='general' else 'general'
    
    
            df_metrics = df_aux[['transaction_date']+col_names+['group_id','group_name']]\
                            .groupby(['transaction_date','group_id','group_name'])\
                                .agg(['mean','median','std','min','max','count'])#[col_names]
    
            #print(df_metrics.columns)
            df_metrics_flat = pd.DataFrame(index=df_metrics.index)
    
            for i in df_metrics.columns:
                df_metrics_flat[i[1]+'_'+i[0]] = df_metrics[i]
    
            df_metrics_flat.reset_index(inplace=True)
    
            for m in df_metrics_flat.group_name.unique():
                df_m_up = df_metrics_flat[df_metrics_flat.group_name==m].copy()
    
                
                for n in ['mean_'+x for x in col_names]:
                    df_m_u = df_m_up[['transaction_date',n]].copy()
                    df_m_u.columns = ['ds','y']
    
                    # rmses = []  # Store the RMSEs for each params here
                    # for params in all_params:
                    #     mod = Prophet(**params).fit(df_m_u)  # Fit model with given params
                    #     df_cv = cross_validation(mod, horizon='14 days', parallel="processes")
                    #     df_p = performance_metrics(df_cv, rolling_window=0)
                    #     rmses.append(df_p['rmse'].values[0])
    
                    # # Find the best parameters
                    # tuning_results = pd.DataFrame(all_params)
                    # tuning_results['rmse'] = rmses
                    # best_params = all_params[np.argmin(rmses)]
                    prophet = Prophet(weekly_seasonality=False,changepoint_prior_scale=0.1)
                    #prophet.add_seasonality(name='yearly', period=330, fourier_order=5)
                    prophet.fit(df_m_u)
                    # create a future data frame 
                    future = prophet.make_future_dataframe(periods=14)
                    forecast = prophet.predict(future)
    
                    #Adjust dataframe to store
                    df_m_u.ds = pd.to_datetime(df_m_u.ds)
    
                    df_store = forecast[['ds','trend','yhat_lower','yhat_upper']].merge(df_m_u,how='left',left_on='ds',right_on='ds')
                    df_store['indicator'] = n
                    df_store['group_id'] = j
                    df_store['group_name'] = m
                    df_store['anomaly'] = np.where((df_store.y<df_store.yhat_lower)|(df_store.y>df_store.yhat_upper),1,0)
                    print(table_names[k],df_store.group_name.unique())
    
                    #Store dataframe
                    df_store.to_sql(schema='ts', name=f'prophet_{table_names[k]}_2021', con=engine_db, index=False, if_exists='append')
    
                    #Store best params
                    # df_bp = pd.DataFrame(best_params,index=[0])
                    # df_bp['bu'] = table_names[k]
                    # df_bp['indicator'] = n
                    # df_bp['group_id'] = j
                    # df_bp['group_name'] = m
                    # df_bp.to_sql(schema='ts', name='prophet_params_2021', con=engine_db, index=False, if_exists='append')
                    
            df_metrics_flat_all = df_metrics_flat_all.append(df_metrics_flat)
        
        except:
            print(f'Indicator {j} does not exist in {table_names[k]}')

#     df_metrics_flat_all.to_sql(schema='ts', name=f'{table_names[k]}_2021', 
#                   con=engine_db, index=True, if_exists='replace')
    
















