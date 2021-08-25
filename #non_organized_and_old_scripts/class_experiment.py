# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:49:20 2021

@author: colompa
"""

class logger_dataframe:
    import pandas as pd
    list_files: list
    df: pd.DataFrame
    threshold: int
    
    def dataset(self):
        for file in list_files:
            load = pd.read_csv(file, sep = ';', header = 0)
            if len(load)>=threshold:
                load = load.iloc[:,:2]
                daily = daily_mean(load, 0)
                x = daily.index.strftime('%Y-%m-%d')
                y = daily.values
                name = os.path.basename(file)[1:-4]   
                df[name] = np.nan
                df.loc[df['date'].isin(x), name] = y[x.isin(df.iloc[:,0])]
        return df
           