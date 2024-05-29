from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
import pickle



def main():
    filename = 'model.pkl'
    model = pickle.load(open(filename, 'rb'))
    dir = './test'
    df = None
    if os.path.isdir(dir):
        for i in os.listdir(dir):
            if i.find('.csv') != -1:
                df = pd.concat([df,pd.read_csv(dir+'/'+i,index_col=0)]) if df is not None else pd.read_csv(dir+'/'+i,index_col=0)
        X = df
        y_pred = model.predict(X)
        print(y_pred)

if __name__ == "__main__":
    main()