import pandas as pd
import sys  
# sys.path.insert(0, '../scripts')
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import clean_data
import loading_data
import utilities
from sklearn.neighbors import LocalOutlierFactor
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
import dvc.api
from urllib.parse import urlparse

def load_versions(path,version):
    repo = './'
    data_url = dvc.api.get_url(
        path=path,
        repo=repo,
        rev=version
    )

    df = loading_data.load_csv(data_url)

    return df,data_url


def ml_pipeline():
    
    mlflow.set_experiment('pharmaceutical')
    with mlflow.start_run():
        # loading data 
        path = 'data/train.csv'
        version='trainV1'

        df_train,data_url = load_versions(path,version)
        
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', df_train.shape[0])
        mlflow.log_param('input_cols', df_train.shape[1])

        # partition data to features and target
        df_train = utilities.format_datetime(df_train,"Date")
        # df_test = utilities.format_datetime(df_train,"Date")
        # cleaned_test = clean_data.Handle_missing_values(df_test,drop_cols=False,drop_rows=False)

        # removing date and get only year value
        df_train_copy = df_train.copy()
        df_train_copy["Year"] = df_train_copy['Date'].dt.year
        df_training = df_train_copy.drop(columns='Date')
        # encode state holidays to numbers
        holidays = {'0': 0, 'a': 1, 'b':2, 'c':3}
        df_training["StateHoliday"] = df_training['StateHoliday'].map(lambda x: holidays[x])
        y_df = df_training['Sales']
        x_df = df_training.drop(columns='Sales')

        #log artifacts: columns used for modeling
        cols_x = pd.DataFrame(list(x_df.columns))
        cols_x.to_csv('features.csv', header=False, index=False)
        mlflow.log_artifact('features.csv')

        cols_y = pd.DataFrame(["Sales"])
        cols_y.to_csv('target.csv', header=False, index=False)
        mlflow.log_artifact('target.csv')

        #run models and store parameters
        
        y_train = np.array(y_df)
        X_train = np.array(x_df)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        # clf_decisions = tree.DecisionTreeClassifier()
        # decision_clf_trained = clf_decisions.fit(X_train,y_train)

if __name__ == "__main__":
    ml_pipeline()


