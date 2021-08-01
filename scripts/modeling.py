import pandas as pd
import sys  
# sys.path.insert(0, '../scripts')
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
# import clean_data
import loading_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import datetime
import pickle


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

# class to include custom functions on dataframes
# in the pipeline
class df_function_transformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

# handle outliers
def handle_outliers(df):
    sales_dec = df.quantile(0.10)
    sales_qua = df.quantile(0.90)
    df = np.where(df < sales_dec, sales_dec,df)
    df = np.where(df >sales_qua, sales_qua,df)
    return df

def isweekend(number):
    if number in range(1,6):
        return 0
    else:
        return 1

def isholiday(x):
    if x == "0":
        return 0
    else:
        return 1

# function to create more features from Date columns
def get_features(df_train): 
    # extracting numerical information from the date columns
    # the year
    df_train_copy = df_train.copy()
    df_train_copy["IsHoliday"] = df_train_copy["StateHoliday"].map(lambda x: isholiday(x))
    df_train_copy["Year"] = df_train_copy['Date'].dt.year
    # which part of the month it is where 0 is begining, 1 is mid and 2 is end
    df_train_copy["Part of the month"] = df_train_copy['Date'].dt.day.apply(lambda x: x // 10)
    df_train_copy.loc[(df_train_copy["Date"].dt.day == 31), "Part of the month"] = 2
    # Is Weekend
    df_train_copy["IsWeekend"] = df_train_copy["DayOfWeek"].map(lambda x: isweekend(x))
    df_train_copy = df_train_copy.drop(columns=["Open","Date","DayOfWeek","StateHoliday","SchoolHoliday"])
    return df_train_copy[["Store","IsHoliday","IsWeekend","Promo","Year","Part of the month"]]
    

# function to convert to dataframe
def format_datetime(df):
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_model_input(df,sales=True):
    if sales:
        df_y = df["Sales"]
    else:
        df_y = df["Customers"]
    df_X = df.drop(columns=["Sales","Customers"])
    return df_X, df_y

# prepare dataframe
def prepare_df(df):
    prepare_df_pipeline = Pipeline([
        ("convert_Date_format", df_function_transformer(format_datetime)),
        ("get features from Date",df_function_transformer(get_features)),
        ])
    return prepare_df_pipeline.fit_transform(df)

# preprocess data
def preprocess(df):
    numerical_preprocessing = Pipeline([('imputation', SimpleImputer())])

    # define which transformer applies to which columns
    impute_encode = ColumnTransformer([
        ('numerical_preprocessing', numerical_preprocessing, ["Store","IsHoliday","IsWeekend","Promo","Year","Part of the month"])
    ])

    training_pipeline = Pipeline([
        ("encode and impute", impute_encode)
        
    ])
    return training_pipeline.fit_transform(df)

def train_model(X,y,model):
    reg = model.fit(X, y)
    return reg

def inference_model(X,model):
    return model.predict(X)

def get_time_now():
    time =datetime.datetime.now()
    time_list = [time.day,time.month,time.year,time.hour,time.minute,time.second]
    time_now = "-".join(str(i) for i in time_list )
    return time_now


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

        # get features and target
        df_features,df_target = prepare_model_input(df_train)
        df_features = prepare_df(df_features)
        #log artifacts: columns used for modeling
        cols_x = pd.DataFrame(list(df_features.columns))
        cols_x.to_csv('features.csv', header=False, index=False)
        mlflow.log_artifact('features.csv')

        cols_y = pd.DataFrame(["Sales"])
        cols_y.to_csv('target.csv', header=False, index=False)
        mlflow.log_artifact('target.csv')

        y = np.array(handle_outliers(df_target))
        X = preprocess(df_features)
        # split into valid and training data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, random_state=12)
        
        # train with training data
        reg = LinearRegression()
        trained_model = train_model(X_train,y_train,reg)
        #run models and store parameters
        score = trained_model.score(X_valid, y_valid)
        y_pred_valid = inference_model(X_valid, trained_model)
        mae = mean_absolute_error(y_valid, y_pred_valid)

        print("The mean absolute error of our model is ",mae)
        print("The score of the trained Linear regression model is ",score)
        mlflow.log_metric('Score of model', score)
        mlflow.log_metric('MAE of model', mae)
        
        model_name = get_time_now()
        # save this dataframe to the database
        pickle.dump(trained_model, open("models/"+model_name+".pkl", 'wb'))
        print('Model Successfully Saved.!!!')

if __name__ == "__main__":
    ml_pipeline()


