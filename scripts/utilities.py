
import pandas as pd

def format_float(value):
    return f'{value:,.2f}'

def format_datetime(df,col_name):
    df[col_name] = pd.to_datetime(df[col_name])
    return df

def format_numeric(df,col_name):
    df[col_name] = pd.to_numeric(df[col_name])
    return df

def convert_ms_to_hrs(df,col_name):
    """ This function converts millisecond to hours"""
    hr = 3600000
    df[col_name+" in hrs"] = df[col_name] / hr
    
    return df[col_name+" in hrs"]


def find_agg(df:pd.DataFrame, agg_column:str, wanted_col:str,agg_metric:str, col_name:str, top:int, order=False )->pd.DataFrame:
    """ This function calculates aggregate of column """
    new_df = df.groupby(agg_column)[wanted_col].agg(agg_metric).reset_index(name=col_name).\
    sort_values(by=col_name, ascending=order)[:top]
    
    return new_df

pd.options.display.float_format = format_float