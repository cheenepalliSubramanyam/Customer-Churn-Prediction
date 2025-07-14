import polars as pl
import pandas as pd
import numpy as np

def reduce_memory_usage(df : pl.DataFrame):
    for col in df.columns:
        if(df[col].dtype==pl.Int64):
            c_max=df[col].max() if df[col].max() else 0
            c_min = df[col].min() if df[col].min() else 0
            if(c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max):
                df=df.with_columns(pl.col(col).cast(pl.Int8))
            elif(c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max):
                df=df.with_columns(pl.col(col).cast(pl.Int16))
            elif(c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max):
                df=df.with_columns(pl.col(col).cast(pl.Int16))
            elif(c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max):
                df=df.with_columns(pl.col(col).cast(pl.Int32))
        if(df[col].dtype==pl.Boolean):
            df=df.with_columns(pl.col(col).cast(pl.Int8))
        if(df[col].dtype==pl.Float64):
            c_max=df[col].max() if df[col].max() else 0
            c_min=df[col].min() if df[col].min() else 0
            if(c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max):
                df=df.with_columns(pl.col(col).cast(pl.Float32))
    return df

def handle_missing(df : pl.DataFrame,threshold : pl.Float32 =  0.5):
    length=len(df)
    for col in df.columns:
        if((df[col].null_count()/length)>threshold):
            df=df.drop(col)
    return df