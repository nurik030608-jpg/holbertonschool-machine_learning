#!/usr/bin/env python3
import pandas as pd

def rename(df):
    # 1. Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})
    
    # 2. Convert the 'Datetime' values to actual datetime objects
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 3. Filter the DataFrame to keep only 'Datetime' and 'Close'
    df = df[['Datetime', 'Close']]
    
    # 4. Return the modified DataFrame
    return df
