#!/usr/bin/env python3
import pandas as pd

def rename(df):
    # 1. Rename the 'Timestamp' column to 'Datetime'
    df = df.rename(columns={'Timestamp': 'Datetime'})
    
    # 2. Convert to datetime specifying the unit as seconds ('s')
    # This corrects the 1970 vs 2019 discrepancy
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    
    # 3. Filter to display only the Datetime and Close columns
    df = df[['Datetime', 'Close']]
    
    return df
