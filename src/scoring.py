import pandas as pd
import numpy as np
from tqdm import tqdm
import typing

from .calendar_preparation import *
# from .ts_feature_engineering import *
from .data_preparation import *


    
def create_scoring_dataset(config: dict,
                           actuals_path: str
                           , historical_path: str
                           , ts_settings: dict 
                           , freq: typing.Optional[str]= None
                           , scoring_request: typing.Optional[str]= None
                           , updated_cal_path: typing.Optional[str]= None
                          ) -> pd.core.frame.DataFrame:
    '''
    Perform checks and create valid scoring dataset. Designed for CST forecasting project, but should generalize.
    
    Inputs:
    -------
    config: project-level configuration json with model, data paths, etc
    actuals_path: file path to most recent actuals
    historical_path: file path to previous actuals / training data
    ts_settings: dictionary of modeling-level details
    freq: (Optional) date-frequency in dataset
    scoring_request: (Optional) forecast point from which a forecast is desired
    updated_cal_path: (Optional) file path to updated calendar file
    '''
    
    # ingest the recent-actuals and historical-training data
    try: 
        actuals = read_file(path= actuals_path
                            , file_type= get_file_type(actuals_path)
                            , dates= [ts_settings['date_col']])
        historicals = read_file(path= historical_path
                            , file_type= get_file_type(historical_path)
                            , dates= [ts_settings['date_col']])
    except: 
        return 'There are problems with the actual or historical files. Cannot be ingested in current format'
    
    # parse the training request to generate a forecast point
    if scoring_request != None:
        fp = scoring_request
    else:
        fp = actuals[ts_settings['date_col']].max()
    
    
    # get the dataset frequency
    if freq != None:
        f1= check_gap(df= actuals
                      , ts_settings= ts_settings)
        f2= check_gap(df= historical
                      , ts_settings= ts_settings)
        
        if f1 != f2:
            return 'The actuals and historicals dataset have different row spacing'
        elif 'Cannot' in f1 | 'Cannot' in f2:
            # if there is an error, return the error from the bad check_gap
            # the error is a longer string
            return max([f1, f2], ken= len)
        else:
            freq= f1
    
    # check if the dataset columns match
    if check_columns(actuals, historicals) == 'Sort':
        actuals = actuals[historicals.columns]
    elif type(check_columns(actuals, historicals)) == list:
        return f'There are problems with columns in actuals or historicals: {check_columns(actuals, historicals)}'
        
    # check if there is enough FDW data
    fdw_actuals = check_fdw(actuals
                            , fp
                            , ts_settings)
    fdw_historicals = check_fdw(historicals
                            , fp
                            , ts_settings)
    
    if fdw_actuals:
        fdw_data = actuals[actuals[ts_settings['date_col']] >= fdw_actuals].copy()
    elif fdw_historicals:
        fdw_data = historicals[historicals[ts_settings['date_col']] >= fdw_actuals].copy()
        # in case there are extra datas needed in actuals, automatically add them in
        fdw_data = fdw_data.append(actuals[actuals[ts_settings['date_col']] >= fdw_actuals]).drop_duplicates()
    else:
        return f"Insufficient actual or historical data provided for FDW of {ts_settings['fdw_start']} {freq}"
        
    # create the scoring dataset
    scoring_data = fdw_data[fdw_data[ts_settings['date_col']] <= fp].copy()
#     scoring_data.loc[scoring_data[scoring_data[ts_settings['date_col']] > fp].index, ts_settings['target']] = np.nan
    
    # create and add the future dates dataframe
    future_dates = pd.date_range(start= (fp + pd.DateOffset(1, freq))
                                 , end= (fp + pd.DateOffset(ts_settings['fd_end'], freq))
                                 , freq= freq)
    tmp = pd.DataFrame(future_dates
                       , columns = [ts_settings['date_col']])
    # augment with calendar events
    tmp = add_calendar_events(ts_settings= ts_settings
                            , df= tmp
                            , historical_cal_path= config['calendar_path']
                            , updated_cal_path= updated_cal_path)

    scoring_data = scoring_data.append(tmp)#.drop_duplicates()
    
    # add KIA values to new rows
    for col in ts_settings['known_in_advance']:
        # this is a custom adjustment to this CST project
        if col == 'Day':
            # this should be improved to automatically detect the first day of year
            scoring_data[col] = scoring_data[ts_settings['date_col']].dt.day_of_year.astype(int) + 3
        elif col == 'Week':
            scoring_data[col] = scoring_data[ts_settings['date_col']].dt.week
        else:
            scoring_data[col].fillna(0, inplace= True)
    
    return scoring_data
