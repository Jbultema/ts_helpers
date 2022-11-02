import typing
import pandas as pd
import numpy as np
import datetime

# from .calendar_preparation import *
from .data_preparation import *
# from .scoring import *


def create_lags(df: pd.core.frame.DataFrame, 
                ts_settings: dict,
                column: str, 
                lags: list
               ) -> pd.core.frame.DataFrame:
    '''
    Create lags of desired column. Will results in increased nan-values at head of dataframe.
    
    Inputs:
    -------
    df: DataFrame with date-ordered data
    ts_settings: Dictionary of modeling configuation
    column: Name of column for feature engineering
    lags: list of lags (int) to be created
    
    Output:
    ------
    df: DataFrame with additional columns
    '''
    for l in lags:
        df[f'{column}_lag_{l}'] = df[column].shift(l)
    return df[df[ts_settings['date_col']].notna()].copy()


def create_rolling(df: pd.core.frame.DataFrame, 
                   column: str, 
                   roll_lengths: list, 
                   method: list = ['mean']
                  ) -> pd.core.frame.DataFrame: 
    '''
    Create rolling statistics for desired column.
    
    Inputs:
    -------
    df: DataFrame with date-ordered data
    column: Name of column for feature engineering
    roll_lengths: list windows for rolling aggregation
    method: list of aggregation methods, valid entries include: 'mean', 'median', 'sum', 'std'
    
    Output:
    ------
    df: DataFrame with additional columns
    '''
    for roll in roll_lengths:
        for m in method:
            df[f'{column}_rolling_{roll}_{m}'] = df[column].rolling(roll).agg({column: m}).fillna(method='bfill').fillna(method='ffill')
    return df


def create_rolling_lags(df: pd.core.frame.DataFrame, 
                        column: str, 
                        lags: list, 
                        roll_length: list, 
                        method: list = ['mean']
                       ) -> pd.core.frame.DataFrame:
    '''
    Create rolling statistics at discrete times in the past.
    
    Inputs:
    -------
    df: DataFrame with date-ordered data
    column: Name of column for feature engineering
    lags: list of lags (int) to be created
    roll_lengths: list windows for rolling aggregation
    method: list of aggregation methods, valid entries include: 'mean', 'median', 'sum', 'std'
    
    Output:
    ------
    df: DataFrame with additional columns
    '''
    
    for l in lags:
        for roll in roll_length:
            for m in method:
                df[f'{column}_lag_{l}_rolling_{roll}_{m}'] = df[column].shift(l).rolling(roll).agg({column: m}).fillna(method='bfill').fillna(method='ffill')
    return df

# need to adjust this to work with all columns
def create_rolling_lags_many_columns(df: pd.core.frame.DataFrame,
                        columns: list,
                        lags: list, 
                        roll_length: list, 
                        method: list = ['mean']
                       ) -> pd.core.frame.DataFrame:
    '''
    Create rolling statistics at discrete times in the past.
    
    Inputs:
    -------
    df: DataFrame with date-ordered data
    column: Name of column for feature engineering
    lags: list of lags (int) to be created
    roll_lengths: list windows for rolling aggregation
    method: list of aggregation methods, valid entries include: 'mean', 'median', 'sum', 'std'
    
    Output:
    ------
    df: DataFrame with additional columns
    '''
    
    for col in columns:
        df = create_rolling_lags(df=df, column= col, lags= lags, roll_length= roll_length, method= method)
    return df

def create_differences(df: pd.core.frame.DataFrame, 
                       column: str, 
                       differences: list
                      ) -> pd.core.frame.DataFrame:
    '''
    Create differenced calculations.
    
    Inputs:
    -------
    df: DataFrame with date-ordered data
    column: Name of column for feature engineering
    differences: list of lags (int) to be used for differencing
    
    Output:
    -------
    df: DataFrame with additional columns
    '''
    for d in differences:
        df[f'{column}_{d}_difference'] = df[column] - df[column].shift(d)
    return df


def create_fd(df: pd.core.frame.DataFrame
               , ts_settings: dict
               , excluded: typing.Optional[list]= None
               , freq: typing.Optional[str]= 'D'
             ) -> pd.core.frame.DataFrame:
    '''
    Duplicates and shifts dataset by forecast distance. Required for forecast distance modeling.
    Be careful with size expansion of datasets, especially with long forecast horizon. This version uses the new approach with cross-join and shift_dates.
    
    Inputs:
    -------
    df: DataFrame with date-ordered data
    ts_settings: dictionary of project settings
    excluded: (Optional) list of columns to not shift   
    
    Output:
    -------
    output_df: DataFrame, linear increase in size by forecast horizon
    '''
#     output_df = pd.DataFrame()
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    target = ts_settings['target']
    
    col_order = df.columns.tolist()

    # determine which columns will shift and which won't
    if series_id is not None:
        shift_cols = list(set(df.columns).difference(set([date_col, series_id])))
        omit_cols = [date_col, series_id]

        if excluded is not None:
            shift_cols = list(set(shift_cols).difference(set(excluded)))
            omit_cols += excluded

            
    # create fd dataset
    df_fd = pd.DataFrame({'FD': range(ts_settings['fd_start'], ts_settings['fd_end']+1)})
    date_series = df[[x for x in df.columns if x != target]].copy()
    df_fd['key']= '0'
    date_series['key']= '0'
    output_df = date_series.merge(df_fd, how= 'outer', left_on='key', right_on='key').drop('key', axis=1)
        
    # loop through each forecast distance, shift date_col, and create proper FD key
    tmp_df = pd.DataFrame()
    for fd in tqdm(range(ts_settings['fd_start'], ts_settings['fd_end']+1), leave= True, desc= 'Preparing FD dataset'):
        # shift date col and merge back into original data
        if series_id is not None:
            loop_df = df[[date_col, series_id, target]].copy()
        else:
            loop_df = df[[date_col, target]].copy()

        loop_df['FD'] = fd
        if freq == 'D':
            loop_df['shifted_date'] = loop_df[date_col] + pd.DateOffset(days= -1*(fd))
        elif freq == 'W':
            loop_df['shifted_date'] = loop_df[date_col] + pd.DateOffset(weeks= -1*(fd))
        elif freq == 'M':
            loop_df['shifted_date'] = loop_df[date_col] + pd.DateOffset(months= -1*(fd))
        else:
            return 'Frequency is not detected, please specify "D", "W", "M", or manually change the code'

        tmp_df = tmp_df.append(loop_df)
    
    # cleanup tmp_df a bit
    if series_id is not None:
        tmp_df.drop_duplicates([date_col, series_id, 'FD'], inplace= True)
    else:
        tmp_df.drop_duplicates([date_col, 'FD'], inplace= True)
    tmp_df.drop(date_col, axis=1, inplace= True)


    # merge shifted_dates on original data into the fd-dataset
    output_df = output_df.merge(tmp_df, how= 'left', left_on= [date_col, series_id, 'FD'], right_on= ['shifted_date', series_id, 'FD'])
#     output_df.drop('shifted_date', axis=1, inplace= True)
    
    # fix column order, and exclude tail-data without a full forecast horizon (all FDs)
    last_date = output_df[output_df[ts_settings['target']].notna()][date_col].max()
    try:
        output_df = output_df[['FD'] + df.columns.tolist()].sort_values([date_col, series_id, 'FD'])
        output_df = output_df[(output_df[date_col]<= last_date) & (output_df[ts_settings['target']].notna())]#.drop_duplicates([date_col, series_id, 'FD'])

    except: 
        output_df = output_df[['FD'] + df.columns.tolist()].sort_values([date_col, 'FD'])
        output_df = output_df[(output_df[date_col]<= last_date) & (output_df[ts_settings['target']].notna())]#.drop_duplicates([date_col, 'FD'])

    return output_df


def add_date_features(df: pd.core.frame.DataFrame
                      , ts_settings: dict
                     ) -> pd.core.frame.DataFrame:
    """
    Add date-features to dataset using the primary date-time column.
    
    Inputs:
    -------
    df: training dataset
    ts_settings: dictionary of modeling settings
    
    Output:
    -------
    df: training dataset with new date-features engineered
    
    """
    shift_name = [x for x in df.columns if 'shift' in x]
    if len(shift_name) == 1:
        date_col = shift_name[0]
    else:
        date_col = ts_settings['date_col']
    
    df[date_col] = pd.to_datetime(df[date_col])
    df['day'] = df[date_col].dt.dayofweek
    df['week'] = df[date_col].dt.week
    df['year'] = df[date_col].dt.year
    df['doy'] = df[date_col].dt.dayofyear
    df['dom'] = df[date_col].dt.day.values
    
    return df


# older approach, very slow
# def create_fd(df: pd.core.frame.DataFrame, 
#               ts_settings: dict,
#               excluded: typing.Optional[list] = None
#              ) -> pd.core.frame.DataFrame:
#     '''
#     Duplicates and shifts dataset by forecast distance. Required for forecast distance modeling.
#     Be careful with size expansion of datasets, especially with long forecast horizon.
    
#     Inputs:
#     -------
#     df: DataFrame with date-ordered data
#     ts_settings: dictionary of project settings
#     excluded: (Optional) list of columns to not shift   
    
#     Output:
#     -------
#     output_df: DataFrame, linear increase in size by forecast horizon
#     '''
#     output_df = pd.DataFrame()
#     date_col = ts_settings['date_col']
#     series_id = ts_settings['series_id']
    
#     # loop through each forecast distance, shift all columns except date_col and series_id
#     for fd in range(ts_settings['fd_start'], ts_settings['fd_end']+1):
        
#         # determine which columns will shift and which won't
#         if series_id is not None:
#             shift_cols = list(set(df.columns).difference(set([date_col, series_id])))
#             omit_cols = [date_col, series_id]
            
#             if excluded is not None:
#                 shift_cols = list(set(shift_cols).difference(set(excluded)))
#                 omit_cols += excluded
            
#             # shift the future values back for each series
#             tmp_df = pd.DataFrame()
#             for series in tqdm(df[series_id].unique()):
#                 series_df = df[df[series_id]==series].copy()
#                 series_df = series_df[shift_cols].shift(-(fd-1))
#                 series_df['FD'] = fd
                
#                 # add the non-shifted columns back in
#                 for col in omit_cols:
#                     series_df[col] = df[df[series_id]==series][col]
                    
#                 # fix column order
#                 series_df = series_df[['FD'] + omit_cols + shift_cols]

#                 # add the series-iteration data for each FD-iteration
#                 tmp_df = tmp_df.append(series_df)

#         else:
#             shift_cols = list(set(df.columns).difference(set([date_col]))) 
#             omit_cols = [date_col]
            
#             if excluded is not None:
#                 shift_cols = list(set(shift_cols).difference(set(excluded)))
#                 omit_cols += excluded
                
#             # shift the future values back
#             tmp_df = df.copy()
#             tmp_df = tmp_df[shift_cols].shift(-(fd-1))
#             tmp_df['FD'] = fd
        
#             # add the non-shifted columns back in
#             for col in omit_cols:
#                 tmp_df[col] = df[col]
                
#         # add the FD-iteration data to output
#         output_df = output_df.append(tmp_df)
        
#     # adding new things to fill gaps created by FD
#     # sort the dataset, trying to use series_id
#     try:
#         output_df = output_df[['FD'] + df.columns.tolist()].sort_values([date_col, series_id, 'FD'])
#         output_df.groupby([date_col, series_id]).fillna(method= 'ffill').fillna(method= 'bfill', inplace= True)
#         output_df.reset_index(drop= True, inplace= True)
#     except: 
#         output_df = output_df[['FD'] + df.columns.tolist()].sort_values([date_col, 'FD'])
#         output_df.groupby([date_col]).fillna(method= 'ffill').fillna(method= 'bfill', inplace= True)
#         output_df.reset_index(drop= True, inplace= True)

#     return output_df


def prepare_ts_dataset(df: pd.core.frame.DataFrame,
                       date_col: str, 
                       target: str,
                       bhg: typing.Optional[int]= -1,
                       series_id: typing.Optional[str]= None,
                       KIAs: typing.Optional[list]= None,
                       freq: typing.Optional[str]= 'D'
                      ):
    
    '''
    Lags all non-KIA values to prevent ts-data-leakage.
    
    Inputs:
    -------
    df: Pandas dataframe training dataset,
    date_col: name of primary date-column,
    target: name of target-column,
    bhg: number of steps into the past to lag non-KIA features (blind history gap)
    series_id: name of primary series-id-column,
    KIA: list of feature names or variables for which values are known at prediction time or known in advance (KIA)
    
    Output:
    df: Pandas dataframe with lagged values
    '''
    assert(bhg <= 0), 'bhg must be zero or negative'
    
    # retain original order to ensure output matches input
    order = df.columns.tolist()
    
    # add target, date_col, and series_id to KIA list for shifting
    if series_id:
        KIAs += [target, date_col, series_id]
    else:
        KIAs += [target, date_col]
    non_KIAs = [x for x in df.columns if x not in KIAs]
    output_df = df.copy()
    
    # shift non-KIAs (easy way)
    df[non_KIAs] = df[non_KIAs].shift(-1 * bhg) #, freq= freq)

    return output_df[order]


def engineer_ts_dataset(df: pd.core.frame.DataFrame,
                        ts_settings: dict,
                        engineer_columns: list,
                        lag: bool= True,
                        rolling: bool= True,
                        rolling_windows: bool= True,
                        difference: bool= True,
                        fill_missing: bool= True,
                        fd_modeling: bool= False,
                        date_features: bool= True,
                        verbose: bool= True,
                        date_freq: typing.Optional[str]= 'D',
                        bhg: typing.Optional[int]= -1,
                        lags: typing.Optional[list]= [1,7,14,28],
                        roll_method: typing.Optional[list]= ['mean'],
                        roll_lengths: typing.Optional[list]= [3,7],
                        roll_window_method: typing.Optional[list]= ['mean'],
                        differences: typing.Optional[list] = [1,7,28],
                        numeric_fill_columns: typing.Optional[list]= [],
                        categorical_fill_columns: typing.Optional[list]= [],
                        fill_numeric_method: typing.Optional[typing.Union[str, dict]]= 'rolling_median',
                        fill_categorical_method: typing.Optional[typing.Union[str, dict]]= 'most recent'
) -> pd.core.frame.DataFrame:
    
    '''
    Wrapper function to prepare TS dataset in single step.
    
    Inputs:
    -------
    df: Pandas dataframe with training data
    ts_settings: dictionary of modeling settings
    engineer_columns: list of columns to perform feature engineering on
    lag: (bool) if engineer_column values should be lagged
    rolling: (bool) if engineer_column values should have rolling statistics created
    rolling_windows: (bool) if engineer_column values should have rolling-window statistics created
    difference: (bool) if engineer column values should have differences calculated,
    fill_missing: (bool) if missing values in all columns should be filled,
    fd_modeling: (bool) if a forecast-distance dataset should be created,
    bhg: (Optional - negative int) blind-history gap that corresponds to how far non-KIA data should be shifted,
    lags: (Optional - list) list of lags to create for engineer_columns,
    roll_method: (Optional - list) list of types of calculations to performe for rolling statistics. Accepts values that work with pandas .rolling().method(),
    roll_length: (Optional - list) length of windows in past for rolling statistics,
    roll_window_method: (Optional - list) list of types of calculations to performe for rolling-window statistics. Accepts values that work with pandas .rolling().method(),
    differences: (Optional - list) list of lags to be used for calculation of differences,
    numeric_fill_columns: (Optional - list) list of columns with numeric values to fill,
    categorical_fill_columns: (Optional - list) list of columns with categorical values to fill,
    fill_numeric_method: (Optional - str, dict) method to be used to fill numeric columns. Accepts values that work with pandas .rolling().method(). String for single method or dict{column_name: method},
    fill_categorical_method: (Optional - str, dict) method to be used to fill categorical columns. Accepts 'most recent' or 'most common'. String for single method or dict{column_name: method}
    
    Output:
    -------
    df: Pandas dataframe with numerous changes
    '''
    # define convenience variables from ts_settings dict
    date_col = ts_settings['date_col'], 
    target = ts_settings['date_col'],
    series_id = ts_settings['series_id']
    
    output_df = df.copy()
    
    # engineer different TS features
    if lag:
        if verbose:
            print(f'*** Creating lags: {lags} ***')
        for col in engineer_columns:
            output_df = create_lags(df= output_df, 
                    ts_settings= ts_settings,
                    column= col, 
                    lags= lags
                    )
            
    if rolling:
        if verbose:
            print(f'*** Creating rolling statistics {roll_method} with historical {roll_lengths} rows ***')
        for col in engineer_columns:
            output_df = create_rolling(df= output_df, 
                                       column= col, 
                                       roll_lengths= roll_lengths, 
                                       method= roll_method
                                      )

    if rolling_windows:
        if verbose:
            print(f'*** Creating rolling-window statistics {roll_method} for lags {lags} with historical {roll_lengths} rows ***')
        output_df = create_rolling_lags_many_columns(df= output_df,
                                                    columns= engineer_columns,
                                                    lags= lags, 
                                                    roll_length= roll_lengths, 
                                                    method= roll_method
                                                   )

    if difference:
        # this method is not written to work simultanously on multiple columns
        # loop through in multiple columns are supplied as engineer_columns
        if verbose:
            print(f'*** Differences columns {differences} for {engineer_columns} ***')
        for col in engineer_columns:
            output_df = create_differences(df= output_df, 
                                           column= col, 
                                           differences= differences
                                          )

    if fd_modeling:
        if verbose:
            print(f"*** Creating Forecast Distance dataset with steps from {ts_settings['fd_start']} to {ts_settings['fd_end']}, using new approach ***")
        pre = output_df.shape[0]
        output_df = create_fd(df= output_df, 
                              ts_settings= ts_settings,
                              excluded= ts_settings['known_in_advance'],
                              freq= date_freq
                             )
        if verbose:
            print(f'*** Caution: Dataset increased in size by {np.round(100*output_df.shape[0]/pre, 0)}% ***')

        
    if fill_missing:
        if verbose:
            print(f'*** Filling missing values in numeric columns: {numeric_fill_columns}, and categorical columns: {categorical_fill_columns} ***')
        output_df = fill_nans(df= output_df,
                             ts_settings= ts_settings,
                             numeric_columns= numeric_fill_columns,
                             numeric_method= fill_numeric_method,
                             roll_length= roll_lengths,
                             cat_columns= categorical_fill_columns,
                             cat_method= fill_categorical_method
                             )

        assert(df.columns.tolist() in output_df.columns.tolist()), 'We lost some columns'
        
    if date_features:
        if verbose:
            print(f'*** Adding date-features to dataset ***')
        output_df = add_date_features(df= output_df,
                                     ts_settings= ts_settings)
                  
    # shift values as appropriate
    if verbose:
        print('*** Preparing Time Series Dataset ***')
    output_df = prepare_ts_dataset(df= output_df,
                       date_col= date_col, 
                       target= target,
                       bhg= bhg,
                       series_id= series_id,
                       KIAs= ts_settings['known_in_advance'].copy(),
                       freq = date_freq
                      )
    
    # drop the shift_date column if it exists
    shift_name = [x for x in output_df.column if 'shift' in x]
    if len(shift_name) != 0:
        try:
            output_df.drop(shift_name[0], axis=1, inplace= True)
        except:
            return output_df
    
    return output_df#[output_df[date_col].notna()].copy()
