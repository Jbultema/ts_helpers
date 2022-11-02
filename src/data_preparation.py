import typing
import json
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm




def read_file(path: str
              , file_type: str
              , dates: list
              , header: typing.Optional[int]= None
             ):
    '''
    Imports a data file type with minimal parameters provided
    '''
    assert file_type in ['csv', 'xls', 'xlsx'], 'file type must be csv, xls, or xlsx format'
    
    # import files of different types
    # check to ensure index seems to be set correctly
    if file_type == 'csv':
        df = pd.read_csv(path, parse_dates= dates)
        if 'untitled: 0' in [x.lower() for x in df.columns]:
            idx = [i for i,x in enumerate(df.columns) if 'untitled' in x]
            assert len(idx) == 1, 'There are naming-convention issues with you file. Untitled columns detected'
            df = pd.read_csv(path, parse_dates= dates, index_col= idx[0])

#             # check for functional header rows
#             if len(idx) > 1:
#                 for i in range(5):
#                     try:
#                         df = pd.read_csv(path, parse_dates= dates, index_col= idx[0], header=i)
#                     except i == max(range(5)):
#                         return f'There are file format issues, likely related to inconsistent use of header between actuals and historicals'
            
        elif 'index' in [x.lower() for x in df.columns]:
            idx = [i for i,x in enumerate(df.columns) if 'index' in x]
            assert len(idx) == 1, 'There are naming-convention issues with you file. Multiple index-columns detected'
            df = pd.read_csv(path, parse_dates= dates, index_col = idx[0])

    else:
        df = pd.read_excel(path, parse_dates= dates)
        if 'untitled: 0' in [x.lower() for x in df.columns]:
            idx = [i for i,x in enumerate(df.columns) if 'untitled: 0' in x]
            assert len(idx) == 1, 'There are naming-convention issues with you file. Untitled columns detected'
            df = pd.read_excel(path, parse_dates= dates, index_col= idx[0])

#             # check for functional header rows
#             if len(idx) > 1:
#                 for i in range(5):
#                     try:
#                         df = pd.read_excel(path, parse_dates= dates, index_col= idx[0], header=i)
#                     except i == max(range(5)):
#                         return f'There are file format issues, likely related to inconsistent use of header between actuals and historicals'
            
        elif 'index' in [x.lower() for x in df.columns]:
            idx = [i for i,x in enumerate(df.columns) if 'index' in x]
            assert len(idx) == 1, 'There are naming-convention issues with you file. Multiple index-columns detected'
            df = pd.read_excel(path, parse_dates= dates, index_col = idx[0])

    return df


def get_file_type(path_string: str
                  , sep: str= '.'
                 ):
    '''
    Parse the string name to get the file type
    '''
    return path_string.split(sep)[-1]


def check_gap(df: pd.core.frame.DataFrame
              , ts_settings: dict
             ):
    '''
    Determine spacing of time-steps from dataframe
    '''
    median_gap = int(df[ts_settings['date_col']].diff().median().days)
    if median_gap == 1:
        freq = 'days'
    elif median_gap == 7:
        freq= 'weeks'
    elif median_gap == 30:
        freq= 'months'
    else:
        return 'Cannot parse the time-step from the provided data. Please provide the frequency explicitly'
    return freq


def check_fdw(df: pd.core.frame.DataFrame
              , fp: str, ts_settings: dict
              , freq: typing.Optional['str']= None
             ):
    '''
    Check if sufficient FDW data is present.
    '''
    if freq is None:
        freq= check_gap(df, ts_settings)
    fdw_date = fp + pd.DateOffset(ts_settings['fdw_start'], freq)
    if fdw_date in df[ts_settings['date_col']].unique():
        return fdw_date
    else:
        return False

    
def check_columns(df1: pd.core.frame.DataFrame
                  , df2: pd.core.frame.DataFrame
                 ):
    '''
    Compare two dataframes to see if columns match
    '''
    if all(df1.columns == df2.columns):
        return True
    elif len(list(set(df1.columns).difference(set(df2.columns)))) == 0:
        return 'Sort'
    else:
        return list(set(df1.columns).difference(set(df2.columns)))


def calc_summary_stats(df: pd.core.frame.DataFrame,
                       date_col: str,
                       target: str,
                       series_id= False
                       
) -> dict:
    """
    Analyze time series data to perform checks and gather summary statistics prior to modeling.
    
    Inputs:
    ------
    df: Pandas dataframe with training data
    date_col: Name of primary datetime column
    target: Name of target column
    series_id: Name of multi-series ID column, or False
    
    Ouput:
    ------
    stats: Dictionary with various statistics on the dataframe
    """

    df[date_col] = pd.to_datetime(df[date_col])
    if series_id:
        df.sort_values(by=[date_col, series_id], ascending=True, inplace=True)
    else:
        df.sort_values(by=[date_col], ascending=True, inplace=True)

    # Create dictionary of helpful statistics
    stats = dict()
    stats['date_col'] = date_col
    stats['target'] = target
    stats['series_id'] = series_id
    stats['rows'] = df.shape[0]
    stats['columns'] = df.shape[1]
    stats['min_target'] = df[target].min()
    stats['max_target'] = df[target].max()
    stats['start_date'] = df[date_col].min()
    stats['end_date'] = df[date_col].max()
    stats['timespan'] = stats['end_date'] - stats['start_date']
    
    if series_id:
        stats['series'] = len(df[series_id].unique())
        stats['median_timestep'] = df.groupby([series_id])[date_col].diff().median()
        stats['min_timestep'] = df.groupby([series_id])[date_col].diff().min()
        stats['max_timestep'] = df.groupby([series_id])[date_col].diff().max()
    else:
        stats['median_timestep'] = df[date_col].diff().median()
        stats['min_timestep'] = df[date_col].diff().min()
        stats['max_timestep'] = df[date_col].diff().max()
    
    if series_id:
        # create data for histogram of series lengths
        stats['series_length'] = (
            df.groupby([series_id])[date_col].apply(lambda x: x.max() - x.min())
            / stats['median_timestep']
        )

        # calculate max gap per series
        stats['series_max_gap'] = (
            df.groupby([series_id])[date_col].apply(lambda x: x.diff().max())
            / stats['median_timestep']
        )
    return stats
    
    
def calc_time_steps(df: pd.core.frame.DataFrame,
                    stats: dict
) -> dict:
    """
    Calculate timesteps per series
    
    Inputs:
    ------
    df: Pandas dataframe with training data
    stats: Output of calc_summary_stats function
    
    Ouput:
    ------
    stats: Dictionary with additional statistics on the dataframe
    """
    # calculate the timestep between rows
    if stats['series_id']:
        timesteps = df.groupby([stats['series_id']])[stats['date_col']].diff() / stats['median_timestep']
        stats['series_time_steps'] = timesteps.mean()
        stats['time_step'] = np.nan

    else:
        timesteps = df[stats['date_col']].diff() / stats['median_timestep']
        stats['time_step'] = timesteps.mean()
        stats['series_time_steps'] = np.nan
    
    return stats


def create_splits(df: pd.core.frame.DataFrame, 
                  date_col: str, 
                  target: str,
                  num_validations: int, 
                  training_duration: int, 
                  validation_duration: int, 
                  holdout: bool = True,
                  gap: int = 0, 
                  series_id= False,
                  specific_dates= False,
                  split_pct= False,
                  trim_leading_zeros: bool = True,
                  trim_lagging_zeros: bool = True
) -> dict: 
    '''
    A function to create backtests for partitioning time_series modeling.
    
    Inputs:
    ------
    df: Pandas dataframe with training data
    date_col: Name of primary-datetime column in dataframe
    target: Name of target-column in dataframe
    num_validations: Number of backtests to create (excluding holdout)
    training_duration: Number of rows for the training data
    validation_duration: Number of rows for the validation data
    holdout: Bool, generate holdout with most recent data
    gap: Number of rows between end of training data and start of validation
    specific dates: Bool, or a list of tuples with 
        (training_start, training_stop, validation_start, validation_stop) for each backtest
    split_pct: False or float. Percent of data to be used for training in training/validation-folds
    trim_leading_zeros: controls if zero-target or missing-target values before first real-observation are removed
    trim_leading_zeros: controls if zero-target or missing-target values after last real-observation are removed

        
    Output:
    ------
    df_dict: nested dictionary of Pandas DataFrames. 
        Ex: {'backtest1': {'training': backtest1_training, 'validation': backtest1_validation}}
    '''
    
    # create the output-dict and intermediate dicts
    df_dict = {}
    training_dict = {}
    validation_dict = {}
    
    # explicitly set datetime format
    # ambiguity in date-formatting often causes problems
    df[date_col] = pd.to_datetime(df[date_col])
    
    # trimming processing
    df_raw = df.copy()
    
    if trim_leading_zeros:
        first1 = df[(df[target].notna())][date_col].min()
        first2 = df[(df[target]!= 0)][date_col].min()
        first_real = max(first1, first2)
        df = df[df[date_col] > first_real].copy()
        print('Leading bad rows removed: ', df_raw.shape[0]-df.shape[0])
        
    if trim_lagging_zeros:
        last_observation = df[(df[target].notna()) | (df[target]!= 0)][date_col].max()
        last1 = df[(df[target].notna())][date_col].max()
        last2 = df[(df[target]!= 0)][date_col].max()
        last_real = min(last1, last2)
        df = df[df[date_col] < last_real].copy()
        print('Lagging bad rows removed: ', df_raw.shape[0]-df.shape[0])

    # calculate stats on the data
    stats = calc_summary_stats(df, date_col, target, series_id)
    stats = calc_time_steps(df, stats)
    df_dict['stats'] = stats
    
    for i in tqdm(range(0, num_validations + holdout), leave= True, desc= 'Creating Backtests'):
        
        # calculate date boundaries
        if i == 0:
            # set dates for first backtest
            max_validation_date = df[date_col].max()
            min_validation_date = pd.to_datetime(max_validation_date - pd.DateOffset(validation_duration) + pd.DateOffset(1)) 
            max_train_date = pd.to_datetime(min_validation_date - pd.DateOffset(gap) - pd.DateOffset(1))
            min_train_date = pd.to_datetime(max_train_date - pd.DateOffset(training_duration-1))
            
        else:
            # use the date values from the previous backtest to determine these
            max_validation_date = max_validation_date - pd.DateOffset(validation_duration)
            min_validation_date = pd.to_datetime(max_validation_date - pd.DateOffset(validation_duration) + pd.DateOffset(1)) 
            max_train_date = pd.to_datetime(min_validation_date - pd.DateOffset(gap) - pd.DateOffset(1))
            min_train_date = pd.to_datetime(max_train_date - pd.DateOffset(training_duration-1))

        ### split the data ###

        # if using split_pct, perform based on % of total data
        # this option is desired for use with a single backtest for internal hyperparameter tuning
        if split_pct:
            cut_point = pd.to_datetime(df[date_col].max() - pd.DateOffset(int(stats['rows']*(1-split_pct))))
            validation = df[df[date_col] >= cut_point].copy()
            training = df[df[date_col] < cut_point].copy()

        # make the split based on specific values provided as list of tuples
        elif specific_dates:
            backtest_dates = specific_dates[i]
            training = df[(df[date_col] >= pd.to_datetime(backtest_dates[0])) & (df[date_col] <= pd.to_datetime(backtest_dates[1]))].copy()
            validation = df[(df[date_col] >= pd.to_datetime(backtest_dates[2])) & (df[date_col] <= pd.to_datetime(backtest_dates[3]))].copy()

        # make the split based on provided durations
        else:
            validation = df[(df[date_col] >= min_validation_date) & (df[date_col] <= max_validation_date)].copy()
            training = df[(df[date_col] >= min_train_date) & (df[date_col] <= max_train_date)].copy()

        # add results to output-dict
        if holdout and i == 0:   
            training_dict['holdout'] = training
            validation_dict['holdout'] = validation
            tqdm.write('*** Holdout Created ***')
        
        elif holdout and i != 0:
            training_dict[f'backtest{i}'] = training
            validation_dict[f'backtest{i}'] = validation
            tqdm.write(f'*** Backtest{i} Created ***')

        else:
            training_dict[f'backtest{i+1}'] = training
            validation_dict[f'backtest{i+1}'] = validation
            tqdm.write(f'*** Backtest{i+1} Created ***')
    
    # combine the dictionaries for output
    df_dict['training'] = training_dict
    df_dict['validation'] = validation_dict
    
    return df_dict


def fill_row_gaps_old(df: pd.core.frame.DataFrame,
                  ts_settings: dict,
                  agg_method: typing.Optional[typing.Union[str,dict]] = 'sum',
                  fill_method: typing.Optional[str] = 'ffill',
                  static_columns: typing.Optional[list]= None,
                  freq: typing.Optional[str] = None
                 ) -> pd.core.frame.DataFrame:
    '''
    Function to fill date-gaps between records. Uses the pivot_table, reindex, and melt approach.
    
    Inputs:
    -------
    df: Pandas dataframe with training data
    ts_settings: Dictionary of settings
    agg_method: (Optional) method to use to aggregate into desired frequency. Can use a dict with method for each column.
    fill_method: (Optional) method to use to fill missing values covariates columns after reindexing
    static_columns: (Optional) list of columns that are constant for all dates
    freq: (Optional) spacing between rows. Can be determined automatically
    
    Output:
    -------
    df: Pandas dataframe with gaps filled in training data
    '''
    assert(agg_method in ['sum', 'mean']), "agg_method parameter allowed values are 'sum' or 'mean'"
    
    # define some convenience variables
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    target = ts_settings['target']
    
    # fix date-typing
    df[date_col] = pd.to_datetime(df[date_col])
    
    # get all available dates
    if freq == None:
        freq = check_gap(df, ts_settings)[0].upper()
        assert(freq != 'C'), 'Date frequency cannot be automatically detected, please provide it explicity'
    dates = pd.date_range(start= df[date_col].min(), end= df[date_col].max(), freq= freq)
#     old_dates = [pd.to_datetime(x) for x in df[date_col].unique()]
#     new_dates = [pd.to_datetime(x) for x in dates]
#     print(f'Filling from {df[date_col].min().date()} to {df[date_col].max().date()}, which will add {len(set(old_dates).difference(new_dates))} dates')
    
    # pivot dataset
    if series_id != None:
        df_wide = df.pivot_table(
            index= series_id,
            columns= date_col,
            values= target,
            aggfunc= agg_method,
            fill_value= 0
            ).reindex(dates, axis=1, fill_value= 0).reset_index()
        
        # change back to long-format,
        df_long = pd.melt(df_wide, id_vars=[series_id], var_name= date_col, value_name= target)#.rename(columns= {''#.drop('value', axis=1)
        
    else:
        df_wide = df.pivot_table(
            index= None,
            columns= date_col,
            values= target,
            aggfunc= agg_method,
            fill_value= 0
            ).reindex(dates, axis=1, fill_value= 0).reset_index()
        
        # change back to long-format,
        df_long = pd.melt(df_wide, id_vars=None, var_name= date_col, value_name= target).iloc[1:,:]

    # join back any of the other column values
    df_long[date_col] = pd.to_datetime(df_long[date_col])

    if series_id != None:
        df_merge = df_long.merge(df[[x for x in df.columns if x != target]], on= [date_col, series_id], how= 'left').fillna(method= fill_method)
    else:
        df_merge = df_long.merge(df[[x for x in df.columns if x != target]], on= [date_col], how= 'left').fillna(method= fill_method)
    

    # fill static_column values
    if static_columns != None:
    
        if series_id != None:
            static_values = df.groupby(series_id)[static_columns].agg(lambda x: x.value_counts().index[0]).to_dict()
            for c in static_columns:
                df_merge[c] = df_merge[series_id].map(static_values[c])
        else:
            df_merge[static_columns] = df_merge[static_columns].fillna(method= 'bfill').fillna(method= 'ffill')
                    
    return df_merge


def fill_row_gaps(df: pd.core.frame.DataFrame,
                  ts_settings: dict,
                  agg_method: typing.Optional[typing.Union[str,dict]] = 'sum',
                  fill_method: typing.Optional[str] = 'ffill',
                  target_fill: typing.Optional[typing.Union[str,int]]= 0,
                   static_columns: typing.Optional[list]= None,
                  freq: typing.Optional[str] = None
                 ) -> pd.core.frame.DataFrame:
    '''
    Function to fill date-gaps between records. Uses the cross-join approach on series-specific start and end date ranges.
    
    Inputs:
    -------
    df: Pandas dataframe with training data
    ts_settings: Dictionary of settings
    agg_method: (Optional) method to use to aggregate into desired frequency. Can use a dict with method for each column.
    fill_method: (Optional) method to use to fill missing values covariates columns after reindexing
    target_fill: (Optional) method to use to fill missing target values
    static_columns: (Optional) list of columns that are constant for all dates
    freq: (Optional) spacing between rows. Can be determined automatically
    
    Output:
    -------
    df: Pandas dataframe with gaps filled in training data
    '''
    assert(agg_method in ['sum', 'mean']), "agg_method parameter allowed values are 'sum' or 'mean'"
    
    # define some convenience variables
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    target = ts_settings['target']
    
    # fix date-typing
    df[date_col] = pd.to_datetime(df[date_col])
    
    # get all available dates
    if freq == None:
        freq = check_gap(df, ts_settings)[0].upper()
        assert(freq != 'C'), 'Date frequency cannot be automatically detected, please provide it explicity'
        
    # fill in missing date-steps
    if series_id != None:
        # build relevant date-steps with series-start and series-stop bounds
        first_obs = df[df[target] != 0].groupby(series_id)[date_col].min().to_dict()
        last_obs = df[df[target] != 0].groupby(series_id)[date_col].max().to_dict()
        series_dates = pd.DataFrame([first_obs, last_obs]).T.reset_index()
        series_dates.columns= [series_id, 'start_date', 'end_date']
        all_dates = pd.DataFrame(pd.date_range(start= df[date_col].min(), end= df[date_col].max(), freq= freq), columns = [date_col])

        # Cross join in pandas
        all_dates['key'] = 0
        series_dates['key'] = 0
        cross = all_dates.merge(series_dates, on='key', how='outer')
        cross =  cross[(cross[date_col] >= pd.to_datetime(cross['start_date'])) & (cross[date_col] <= pd.to_datetime(cross['end_date']))].copy().drop(['start_date', 'end_date', 'key'], axis=1)
        
        # add in known information
        output_df = cross.merge(df, on= [date_col, series_id], how= 'left')
    
    else:
        all_dates = pd.DataFrame(pd.date_range(start= df[date_col].min(), end= df[date_col].max(), freq= freq), columns = [date_col])
        # add in known information
        output_df = all_dates.merge(df, on= [date_col, series_id], how= 'left')
        
    # fill missing targets values
    if type(target_fill) == int:
        output_df[target].fillna(value= target_fill, inplace= True)
    else:
        output_df[target].fillna(method= target_fill, inplace= True)


    # fill static_column values
    if static_columns != None:
    
        if series_id != None:
            static_values = df.groupby(series_id)[static_columns].agg(lambda x: x.value_counts().index[0]).to_dict()
            for c in static_columns:
                output_df[c] = output_df[series_id].map(static_values[c])
        else:
            output_df[static_columns] = output_df[static_columns].fillna(method= 'bfill').fillna(method= 'ffill')

    return output_df


def fill_nans(df: pd.core.frame.DataFrame,
             ts_settings: dict,
             numeric_columns: typing.Optional[list]= None,
             numeric_method: typing.Optional[dict]= 'rolling_median',
             roll_length: typing.Optional[int]= 7,
             cat_columns: typing.Optional[list]= None,
             cat_method: typing.Optional[dict]= 'most recent'
             ) -> pd.core.frame.DataFrame:
    '''
    Fill missing values in defined columns.
    
    Inputs:
    -------
    df: Pandas dataframe
    ts_settings: dictionary of settings
    numeric_columns: names of numeric-type columns to fill
    numeric_methods: method to be used to fill. Accepts any methods that can be used by 'create_rolling()' function (mean, median, sum) or dict with a method for each column
    roll_length: number of time-steps in past to use for rolling statistic
    cat_columns: names of categorical/text columns to fill
    cat_method: method to used to fill categorical. Accepts 'most recent' or 'most common' or dict with a method for each column
    
    Outputs:
    --------
    df: Pandas dataframe    
    '''
    
    assert(numeric_method in ['mean', 'median', 'max', 'rolling_mean', 'rolling_median', 'rolling_max']), "numeric_method parameter accepts 'mean', 'median', 'max', 'rolling_mean', 'rolling_median', or 'rolling_max'"
    assert(cat_method in ['most recent', 'most common']), "cat_method parameter accepts 'most recent' or 'most common'"
    
    # define convenience variables
    series_id = ts_settings['series_id']
    
    # fill numeric values
    if 'rolling' in numeric_method:
        roll_method = numeric_method.split("_")[1]
        
        # create rolling dataset and index-match missing values
        for n in numeric_columns:
            tmp = df.copy()
            n_idx = df[df[n].isna()].index
            tmp_roll = create_rolling(df= tmp,
                                       column= n,
                                       roll_lengths= [roll_length],
                                       method= [roll_method])
#             tmp_roll = tmp_roll.loc[n_idx,f'{n}_rolling_{roll_length}_{roll_method}'].copy()
            df.loc[n_idx, n] = tmp_roll.loc[n_idx, f'{n}_rolling_{roll_length}_{roll_method}']
    else:
        # fill column nans with global values
        df[numeric_columns] = df[numeric_columns].fillna(method= numeric_method)
        
    # fill categorical columns
    if cat_method == 'most recent':
        df[cat_columns] = df[cat_columns].fillna(method= 'ffill')
        
    elif cat_method == 'most common' and series_id != None:
        for c in cat_columns:
            mode_dict = df.groupby(series_id)[c].agg(lambda x:x.value_counts().index[0]).to_dict() # get the most common value for each series
            c_idx = df[df[c].isna()].index.values
            df.loc[c_idx, c] = df.loc[c_idx, series_id].map(mode_dict)
    
    elif cat_method == 'most common' and series_id == None:
        for c in cat_columns:
            mode = df[c].value_counts().index[0]
            c_idx = df[df[c].isna()].index.values
            df.loc[c_idx, c] = mode
   
    else:
        print(f'There is a problem with categorical column: {c}')
        
    return df

# imports at bottom for circular logic workaround
from .calendar_preparation import *