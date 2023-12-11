import pandas as pd
import numpy as np


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    car_matrix = df.pivot(index="id_1", columns="id_2", values="car").fillna(0)

    # setting diagonals to 0
    for i in car_matrix.columns:
        car_matrix.at[i, i] = 0
    return car_matrix




def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    #df = pd.read_csv(path)

    conditions = [
        (df["car"] <= 15), (df["car"] > 15) & (df["car"] <= 25), (df["car"] > 25)

    ]
    labels = ["low", "medium", "high"]
    df["car_type"] = np.select(conditions, labels, default=None)

    count = df["car_type"].value_counts().to_dict()

    sorted_dict = dict(sorted(count.items()))

    return sorted_dict

    #return dict()


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    #df = pd.read_csv(path)
    bus_mean = np.mean(df["bus"])

    result = list(np.where(df["bus"] > 2 * bus_mean)[0])

    return result


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    #df = pd.read_csv(path)

    # target -->route column values
    result = sorted(df.loc[df["truck"] > 7, "route"].unique())
    return result

    #return list()


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    df2 = matrix.copy()
    multiplied_matrix = np.where(df2 > 20, df2 * 0.75, df2 * 1.25)
    multiplied_matrix = np.round(multiplied_matrix, 1)
    return pd.DataFrame(multiplied_matrix, columns=df2.columns, index=df2.index)


    #return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    days_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    # Check if there's a gap of 7 days between start and end days
    df['start_day_num'] = df['startDay'].map(days_mapping)
    df['end_day_num'] = df['endDay'].map(days_mapping)
    df['duration'] = np.where(df['end_day_num'] - df['start_day_num'] >= -1, 6, df['end_day_num'] - df['start_day_num'])
    full_24_hours = (pd.to_datetime(df['endTime'] + ' 00:00:01') - pd.to_datetime(df['startTime']) >= pd.to_timedelta(
        '0 days'))
    all_days_present = (df['duration'] == 6)
    result_series = full_24_hours & all_days_present
    result_series.index = pd.MultiIndex.from_frame(df[['id', 'id_2']])
    return result_series

    #return pd.Series()
