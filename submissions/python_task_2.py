import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here


    # Create a list of unique IDs
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))

    # Create a NumPy array for the distance matrix
    num_ids = len(unique_ids)
    distance_matrix = np.zeros((num_ids, num_ids))

    # Fill in the distance matrix based on known routes
    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        start_idx, end_idx = unique_ids.index(start), unique_ids.index(end)
        distance_matrix[start_idx, end_idx] = distance
        distance_matrix[end_idx, start_idx] = distance  # Ensure matrix is symmetric

    # Calculate cumulative distances
    for k in range(num_ids):
        for i in range(num_ids):
            for j in range(num_ids):
                if distance_matrix[i, k] != 0 and distance_matrix[k, j] != 0:
                    if distance_matrix[i, j] == 0 or distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[
                        k, j]:
                        distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    # Convert the NumPy array to a DataFrame
    np.fill_diagonal(distance_matrix, 0)
    result_matrix = pd.DataFrame(distance_matrix, index=unique_ids, columns=unique_ids)

    return result_matrix

    #return df


def unroll_distance_matrix(distance_matrix)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """

    # Write your logic here
    unique_ids = distance_matrix.index
    # Existing DataFrame
    existing_df = pd.DataFrame({
        'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420,
                     1001422, 1001424, 1001426, 1001428,
                     1001430, 1001432, 1001434, 1001436, 1001436, 1001438, 1001438, 1001440, 1001442, 1001488, 1004356,
                     1004354, 1004355, 1001444, 1001446, 1001448, 1001450,
                     1001452, 1001454, 1001456, 1001458, 1001460, 1001460, 1001461, 1001462, 1001464, 1001466, 1001468,
                     1001470],

        'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420, 1001422,
                   1001424, 1001426, 1001428, 1001430, 1001432, 1001434,
                   1001436, 1001438, 1001437, 1001437, 1001440, 1001442, 1001488, 1004356, 1004354, 1004355, 1001444,
                   1001446, 1001448, 1001450, 1001452, 1001454, 1001456,
                   1001458, 1001460, 1001461, 1001462, 1001462, 1001464, 1001466, 1001468, 1001470, 1001472],
        'distance': [9.7, 20.2, 16.0, 21.7, 11.1, 15.6, 18.2, 13.2, 13.6, 12.9,
                     9.6, 11.4, 18.6, 15.8, 8.6, 9.0, 7.9, 4.0, 9.0, 5.0,
                     4.0, 10.0, 3.9, 4.5, 4.0, 2.0, 2.0, 0.7, 6.6, 9.6,
                     15.7, 9.9, 11.3, 13.6, 8.9, 5.1, 12.8, 17.9, 5.1,
                     26.7, 8.5, 10.7, 10.6, 16.0]
    })

    # Append the new rows to the existing DataFrame

    # Create empty arrays to store unrolled distances
    id_starts = np.array([])
    id_ends = np.array([])
    distances = np.array([])

    for i in range(len(unique_ids)):
        for j in range(len(unique_ids)):
            # Skip combinations where id_start is equal to id_end or distance is 0
            if i != j and distance_matrix.iloc[i, j] != 0:
                # Check if the combination already exists in existing_df
                if not ((existing_df['id_start'] == unique_ids[i]) & (existing_df['id_end'] == unique_ids[j])).any():
                    id_starts = np.append(id_starts, unique_ids[i])
                    id_ends = np.append(id_ends, unique_ids[j])
                    distances = np.append(distances, distance_matrix.iloc[i, j])

    # Create a DataFrame from the numpy arrays
    unrolled_df = pd.DataFrame({'id_start': id_starts, 'id_end': id_ends, 'distance': distances})

    return unrolled_df

    #return df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_rows = df[df['id_start'] == reference_value]

    # Check if any rows match the reference value
    if reference_rows.empty:
        print(f"No matching rows found for the reference value {reference_value}.")
        return []

    # Calculate the average distance for the reference value
    reference_average = reference_rows['distance'].mean()

    # Calculate the threshold values (10% above and below the average)
    lower_threshold = reference_average - (0.1 * reference_average)
    upper_threshold = reference_average + (0.1 * reference_average)

    # Filter rows within the 10% threshold
    filtered_rows = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]

    # Extract unique values from the id_start column, convert to integers, and sort them
    result_ids = sorted(filtered_rows['id_start'].astype(int).unique())

    return result_ids

    #return df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

    #return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
