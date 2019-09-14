import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Chandra.Time import DateTime

time2000 = DateTime('2000:001:00:00:00').secs
time2010 = DateTime('2010:001:00:00:00').secs
secs_per_year = (time2010 - time2000) / 10.0

data_map = {"pitch": "DP_PITCH",
            "roll": "DP_ROLL",
            "sim_z": "SIM_Z"}

pwr_states = ["ccd_count", "fep_count", "clocking"]


def make_phase(times):
    # This is the contribution from the variation in the solar
    # heating due to the Earth's elliptical orbit
    t_year = (times - time2000) / secs_per_year
    return np.cos(2.0*np.pi*t_year)


def scale_training(train_set, raw_msid_val):
    # normalize data (to be between 0 and 1) and then reshape
    scaler_full = MinMaxScaler()
    scaled = scaler_full.fit_transform(train_set)
    # creating a seperate scaling for msid_vals cause honestly it's ruining my life
    scaler_msid = MinMaxScaler()
    scaled_msid_val = scaler_msid.fit_transform(raw_msid_val)
    # scaled_df = pd.DataFrame(scaled, columns = raw.columns).iloc[::spacing_int,:]
    scaled_train = pd.DataFrame(scaled, columns=train_set.columns)
    return scaler_full, scaler_msid, scaled_train


def clean_data(data, cols, pos):
    #cleaning that needs to occur for all sets(train, validation, test)
    #first take out any null values with a mask
    #return the time data and the msid data (the time data is for plotting)
    subset = data[cols]
    mask = [all(tup) for tup in zip(*[~np.isnan(subset['{}'.format(i)]) for i in pos])]
    masked = subset[mask]
    #seperate out the time data
    msid_times = masked['msid_times']
    raw_set = masked.drop(['msid_times'], axis=1)
    return raw_set, msid_times

#############################################
#reshape the data into time arrays so it looks like [val(t-n), pitch(t-n), roll(t-n),..., val(t), pitch(t)]
#we can choose and play around with n
#https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
#assumes order by time
def reshape_to_multi_time(data, frames=1):
    col_names = data.columns.values
    cols, names = list(), list()
    #input sequence (t-n, ... t-1)
    for i in range(frames, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (name,  i)) for name in col_names]
    #forecast sequence (t, t+1, ... t+n)
    for i in range(0,1):
        cols.append(data.shift(-i))
        if i == 0: 
            names += [('%s(t)' % (name)) for name in col_names]
        else:
            names += [('%s(t+%d)' % (name, i)) for name in col_names]
    #put it all together
    agg = pd.concat(cols, axis = 1)
    agg.columns = names
    # drops rows with NaN values
    agg_full = agg.dropna()
    return agg_full


def shaping_data(scaled_df, pos, frames):
    # reshapes data for an lstm
    dat = reshape_to_multi_time(scaled_df, frames=frames)
    # we drop these values since we're not predicting them
    drop_pos = [id + "(t)" for id in pos]
    shaped = dat.drop(drop_pos, axis=1).values
    return shaped, dat.first_valid_index()


def split_shaped_data(shaped_data, time, percentage, offset):
    chunk = int(shaped_data.shape[0]*(1-percentage))
    left_chunk, left_time = shaped_data[:chunk], time[:chunk]
    right_chunk, right_time = shaped_data[chunk:], time[chunk+offset:]
    return left_chunk, left_time, right_chunk, right_time


def split_io(interval, frames, n_features):
    # split into inputs and outputs
    interval_X, interval_y = interval[:,:-1], interval[:,-1]
    # reshape input to be 3D tensor with shape (samples, timesteps, features)
    interval_X = interval_X.reshape((interval_X.shape[0], frames, n_features))
    return interval_X, interval_y
