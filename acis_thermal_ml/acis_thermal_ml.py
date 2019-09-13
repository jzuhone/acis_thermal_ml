from acis_thermal_ml import utils as dmf
from .utils import make_phase, data_map, \
    acis_states
import pandas as pd
import numpy as np
from Ska.engarchive import fetch
from Chandra.Time import DateTime, secs2date
from Chandra.cmd_states import fetch_states, interpolate_states
from keras import callbacks
import pickle
np.random.seed(0)

inputs = ["pitch", "roll", "sim_z", "ccd_count",
          "fep_count", "clocking"]

def create_model(n_neurons, timesteps, data_dim, p_W, p_U, weight_decay, p_dense):
    from keras.models import Sequential
    from keras.layers.recurrent import LSTM
    from keras.layers.core import Dense, Dropout
    from keras.regularizers import l2
    # an LSTM model takes as a 3d tensor so we need to reshape our data to fit that
    # the shape is a 3d tensor with dimensions (batch_size, time_steps, features)
    # dropout is used for uncertainty calculations
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(timesteps, data_dim),
                   kernel_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay),
                   bias_regularizer=l2(weight_decay), dropout=p_W, recurrent_dropout=p_U))
    model.add(Dropout(p_dense))
    model.add(Dense(1, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


class ACISThermalML(object):
    def __init__(self, msid, inputs, frames=8, percentage=0.2, epochs=100, 
                 n_neurons=None):
        self.msid = msid
        self.inputs = inputs
        self.frames = frames
        self.percentage = percentage
        self.epochs = epochs
        self.n_neurons = n_neurons
        self.cols = ['msid_times', 'msid_vals', 'phase'] + self.inputs
        self.pos = self.cols[2:]
        self.n_features = len(self.pos) + 1
        self.model = None

    def get_cmd_states(self, datestart, datestop, times):
        tstart = DateTime(datestart).secs - 50.0*328.0
        tstop = DateTime(datestop).secs + 50.0*328.0
        states = fetch_states(tstart, tstop, dbi="hdf5")
        cmd_states = interpolate_states(states, times)
        return cmd_states

    def get_fitting_data(self, start, stop, inputs):
        msids = [self.msid] + [data_map[input] for input in inputs 
                               if input not in acis_states]
        data = fetch.MSIDset(msids, start, stop, stat='5min',
                             filter_bad=True)
        states = self.get_cmd_states(data.datestart, data.datestop, 
                                     data[self.msid].times)
        combined_dict = {'msid_times': data[self.msid].times,
                         'msid_vals': data[self.msid].vals,
                         'phase': make_phase(data[self.msid].times)}
        for input in inputs:
            if input in data_map:
                combined_dict[input] = data[data_map[input]].vals
            elif input in states:
                combined_dict[input] = states[input]
        return combined_dict

    def get_prediction_data(self, times, att_data, cmd_states):
        states = interpolate_states(cmd_states, times)
        combined_dict = {'msid_times': times,
                         'msid_vals': data[self.msid].vals,
                         'phase': make_phase(times)}
        combined_dict.update(att_data)
        for key in states:
            if key in self.inputs:
                combined_dict[key] = states[key]
        return combined_dict

    def fit_model(self):
        pass

    def plot_stats(self, history):
        import matplotlib.pyplot as plt
        plt.rc("font", size=18)
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='train')
        ax.plot(history.history['val_loss'], label='validation')
        ax.set_title('loss {}'.format(train_year))
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss over epochs')
        ax.legend()
        ax.set_yscale('log')
        return fig

    def make_predict_data(self):
        pass

    def make_week_predict(self, times, att_data, cmd_states, scaler_file, 
                          weights_file):
        scaler_all = pickle.load(open(scaler_file, "rb"))
        predict_inputs = self.get_prediction_data(times, att_data, cmd_states)
        predict_clean_set, predict_times = dmf.clean_data(predict_inputs, self.cols, self.pos)
        predict_scaled_set = pd.DataFrame(scaler_all.transform(predict_clean_set),
                                          columns=predict_clean_set.columns)
        shaped_test, begin_int_test = dmf.shaping_data(predict_scaled_set, self.pos, self.frames)
        predict_times = predict_times[begin_int_test:]
        predict_data = self.predict(test_X)
        return predict_times, predict_data

    def predict(self, test_X):
        predictions = self.model.predict(test_X)
        scaled_predictions = scaler_msid.inverse_transform(predictions)
        scaled_predictions = np.squeeze(scaled_predictions) - 273.15
        return scaled_predictions

    def write_prediction(self, filename, predict_time, predict_data):
        from astropy.table import Table
        temp_table = Table([predict_time, secs2date(predict_time), predict_data],
                           names=['time', 'date', self.msid.lower()],
                           copy=False)
        temp_table['time'].format = '%.2f'
        temp_table[self.msid.lower()].format = '%.2f'
        temp_table.write(filename, format='ascii', delimiter='\t', overwrite=True)


msid_value = '1DPAMZT'

###### STEP 2: SINGLE OR COMPARE MULTIPLE YEARS ######
train_year = 2018
test_year = 2019

file_format = '{}_{}_msid_data.pkl'
test_msid_file = file_format.format(msid_value, test_year)

###### STEP 3: PULL + COLLATE DATA ######
# 'collate' means calculate and assign a pitch, roll, yaw value to each time period
# this is done using sad_common_functions can look there for more ingo
#########################################

#if we don't have data for training years we pull, collate and save it 
for i, pull_year in enumerate([train_year, test_year]):
    if i == 0:
        start = '{}:180:00:00:00'.format(pull_year-2)
        stop = '{}:366:24:60:60'.format(pull_year)
    else:
        start = '{}:001:00:00:00'.format(pull_year)
        stop = '{}:366:24:60:60'.format(pull_year)
    data = fetch.MSIDset([msid_value, 'DP_PITCH', 'DP_ROLL', "SIM_Z"], start, stop, stat='5min', 
                         filter_bad=True)
    data.interpolate(times=data["DP_PITCH"].times)
    combined_dict = get_all_data(msid_value, data)
    save_file = file_format.format(msid_value, pull_year)
    f = open(save_file, "wb")
    pickle.dump(combined_dict, f)
    f.close()


test_msid_file = file_format.format(msid_value, test_year)
train_msid_file = file_format.format(msid_value, train_year)
train_dict = pd.read_pickle(train_msid_file)
test_dict = pd.read_pickle(test_msid_file)
train_set = pd.DataFrame(train_dict)
test_set = pd.DataFrame(test_dict)


# variables we will use 
#cols = ['msid_times', 'msid_vals', 'pitch', 'roll', 'ccd_count', 
#        'fep_count', 'clocking', 'sim_z', 'phase']
#pos = cols[2:]
#n_features = len(pos)+1
#frames = 8
#percentage = 0.2


#first we clean our training set and rows with null values
train_clean_set, train_time = dmf.clean_data(train_set, cols, pos)
raw_msid_val = train_clean_set.drop(pos, axis=1)
#scale training data and return the scalers so we can use them to unscale
scaler_all, scaler_msid, scaled_train = dmf.scale_training(train_clean_set, raw_msid_val)
# shape data using our common function 
# this common function returns data in roll(t-8), pitch(t-8), val(t-8)... val(t) 
# drops pitch(t) and roll(t)
shaped_train, begin_int = dmf.shaping_data(scaled_train, pos, frames)
# take the shaped data and return an averaged version 
variables = pos + ['msid_vals']
#
# split training data that has already been shaped into training and validation
shaped_train_full, train_time_full, shaped_val_full, val_time_full = \
    dmf.split_shaped_data(shaped_train, train_time, percentage, begin_int)

#
# now clean, scale and shape the test data
#
test_clean_set, test_time = dmf.clean_data(test_set, cols, pos)
# using the same scaler as the train set to scale the test set
scaled_test = scaler_all.transform(test_clean_set)
scaled_test = pd.DataFrame(scaled_test, columns=test_clean_set.columns)
shaped_test, begin_int_test = dmf.shaping_data(scaled_test, pos, frames)
test_time = test_time[begin_int_test:]


# take shaped test, train and validation data and return inputs/outputs
# split io reshapes data to be 3D tensor shape (samples, timesteps, features)
# essentially whatev_X [[roll(t-8),pitch(t-8),val(t-8)]...[roll(t-1),pitch(t-1),val(t-1)]]
# whatev_y is [val(t)]
train_X, train_y = dmf.split_io(shaped_train_full, frames, n_features)
validate_X, validate_y = dmf.split_io(shaped_val_full, frames, n_features)
test_X, test_y = dmf.split_io(shaped_test, frames, n_features)

#send p_dense to 0 to ignore dropout
p_W, p_U, p_dense, weight_decay, batch_size, n_neurons = 0.01, 0.001, 0.0, 1e-6, 512, 16
epochs = 100
#create model
timesteps, data_dim = train_X.shape[1] , train_X.shape[2]
checkpoint_path = 'weights_best_{}_yr{}_pW_{}_pU_{}_pdense{}'.format(msid_value,train_year, p_W, p_U, p_dense)
model = create_model(n_neurons, timesteps, data_dim, p_W, p_U, weight_decay, p_dense)

# checkpoint path to save weights
checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                         verbose=0, save_best_only=True,
                                         save_weights_only=True, mode='min')

history = model.fit(train_X, train_y, validation_data=(validate_X, validate_y),
                    batch_size=batch_size, epochs=epochs, callbacks=[checkpointer], 
                    shuffle=False, verbose=0)

