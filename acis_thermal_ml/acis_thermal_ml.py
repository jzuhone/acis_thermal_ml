from acis_thermal_ml import utils as dmf
from .utils import make_phase, data_map, \
    acis_states
import pandas as pd
import numpy as np
import Ska.engarchive.fetch_sci as fetch
from Chandra.Time import DateTime, secs2date
from Chandra.cmd_states import fetch_states, interpolate_states
from keras import callbacks
import pickle
np.random.seed(0)


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
                 n_neurons=16, batch_size=512):
        self.msid = msid.upper()
        self.inputs = inputs
        self.frames = frames
        self.percentage = percentage
        self.epochs = epochs
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.cols = ['msid_times', 'msid_vals', 'phase'] + self.inputs
        self.pos = self.cols[2:]
        self.n_features = len(self.pos) + 1
        self.model = None
        self.scaler_all = None
        self.scaler_msid = None
        self.scaler_all_file = "{}_scaler_all.pkl".format(self.msid.lower())
        self.scaler_msid_file = "{}_scaler_msid.pkl".format(self.msid.lower())

    def get_cmd_states(self, datestart, datestop, times):
        tstart = DateTime(datestart).secs - 50.0*328.0
        tstop = DateTime(datestop).secs + 50.0*328.0
        states = fetch_states(tstart, tstop, dbi="hdf5")
        cmd_states = interpolate_states(states, times)
        return cmd_states

    def get_fitting_data(self, start, stop):
        msids = [self.msid] + [data_map[input] for input in self.inputs 
                               if input not in acis_states]
        data = fetch.MSIDset(msids, start, stop, stat='5min',
                             filter_bad=True)
        data.interpolate(times=data["DP_PITCH"].times)
        states = self.get_cmd_states(data.datestart, data.datestop, 
                                     data[self.msid].times)
        combined_dict = {'msid_times': data[self.msid].times,
                         'msid_vals': data[self.msid].vals,
                         'phase': make_phase(data[self.msid].times)}
        for input in self.inputs:
            if input in data_map:
                combined_dict[input] = data[data_map[input]].vals
            elif input in states.dtype.names:
                combined_dict[input] = states[input]
        return pd.DataFrame(combined_dict)

    def get_prediction_data(self, times, T_init, att_data, cmd_states):
        states = interpolate_states(cmd_states, times)
        combined_dict = {'msid_times': times,
                         'msid_vals': T_init*np.ones_like(times),
                         'phase': make_phase(times)}
        combined_dict.update(att_data)
        for key in states:
            if key in self.inputs:
                combined_dict[key] = states[key]
        return combined_dict

    def train_and_fit_model(self, start, stop):
        train_set = self.get_fitting_data(start, stop)
        train_clean_set, train_time = dmf.clean_data(train_set, self.cols, self.pos)
        raw_msid_val = train_clean_set.drop(self.pos, axis=1)
        scaler_all, scaler_msid, scaled_train = \
            dmf.scale_training(train_clean_set, raw_msid_val)
        self.scaler_all = scaler_all
        with open(self.scaler_all_file, 'wb') as f:
            pickle.dump(self.scaler_all, f)
        self.scaler_msid = scaler_msid
        with open(self.scaler_msid_file, 'wb') as f:
            pickle.dump(self.scaler_msid, f)
        shaped_train, begin_int = dmf.shaping_data(scaled_train, self.pos, self.frames)
        shaped_train_full, train_time_full, shaped_val_full, val_time_full = \
            dmf.split_shaped_data(shaped_train, train_time, self.percentage, begin_int)
        train_x, train_y = dmf.split_io(shaped_train_full, self.frames,
                                        self.n_features)
        validate_x, validate_y = dmf.split_io(shaped_val_full, self.frames,
                                              self.n_features)

        p_W, p_U, p_dense, weight_decay = 0.01, 0.001, 0.0, 1e-6
        # create model
        timesteps, data_dim = train_x.shape[1], train_x.shape[2]
        checkpoint_path = 'weights_best_{}_pW_{}_pU_{}_pdense{}'.format(self.msid, p_W, p_U, p_dense)
        self.model = create_model(self.n_neurons, timesteps, data_dim, p_W, p_U, 
                                  weight_decay, p_dense)

        # checkpoint path to save weights
        checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 verbose=0, save_best_only=True,
                                                 save_weights_only=True, mode='min')

        history = self.model.fit(train_x, train_y, validation_data=(validate_x, validate_y),
                                 batch_size=self.batch_size, epochs=self.epochs,
                                 callbacks=[checkpointer], shuffle=False, verbose=0)

        self.plot_stats(history)

    def plot_stats(self, history):
        import matplotlib.pyplot as plt
        plt.rc("font", size=18)
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='train')
        ax.plot(history.history['val_loss'], label='validation')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss over epochs')
        ax.legend()
        ax.set_yscale('log')
        fig.savefig("stats.png", bbox_inches='tight')
        return fig

    def _predict_model(self, predict_inputs):
        if self.scaler_all is None:
            self.scaler_all = pickle.load(open(self.scaler_all_file, "rb"))
        if self.scaler_msid is None:
            self.scaler_msid = pickle.load(open(self.scaler_msid_file, "rb"))
        predict_clean_set, predict_times = dmf.clean_data(predict_inputs, self.cols, self.pos)
        predict_scaled_set = pd.DataFrame(self.scaler_all.transform(predict_clean_set),
                                          columns=predict_clean_set.columns)
        shaped_predict, begin_int_predict = dmf.shaping_data(predict_scaled_set,
                                                             self.pos, self.frames)
        predict_times = predict_times[begin_int_predict:]
        predict_x, _ = dmf.split_io(shaped_predict, self.frames, self.n_features)
        predictions = self.model.predict(predict_x)
        predict_data = np.squeeze(self.scaler_msid.inverse_transform(predictions))
        self.write_prediction("temperatures.dat", predict_times, predict_data)
        return predict_times, predict_data

    def test_model(self, start, stop):
        predict_inputs = self.get_fitting_data(start, stop)
        return self._predict_model(predict_inputs)

    def predict_model(self, times, T_init, att_data, cmd_states):
        predict_inputs = self.get_prediction_data(times, T_init, att_data, cmd_states)
        return self._predict_model(predict_inputs)

    def write_prediction(self, filename, predict_times, predict_data):
        from astropy.table import Table
        temp_table = Table([predict_times, secs2date(predict_times), predict_data],
                           names=['time', 'date', self.msid.lower()],
                           copy=False)
        temp_table['time'].format = '%.2f'
        temp_table[self.msid.lower()].format = '%.2f'
        temp_table.write(filename, format='ascii', delimiter='\t', overwrite=True)




