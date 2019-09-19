from .model_run import ModelRun
from .utils import data_map, \
    pwr_states, clean_data, \
    scale_training, shaping_data, \
    split_io, split_shaped_data
import pandas as pd
import numpy as np
import Ska.engarchive.fetch_sci as fetch
from Chandra.Time import DateTime, secs2date, date2secs
from Chandra.cmd_states import fetch_states, interpolate_states
import pickle
np.random.seed(0)
import Ska.Numpy

class ACISThermalML(object):
    def __init__(self, msid, inputs, frames=8, percentage=0.2, epochs=100, 
                 n_neurons=16, batch_size=512, p_W=0.01, p_U=0.001, p_dense=0.0,
                 weight_decay=1.0e-6):
        self.msid = msid.upper()
        self.inputs = inputs
        self.frames = frames
        self.percentage = percentage
        self.epochs = epochs
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.p_W = p_W
        self.p_U = p_U
        self.p_dense = p_dense
        self.weight_decay = weight_decay
        self.cols = ['msid_times', 'msid_vals', 'd_sun'] + self.inputs
        self.pos = self.cols[2:]
        self.n_features = len(self.pos) + 1
        self.model = None
        self.scaler_all = None
        self.scaler_msid = None
        self.scaler_all_file = "{}_scaler_all.pkl".format(self.msid.lower())
        self.scaler_msid_file = "{}_scaler_msid.pkl".format(self.msid.lower())
        self.checkpoint_path = 'weights_best_{}_pW_{}_pU_{}_pdense{}'.format(self.msid, p_W, p_U, p_dense)
        self.model_path = 'model_{}.hdf5'.format(self.msid)

    @staticmethod
    def _eng_match_times(start, stop, dt):
        """Return an array of times between ``start`` and ``stop`` at ``dt``
        sec intervals.  The times are roughly aligned (within 1 sec) to the
        timestamps in the '5min' (328 sec) Ska eng archive data.
        """
        time0 = 410270764.0
        i0 = int((DateTime(start).secs - time0) / dt) + 1
        i1 = int((DateTime(stop).secs - time0) / dt)
        return time0 + np.arange(i0, i1) * dt

    def create_model(self, n_neurons, timesteps, data_dim, p_W, p_U, weight_decay, p_dense):
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
        self.model = model

    def get_cmd_states(self, datestart, datestop, times):
        tstart = DateTime(datestart).secs - 50.0*328.0
        tstop = DateTime(datestop).secs + 50.0*328.0
        states = fetch_states(tstart, tstop, dbi="hdf5")
        cmd_states = interpolate_states(states, times)
        return cmd_states

    def get_fitting_data(self, start, stop):
        tstart = date2secs(start)
        tstop = date2secs(stop)
        times = self._eng_match_times(tstart, tstop, 328.0)
        msids = [self.msid] + [data_map[input] for input in self.inputs
                               if input not in pwr_states]
        msids += ['solarephem0_{}'.format(ax) for ax in "xyz"]
        data = fetch.MSIDset(msids, start, stop, stat='5min',
                             filter_bad=True)
        data.interpolate(times=times)
        msid_vals = Ska.Numpy.smooth(data[self.msid].vals,10)
        sun_eci = np.array([data['solarephem0_x'].vals,
                            data['solarephem0_y'].vals,
                            data['solarephem0_z'].vals])
        d_sun = np.sqrt((sun_eci**2).sum(axis=0))
        states = self.get_cmd_states(data.datestart, data.datestop, times)
        combined_dict = {'msid_times': times,
                         'msid_vals': msid_vals,
                         'd_sun': d_sun}
        for input in self.inputs:
            if input in data_map:
                combined_dict[input] = data[data_map[input]].vals
            elif input in states.dtype.names:
                combined_dict[input] = states[input]
        return pd.DataFrame(combined_dict)

    def get_prediction_data(self, tstart, tstop, T_init, att_data, cmd_states):
        times = self._eng_match_times(tstart, tstop, 328.0)
        states = interpolate_states(cmd_states, times)
        if T_init is None:
            msid_vals = fetch.MSID(self.msid, tstart, tstop, stat='5min',
                                   filter_bad=True)
            msid_vals.interpolate(times=times)
            msid_vals = msid_vals.vals
        else:
            msid_vals = T_init*np.ones_like(times)
        combined_dict = {'msid_times': times,
                         'msid_vals': msid_vals}
        att_times = att_data.pop("times")
        d_sun = Ska.Numpy.interpolate(att_data.pop("d_sun"), att_times,
                                      times, method="linear")
        combined_dict['d_sun'] = d_sun
        for input in self.inputs:
            if input in att_data:
                combined_dict[input] = Ska.Numpy.interpolate(att_data[input], att_times,
                                                             times, method="linear")
            elif input == "sim_z":
                combined_dict["sim_z"] = -0.0025143153015598743*states["simpos"]
            elif input in pwr_states:
                combined_dict[input] = states[input]
        return pd.DataFrame(combined_dict)

    def train_and_fit_model(self, start, stop):
        train_set = self.get_fitting_data(start, stop)
        train_clean_set, train_time = clean_data(train_set, self.cols, self.pos)
        raw_msid_val = train_clean_set.drop(self.pos, axis=1)
        scaler_all, scaler_msid, scaled_train = \
            scale_training(train_clean_set, raw_msid_val)
        self.scaler_all = scaler_all
        with open(self.scaler_all_file, 'wb') as f:
            pickle.dump(self.scaler_all, f)
        self.scaler_msid = scaler_msid
        with open(self.scaler_msid_file, 'wb') as f:
            pickle.dump(self.scaler_msid, f)
        shaped_train, begin_int = shaping_data(scaled_train, self.pos, self.frames)
        shaped_train_full, train_time_full, shaped_val_full, val_time_full = \
            split_shaped_data(shaped_train, train_time, self.percentage, begin_int)
        train_x, train_y = split_io(shaped_train_full, self.frames,
                                    self.n_features)
        validation_data = split_io(shaped_val_full, self.frames,
                                   self.n_features)

        # create model
        timesteps, data_dim = train_x.shape[1], train_x.shape[2]
        self.create_model(self.n_neurons, timesteps, data_dim, self.p_W, self.p_U, 
                          self.weight_decay, self.p_dense)

        history = self.model.fit(train_x, train_y, validation_data=validation_data,
                                 batch_size=self.batch_size, epochs=self.epochs,
                                 shuffle=False, verbose=0)

        self.model.save(self.model_path)

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
        from keras.models import load_model
        if self.model is None:
            self.model = load_model(self.model_path)
        if self.scaler_all is None:
            self.scaler_all = pickle.load(open(self.scaler_all_file, "rb"))
        if self.scaler_msid is None:
            self.scaler_msid = pickle.load(open(self.scaler_msid_file, "rb"))
        predict_clean_set, predict_times = clean_data(predict_inputs, self.cols, self.pos)
        predict_scaled_set = pd.DataFrame(self.scaler_all.transform(predict_clean_set),
                                          columns=predict_clean_set.columns)
        shaped_predict, begin_int_predict = shaping_data(predict_scaled_set,
                                                         self.pos, self.frames)
        predict_times = predict_times[begin_int_predict:]
        predict_x, _ = split_io(shaped_predict, self.frames, self.n_features)
        #predictions = self.model.predict(predict_x)
        #predict_data = np.squeeze(self.scaler_msid.inverse_transform(predictions))
        #self.write_prediction("temperatures.dat", predict_times, predict_data)
        #return predict_times, predict_data
        T_out = np.zeros_like(predict_times)
        T_out[0] = predict_clean_set["msid_vals"][begin_int_predict]
        for i in range(len(predict_x)-1):
            predictions = self.model.predict(np.array([predict_x[i,:,:]]))
            predict_x[i+1,:,0] = predictions
            predict_data = np.squeeze(self.scaler_msid.inverse_transform(predictions))
            T_out[i+1] = predict_data
        return predict_times, T_out

    def test_model(self, start, stop): 
        predict_inputs = self.get_fitting_data(start, stop)
        return self._predict_model(predict_inputs)

    def predict_model(self, tstart, tstop, T_init, att_data, cmd_states):
        predict_inputs = self.get_prediction_data(tstart, tstop, T_init, 
                                                  att_data, cmd_states)
        predict_times, predict_data = self._predict_model(predict_inputs)
        return ModelRun(self.frames, self.msid, np.array(predict_times), predict_data, predict_inputs)

    def write_prediction(self, filename, predict_times, predict_data):
        from astropy.table import Table
        temp_table = Table([predict_times, secs2date(predict_times), predict_data],
                           names=['time', 'date', self.msid.lower()],
                           copy=False)
        temp_table['time'].format = '%.2f'
        temp_table[self.msid.lower()].format = '%.2f'
        temp_table.write(filename, format='ascii', delimiter='\t', overwrite=True)




