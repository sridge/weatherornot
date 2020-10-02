import datetime
import glob
import os

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf
from tqdm import tqdm

def load_csv(path):
    
    '''minor cleaning and concatenates csv files'''
    
    all_files = glob.glob(path)

    li = []

    for filename in tqdm(all_files):
        df = pd.read_csv(filename, index_col=None, header=0)
        df.columns = ['Id', 'Speed', 'TravelTime', 'Status', 'DataAsOf', 'linkId']
        if df['Speed'].dtype != np.float64:
            cond = ((df['Speed'].str.contains('Speed')) | (df['Speed'].str.contains('Bronx')))
            df[cond]=np.nan
        df['Speed'] = df['Speed'].astype(float)
        df['DataAsOf'] = df['DataAsOf'].astype(str)
        df = df[['Speed','DataAsOf','linkId']]
        li.append(df)


    return pd.concat(li, axis=0, ignore_index=True)

def plot_traffic_speed(df,start,end,figsize = (10,3),ticks='1D'):
    
    '''plots and resamples'''

    plt.figure(dpi=300,figsize=figsize)
    dates = pd.date_range(start=start,end=end,freq=ticks)

    mask = (df['DataAsOf'] > pd.to_datetime(start)) & (df['DataAsOf'] <= pd.to_datetime(end))

    df_filtered = df[mask]

    df_list = []


    for group in df_filtered.groupby('Id'):
        df_tem = group[1].set_index('DataAsOf')
        plt.plot(df_tem.index,df_tem['Speed_norm'],alpha=0.05,color='k')
        plt.xticks(ticks = dates,labels = dates,rotation=45,ha='right',size=5)

        df_i = df_tem.rename(columns = {'Speed_norm':df_tem['Id'][0]})[df_tem['Id'][0]]
        df_i = df_i.resample('5min').mean()

        df_list.append(df_i)

    df_rs = pd.concat(df_list, axis=1, ignore_index=False)
    df_rs.median(axis=1).plot(c='tab:orange')


#     for time in df_weather[df_weather['PRCP'] > 0.25].index.values:
#         plt.axvline(time,lw=1,c='0.8',ls='--')

#     for time in df_weather[df_weather['SNOW'] > 0.2].index.values:
#         plt.axvline(time,lw=1,c='r',ls='--')


    plt.xlim(pd.to_datetime(start),pd.to_datetime(end))
    plt.ylabel('Traffic Speed Anomaly (Z)')
    plt.xlabel('Time')
    
    return df_rs

def visualize_prediction(df,ticks,test_case,time,dense_no_weather,dense_weather,cond=0,width = 12):
    
    inds = test_case[(test_case['tp']>cond)].index
    ind_0 = np.random.randint(0,len(inds)-1)
    ind_0 = inds[ind_0]

    plt.figure(dpi=300,figsize=[10,3])
    plt.plot(time.loc[ind_0:ind_0+3*width],df.loc[ind_0:ind_0+3*width]['speed'].values,label='actual')

    df_prec = df.loc[ind_0:ind_0+width].copy()
    
    prediction_no_weather = dense_no_weather.predict(df_prec.drop(['tp','sf'],axis=1))
    prediction_weather = dense_weather.predict(df_prec)
    
    plt.plot(time.loc[ind_0+width:ind_0+2*width],prediction_no_weather,label='predicted (nwx)',c='0.5',lw=5,ls='-')
    plt.plot(time.loc[ind_0+width:ind_0+2*width],prediction_weather,label='predicted (wx)',c='k',lw=5,ls='-')

    plt.grid()
    start = time.loc[ind_0]
    end = time.loc[ind_0+3*width]
    dates = pd.date_range(start=start,end=end,freq=ticks)
    plt.xticks(ticks = dates,labels = dates,rotation=45,ha='right',size=10)
    plt.legend()

def train_test_val_split(df):

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]
    
    return (train_df,val_df,test_df), num_features

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                   train_df, val_df, test_df,
                   label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
              self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        
#         inds = np.where(~np.isnan(features).any(axis=2)[:,1])[0]
#         features = tf.convert_to_tensor(features.numpy()[inds,])

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns],
                              axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        

        return inputs, labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                        data=data,
                        targets=None,
                        sequence_length=self.total_window_size,
                        sequence_stride=1,
                        shuffle=True,
                        batch_size=32,)

        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def compile_and_fit(model, window, patience=1,MAX_EPOCHS = 20):
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history
