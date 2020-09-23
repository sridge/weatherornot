import glob

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt 

from tqdm import tqdm



def load_csv(path):
    
    '''minor cleaning and concatenates csv files'''
    
    all_files = glob.glob(path)

    li = []

    for filename in tqdm(all_files):
        df = pd.read_csv(filename, index_col=None, header=0)
        df['Speed'] = df['Speed'][df['Speed'] !=  'Bronx'].astype(float)
        df['DataAsOf'] = df['DataAsOf'][df['Speed'] !=  'Bronx'].astype(str)
        df = df[['Speed','DataAsOf','Id']]
        li.append(df)
        

    return pd.concat(li, axis=0, ignore_index=True)

# def clean_df(df):
    
#     return print('nothing')

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

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                   train_df=train_df, val_df=val_df, test_df=test_df,
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

