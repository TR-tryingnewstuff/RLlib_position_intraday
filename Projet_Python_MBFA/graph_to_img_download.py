from email.mime import image
from get_data_stock_trading_mbfa import *
import pandas
import pandas_ta
import numpy as np
import io
import numba
import multiprocessing

from matplotlib.lines import Line2D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image as im
from joblib import Parallel, delayed
mpl.rcParams['savefig.pad_inches'] = 0

WINDOW = 60
df = data_main(0)
print(df.head())

def download_image_as_array(df, i):

    df = df.copy()
    df = df.iloc[i:i+WINDOW]

    df = df*1
    # Creating Subplots
    fig = Figure(figsize=(5, 5))
    fig.tight_layout(pad=0)
    ax = fig.gca()


    # avoiding look forward bias 
    df.iloc[-8:, 9] = 0
    df.iloc[-8:, 10] = 0


    # Plotting Candles 
    color_index = df.apply(lambda x: 'red' if (x.cond_bull_ob == 1) else 'blue' if (x.cond_bear_ob == 1) else 'gray' if (x.open < x.close) else 'black', axis=1)
    date_index = np.array(df.index)

    bars = np.array(df.close)-np.array(df.open)
    wicks = np.array(df.high)-np.array(df.low)
    ax.bar(date_index, bars, width=0.8, bottom=df.open, color=color_index)
    ax.bar(date_index, wicks, width=0.2, bottom=df.low, color=color_index)
    ax.margins(0)

    # Adding Fibonacci measures as horizontal lines
    ax.axhline(y = df['fib_1'].values[-1])
    ax.axhline(y = df['fib_0.75'].values[-1], color='purple')
    ax.axhline(y = df['fib_0.5'].values[-1])
    ax.axhline(y = df['fib_0.25'].values[-1], color='purple')
    ax.axhline(y = df['fib_0'].values[-1])
    
    # Indicating sessions (Asian Range, London, New-York) as colored boxs 
    loc_AR = df.loc[df['hour'].isin([20, 21, 22, 23])]
    loc_LO = df.loc[df['hour'].isin([1, 2, 3])]
    loc_NY = df.loc[df['hour'].isin([9, 10, 11])]
    loc_NY_PM = df.loc[df['hour'].isin([13, 14, 15])]

    ax.barh(loc_AR.low.min(), 1, align='edge',height=loc_AR.high.max()-loc_AR.low.min(), left=loc_AR.index, alpha=0.1, color='gray')
    ax.barh(loc_LO.low.min(), 1, align='edge',height=loc_LO.high.max()-loc_LO.low.min(), left=loc_LO.index, alpha=0.1, color='green')
    ax.barh(loc_NY.low.min(), 1, align='edge',height=loc_NY.high.max()-loc_NY.low.min(), left=loc_NY.index, alpha=0.1, color='orange')
    ax.barh(loc_NY_PM.low.min(), 1, align='edge',height=loc_NY_PM.high.max()-loc_NY_PM.low.min(), left=loc_NY_PM.index, alpha=0.15, color='yellow')


    fig.savefig(f"/Volumes/NO NAME/graph_image_{i}.png")
    print(i)

    return 
  
  
#numba_function = numba.jit(download_image_as_array)

if __name__ == '__main__':
  image_obj, _ = Parallel(n_jobs=3, backend='multiprocessing', max_nbytes=None)(delayed(download_image_as_array)(df, i) for df, i in zip([df*100], range(23000, 23000+100)))

print(image_obj)
i = 23000
for image in image_obj:
    image.savefig(f"/Volumes/NO NAME/graph_image_{i}.png")
    i+=1



print('done')

#for i in range(23000, len(df) - WINDOW - 100):
    
  #  rolling_df = df.iloc[i:i+WINDOW]
    
    #numba_function(rolling_df, i)
  #  print(i)
    