import pandas as pd 
import numpy as np

WINDOW = 60

SEASONALS = [-1, 0, 1, 1, 0, -1, 1, 0, 0, 1, 1, 1]
DAILY_INDICATORS = ['bull_ob', 'bear_ob', 'bull_bb', 'bear_bb', 'bull_low_liquidity', 'bear_high_liquidity']


def get_15min_data(start, indicator=True, economic_calendar = True):
    """Returns the S&P 500 historical 15min data + some custom indicators if indicators = True"""
    
    df = pd.read_csv("/Users/thomasrigou/Downloads/es-15m.csv", delimiter=';', names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.iloc[start:].reset_index()

    df['index'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S', exact=True, infer_datetime_format=True)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', exact=True, infer_datetime_format=True)

    df['hour'] = df['index'].dt.hour
    df['weekday'] = df['index'].dt.weekday
    
    df = df.dropna()
    

    if indicator:
        df['daily_move'] = (df['close'] - df['open'])/ df['open']  * 100
        df = lows_highs(df)
        df = order_block(df)
        
        df['lows'] = df.apply(lambda x : x.low if x.lows == 1 else None,axis=1).ffill()
        df['highs'] = df.apply(lambda x : x.high if x.highs == 1 else None,axis=1).ffill()

    if economic_calendar:
        df = add_economic_calendar(df)
    
    df = df.fillna(0)
    df = df.drop(['time'] , axis=1)
    

    return df

def get_daily_data(start, indicator=True):
    """Returns the S&P 500 historical daily data + some custom indicators if indicators = True"""
    
    df = pd.read_csv("/Users/thomasrigou/Downloads/es-15m.csv", delimiter=';', names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.iloc[start:].reset_index()

    df['index'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M:%S', exact=True, infer_datetime_format=True)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', exact=True, infer_datetime_format=True)
    
    if indicator:
        df = df.groupby('date').agg({'open': 'first', 'high': np.max, 'low': np.min, 'close': 'last'})
        df['daily_move'] = (df['close'] - df['open'])/ df['open']  * 100
        df['ADR'] = (df['close'] - df['open']).rolling(5).mean().shift(1)
      
        df = fib(df) # FIBONACCI measures

    return df

def lows_highs(df):
    """Returns the dataframe with lows and highs indicated"""
    
    df['lows'] = ((df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1)))
    df['highs'] = ((df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1)))
    
    df['int_lows'] =  (df['low'] < df['low'].shift(2)) & (df['low'] < df['low'].shift(1)) &(df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2)) 
    df['int_highs'] =  (df['high'] > df['high'].shift(2)) & (df['high'] > df['high'].shift(1)) &(df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2)) 

    df['lows_p'] = df.apply(lambda x: x.low if x.lows else None, axis=1).ffill()
    df['highs_p'] = df.apply(lambda x: x.high if x.highs else None, axis=1).ffill()
    

    return df

def order_block(df):
    """Returns the dataframe with orderblocks indicated"""

    
    df['cond_bull_ob'] = (df['close'] < df['open']) & (df['close'].shift(-1) > df['open'].shift(-1))
    df['cond_bull_ob'] = df['cond_bull_ob'] & ((df['close'].shift(-2) > df['open'].shift(-2)) | (df['close'].shift(-3) > df['open'].shift(-3)) | (df['close'].shift(-1) > df['open'].shift(-1) * 1.001))
    df['cond_bull_ob'] = df['cond_bull_ob'] & (df['highs_p'] <= df['close'].shift(-8).rolling(8).max())
    df['cond_bull_ob'] = df['cond_bull_ob'] & (df['lows_p'] <= df['low'].shift(-7).rolling(4).min())
    
    df['cond_bear_ob'] = (df['close'] > df['open']) & (df['close'].shift(-1) < df['open'].shift(-1))
    df['cond_bear_ob'] = df['cond_bear_ob'] & ((df['close'].shift(-2) < df['open'].shift(-2)) | (df['close'].shift(-3) < df['open'].shift(-3)) | (df['close'].shift(-1) < df['open'].shift(-1) * 0.999))
    df['cond_bear_ob'] = df['cond_bear_ob'] & (df['lows_p'] >= df['close'].shift(-8).rolling(8).min())
    df['cond_bear_ob'] = df['cond_bear_ob'] & (df['highs_p'] >= df['high'].shift(-7).rolling(4).max())
    
   # df = df.drop(['cond_bear_ob', 'cond_bull_ob'], axis=1)

    return df
    
def breaker_block(df):
    """Returns the dataframe with breaker blocks indicated"""   
    
    df['cond_bull_bb'] =  ( df['int_highs'] | df['int_highs'].shift(-1) ) & (df['daily_move'] > 0)  & ((df['close'].shift(-1)*1.004 < df['low']) & ((df['low'].shift(-1) - df['close'].shift(-1))/df['low'].shift(-1) < 0.003) |  ((df['daily_move'].shift(-1) < 0) & (df['close'].shift(-2)*1.004 < df['low']) & ((df['low'].shift(-1) - df['close'].shift(-1))/df['low'].shift(-1) < 0.003))) & (df['high'].shift(-1)-df['open'].shift(-1) < df['open'].shift(-1) - df['close'].shift(-1) )
    df['cond_bear_bb'] =  ( df['int_lows'] | df['int_lows'].shift(-1)  ) & (df['daily_move'] < 0)  & ((df['close'].shift(-1)*0.996 > df['high']) & ((df['close'].shift(-1) - df['high'].shift(-1))/df['high'].shift(-1) < 0.003) |  ((df['daily_move'].shift(-1) > 0) & (df['close'].shift(-2)*0.996 > df['high']) & ((df['close'].shift(-1) - df['high'].shift(-1))/df['high'].shift(-1) < 0.003))) & (df['open'].shift(-1)-df['low'].shift(-1) < df['close'].shift(-1) - df['open'].shift(-1) )

    df['bull_bb'] = df.apply(lambda x : x.open if x.cond_bull_bb else None, axis=1).ffill().shift(2)
    df['bear_bb'] = df.apply(lambda x : x.open if x.cond_bear_bb else None, axis=1).ffill().shift(2)
    
    #df = df.drop(['cond_bear_bb', 'cond_bull_bb'], axis=1)

    return df

def add_economic_calendar(df): 
    
    """returns a dataframe with the number of news for the EUR, GBP, JPY, USD currencies with the specified news impact MEDIUM or HIGH"""
    
    news = pd.read_csv('/Users/thomasrigou/economic_calendar_2007_02_13_2023_02_28.csv')

    news['Start'] = pd.to_datetime(news['Start'])
    news = news.set_index(['Start'])
    news['date'] = news.index.date
    
    news = news.groupby(['date', 'Impact', 'Currency']).count()

    news = news.reset_index().set_index('date')


    news_m_gbp = news.loc[(news.Currency == 'GBP') & (news.Impact == 'MEDIUM')].Name
    news_m_usd = news.loc[(news.Currency == 'USD') & (news.Impact == 'MEDIUM')].Name
    news_m_jpy = news.loc[(news.Currency == 'JPY') & (news.Impact == 'MEDIUM')].Name
    news_m_eur = news.loc[(news.Currency == 'EUR') & (news.Impact == 'MEDIUM')].Name

    news_h_gbp = news.loc[(news.Currency == 'GBP') & (news.Impact == 'HIGH')].Name
    news_h_usd = news.loc[(news.Currency == 'USD') & (news.Impact == 'HIGH')].Name
    news_h_jpy = news.loc[(news.Currency == 'JPY') & (news.Impact == 'HIGH')].Name
    news_h_eur = news.loc[(news.Currency == 'EUR') & (news.Impact == 'HIGH')].Name



    news = pd.concat([news_h_eur, news_h_gbp, news_h_jpy, news_h_usd, news_m_eur, news_m_gbp, news_m_jpy, news_m_usd], axis=1, keys=['h_eur','h_gbp', 'h_jpy', 'h_usd', 'm_eur', 'm_gbp', 'm_jpy', 'm_usd'])
    news = news.fillna(0)
   
    df = pd.concat([df, news],axis = 1)
   

    return df 

def fib(df):
    """Returns the dataframe with fibonacci measures indicated
    to be used on higher dataframe"""

    
    price_range = df['high'].shift(1) - df['low'].shift(1)
    df['fib_1'] = df['high'].shift(1)
    df['fib_0.75'] = df['low'].shift(1) + price_range * 0.75
    df['fib_0.5'] = df['low'].shift(1) + price_range * 0.5
    df['fib_0.25'] = df['low'].shift(1) + price_range * 0.25
    df['fib_0'] = df['low'].shift(1)
    
    return df

def data_main(size):
    df = get_15min_data(size, economic_calendar=False)

    df_daily = get_daily_data(size).dropna()  

    df = pd.merge(df, df_daily, 'left', left_on='date', right_on='date', suffixes=('', '_d')).ffill().dropna().reset_index(drop=True)
    df = df.drop(['index', 'date', 'open_d', 'high_d', 'low_d', 'close_d', 'highs_p', 'lows_p',  'daily_move', 'daily_move_d', 'int_highs', 'int_lows'], axis=1) 
    #df = df.drop(['relative_range_position_20', 'relative_range_position_40', 'relative_range_position_60', 'bearish_market_structure', 'bullish_market_structure', 'ADR', 'bull_low_liquidity', 'bear_high_liquidity'], axis=1)      

    df = df*1 # converts bool to int
    
    return df
