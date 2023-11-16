"""Wind prediction model"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMAResults, ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from typing import Tuple
from tqdm.autonotebook import trange
import statsmodels.api as sm


INTERVAL = pd.Timedelta(minutes=15)


def read_file(dir_path='./', processed=False) -> pd.DataFrame:
    if os.path.exists(dir_path+'ERDCSlow.hdf5') and processed:
        df = pd.read_hdf(dir_path+'ERDCSlow.hdf5', key='table', index_col='Time')
    else:
        df = pd.read_csv(dir_path + 'ERDCSlow.csv',
                        index_col='Time', parse_dates=['Time'])
        cols = []
        for col in df.columns:
            c = pd.to_numeric(df[col], errors='coerce', downcast='float')
            cols.append(c)
        df = pd.concat(cols, axis=1)

        df = df[['Speed', 'Direction', 'Temp']]
        df = df.dropna(axis=0, how='any')
        df.to_hdf(dir_path + '/ERDCSlow.hdf5', key='table')
    df['dx'] = np.cos(df.Direction)
    df['dy'] = np.sin(df.Direction)
    return df



def find_sequences(df, interval=pd.Timedelta(minutes=1), agg_fn='mean'):
    resampler = df.resample(interval)
    if agg_fn=='mean':
        dfm = resampler.mean()
    elif agg_fn=='max':
        dfm = resampler.max()
    elif isinstance(agg_fn, float):
        dfm = resampler.quantile(agg_fn)
    i, ii = 0, dfm.index[0]
    j, jj = 1, dfm.index[1]
    seqs = []
    while j < len(dfm):
        while i < len(dfm):
            if not np.isnan(dfm.Speed.iloc[i]):
                break
            i += 1
        j = i+1
        while j <= len(dfm):
            if j==len(dfm) or np.isnan(dfm.Speed.iloc[j]):
                seqs.append(dfm.iloc[i:j])
                i = j+1
                break
            j += 1

    for s in seqs:
        s.index.freq = interval

    return [s for s in seqs if len(s)>=4]



def pretrain(model_class, data: pd.DataFrame, exog=None, **kwargs) -> ARIMA:
    model = model_class(data, exog=exog, **kwargs)
    return model.fit()



def update(result: ARIMAResults, data: pd.DataFrame, exog, interval, append=True):
    if data.index[-1] - result.data.dates[-1] > interval:
        model = result.model.clone(data, exog=exog)
        result = model.fit(start_params=result.params)
    else:
        if append:
            result = result.append(data, exog=exog, refit=True)
        else:
            result = result.extend
    return result



def predict(result: ARIMAResults, data: pd.DataFrame, exog, interval, in_sample=True) -> Tuple[ARIMAResults, float]:
    if data.index[0] - result.data.dates[-1] > interval:
       result = result.apply(data, exog=exog)
    else:
        result = result.append(data, exog=exog)
    if in_sample:
        pred = result.predict(start=data.index[0], dynamic=False)
    else:
        # pred = result.forecast(steps=1)
        pred = result.predict(start=data.index[0], dynamic=0)
    return result, pred



def trial(
    interval, order, df, seq_kwargs={}, filter_fn=lambda df: df, split=0.25,
    endog_col='Speed', exog_cols=None
):
    df = filter_fn(df)
    seqs = find_sequences(df, interval, **seq_kwargs)
    # leave one out testing
    rmses = []
    rmses_persistent = []
    ntest = 0
    splits = int(len(seqs) * split)

    for i in trange(len(seqs), leave=False, desc='splits'):
        if isinstance(split, float) and 0<split<1:
            ntrain = int(len(seqs[i]) * (1-split))
            dtrain = [seqs[i].iloc[:ntrain]]
            dtest = [seqs[i].iloc[ntrain:]]
        else:
            if i==split:
                break
            dtest = [seqs[i]]
            dtrain = [seqs[j] for j in range(splits) if j!=i]

        endog = dtrain[0][endog_col]
        exog = exog_cols if exog_cols is None else dtrain[0][exog_cols]
        res = pretrain(sm.tsa.ARIMA, endog, exog=exog, order=order)
        for dft in dtrain[1:]:
            endog = dft[endog_col]
            exog = exog_cols if exog_cols is None else dft[exog_cols]
            res = update(res, endog, exog=exog, interval=interval)
        for dft in dtest:
            if dft.index[0] < res.data.dates[-1]:
                dft = dft.copy()
                dft.index += (res.data.dates[-1] - dft.index[0]) + (2 * interval)
                dft.index.freq = interval
            ntest += len(dft)
            # predictions = []
            # for k in trange(len(dft.Speed), leave=False, desc='instance'):
            #     res, pred = predict(res, dft.Speed.iloc[k:k+1], interval)
            #     predictions.append(pred)
            # pred = pd.concat(predictions)
            endog = dft[endog_col]
            exog = exog_cols if exog_cols is None else dft[exog_cols]
            res, pred = predict(res, endog, exog, interval)
            rmses.append(sm.tools.eval_measures.rmse(pred, endog) * len(dft))
            rmses_persistent.append(sm.tools.eval_measures.rmse(endog.iloc[:-1], endog.iloc[1:]) * (len(dft)-1))
            print(f'Split {i} \tRMSE: {rmses[-1]/len(dft):.2f}')
    return dict(
        rmse=sum(rmses) / ntest,
        rmse_persistent=sum(rmses_persistent) / (ntest - splits)
    )



def train_model(interval, order, df, endog_col, exog_cols, seq_kwargs={}):
    seqs = find_sequences(df, interval, **seq_kwargs)
    endog = seqs[0][endog_col]
    exog = exog_cols if exog_cols is None else seqs[0][exog_cols]
    res = pretrain(sm.tsa.ARIMA, endog, exog=exog, order=order)
    for dft in seqs[1:]:
        endog = dft[endog_col]
        exog = exog_cols if exog_cols is None else dft[exog_cols]
        res = update(res, endog, exog=exog, interval=interval)
    return res


def save_arma_model(res, path):
    pass



def load_arma_model(path) -> ARIMA:
    pass


def save_combined_model(models, path='arma_model.pickle'):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(models, f)


def load_combined_model(path='arma_model.pickle'):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)



def combined_predict(df, res=Tuple[ARIMAResults, ARIMAResults, ARIMAResults], interval=INTERVAL):
    arma_speed, arma_dx, arma_dy = res
    arma_speed, pred_speed = predict(arma_speed, df.Speed, None, interval=interval)
    arma_dx, pred_dx = predict(arma_dx, df.dx, None, interval=interval)
    arma_dy, pred_dy = predict(arma_dy, df.dy, None, interval=interval)
    pred = pd.concat((pred_speed, pred_dx, pred_dy), axis=1)
    pred.columns = ['Speed', 'dx', 'dy']
    return(arma_speed, arma_dx, arma_dy), pred



if __name__=='__main__':
    import sys
    df = read_file(processed=True)
    dfog = df # original dataframe with all data
    # Testing data
    dft = dfog.loc['2022-06-28':'2022-07-07']
    # Training data
    df = dfog.loc['2022-03-10':'2022-06-27']
    if sys.argv[1]=='train':
        print('Training models')
        arma_speed = train_model(INTERVAL, (1,0,20), df, 'Speed', None)
        arma_dx = train_model(INTERVAL, (1,0,20), df, 'dx', None)
        arma_dy = train_model(INTERVAL, (1,0,20), df, 'dy', None)

        save_combined_model((arma_speed, arma_dx, arma_dy))
    else:
        print('Loading and testing model')
        res = load_combined_model()
        seqs = find_sequences(dft, INTERVAL)
        res, pred = combined_predict(seqs[0], res)
        print(pred.head())

        rmse_speed = sm.tools.eval_measures.rmse(pred.Speed, seqs[0].Speed)
        rmse_dx = sm.tools.eval_measures.rmse(pred.dx, seqs[0].dx)
        rmse_dy = sm.tools.eval_measures.rmse(pred.dx, seqs[0].dy)
        print(f'rmse_speed (m/s)={rmse_speed:.2f}')
        print(f'rmse_dx (cos(direction))={rmse_dx:.2f}')
        print(f'rmse_dy (sin(direction))={rmse_dy:.2f}')