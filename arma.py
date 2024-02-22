"""Wind prediction model"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMAResults, ARIMA
from typing import Tuple, List, Iterable, Union
from tqdm.autonotebook import trange, tqdm
import statsmodels.api as sm

from data import read_erdc_file, find_sequences


INTERVAL = pd.Timedelta(minutes=15)


def pretrain(model_class, data: pd.DataFrame, exog=None, **kwargs) -> ARIMA:
    """
    Fit a model to the data
    """
    model = model_class(data, exog=exog, **kwargs)
    return model.fit()


def update(
        result: Union[ARIMAResults,ARIMA],
        data: Union[pd.DataFrame, Iterable[pd.DataFrame]],
        exog,
        interval,
        append=False,
        model_kwargs={}
    ) -> ARIMAResults:
    """
    Update the model with new data. By default, the model does not keep the old data.
    """
    if isinstance(data, pd.DataFrame):
        data = [data]
        exog = [exog]
    if result.__name__=='ARIMA' and isinstance(result, type):
        result = pretrain(result, data[0], exog=exog[0], **model_kwargs)
    for i, d in enumerate(data[1:]):
        if (d.index[-1] - result.data.dates[-1]) > interval:
            result = result.apply(d, exog=exog[i+1], refit=True, copy_initialization=True)
        else:
            if append:
                result = result.append(d, exog=exog[i+1], refit=True)
            else:
                result = result.extend(d, exog=exog[i+1], refit=True)
    return result
train = update


def predict(result: ARIMAResults, data: pd.DataFrame, exog, interval, in_sample=True) -> Tuple[ARIMAResults, float]:
    if data.index[0] - result.data.dates[-1] > interval or result.data.endog is None:
       result = result.apply(data, exog=exog)
    else:
        result = result.append(data, exog=exog)
    if in_sample:
        pred = result.predict(start=data.index[0], dynamic=False)
    else:
        # pred = result.forecast(steps=1)
        pred = result.predict(start=data.index[0], dynamic=0)
    return result, pred


def train_model(interval, order, df, endog_col, exog_cols, seq_kwargs={}) -> ARIMAResults:
    seqs = find_sequences(df, interval, **seq_kwargs)
    endog = seqs[0][endog_col]
    exog = exog_cols if exog_cols is None else seqs[0][exog_cols]
    res = pretrain(sm.tsa.ARIMA, endog, exog=exog, order=order)
    for dft in tqdm(seqs[1:], leave=False, desc='training'):
        endog = dft[endog_col]
        exog = exog_cols if exog_cols is None else dft[exog_cols]
        res = update(res, endog, exog=exog, interval=interval)
    return res


def save_combined_model(models, basename='bin/arma_model'):
    for i, m in enumerate(models):
        m.save(f'{basename}_{i}.pkl', remove_data=True)


def load_combined_model(basename='bin/arma_model') -> List[ARIMAResults]:
    from glob import glob
    files = glob(f'{basename}_*.pkl')
    files.sort()
    models = []
    for f in files:
        models.append(ARIMAResults.load(f))
    return models


def combined_predict(df, res=Tuple[ARIMAResults, ARIMAResults, ARIMAResults], interval=INTERVAL):
    arma_speed, arma_dx, arma_dy = res
    arma_speed, pred_speed = predict(arma_speed, df.Speed, None, interval=interval)
    arma_dx, pred_dx = predict(arma_dx, df.dx, None, interval=interval)
    arma_dy, pred_dy = predict(arma_dy, df.dy, None, interval=interval)
    pred = pd.concat((pred_speed, pred_dx, pred_dy), axis=1)
    pred.columns = ['Speed', 'dx', 'dy']
    return (arma_speed, arma_dx, arma_dy), pred


def combined_simulate(n, df, res=Tuple[ARIMAResults, ARIMAResults, ARIMAResults]):
    arma_speed, arma_dx, arma_dy = res
    sim_speed = arma_speed.apply(df.Speed).simulate(n)
    sim_dx = arma_dx.apply(df.dx).simulate(n)
    sim_dy = arma_dy.apply(df.dy).simulate(n)
    pred = pd.concat((sim_speed, sim_dx, sim_dy), axis=1)
    pred.columns = ['Speed', 'dx', 'dy']
    return res, pred



if __name__=='__main__':
    import sys
    df = read_erdc_file(processed=True)
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
        rmse_dy = sm.tools.eval_measures.rmse(pred.dy, seqs[0].dy)
        print(f'rmse_speed (m/s)={rmse_speed:.2f}')
        print(f'rmse_dx (cos(direction))={rmse_dx:.2f}')
        print(f'rmse_dy (sin(direction))={rmse_dy:.2f}')