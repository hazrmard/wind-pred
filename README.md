# wind-pred

This is a model that learns the speed and direction of wind to predict. The three output variables are:

1. Speed (m/s)
2. dx (the cosine of heading)
3. dy (the sine of heading)

## Getting started

Install python. Create a virtual environment. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### Training

Run the python command:

```bash
python windpred.py train
```

Once training is finished, you can verify that the model is working by running:

```bash
python windpred.py test
```

This will load a validation dataset and output the RMSE measure for Speed and direction (dx, dy components).

### Running the model

```python
from windpred import load_combined_model, combined_predict, INTERVAL

# Create a dataframe containing past measurement(s)
# make sure frequency of input data is the same as INTERVAL,
# that is, timestamps of rows are INTERVAL apart
dft = pd.DataFrame(
    # The data:
    dict(Speed=[10, 9], dx=[0.5, 0.4], dy=[0.3, 0.6]),
    # The corresponding time stamps  at consistent intervals:
    index=[pd.Timestamp(2023,11,16,12,15,45), pd.Timestamp(2023,11,16,12,30,45), ..]
    )

# Or, you can load the ERDC data and create dataframes of contiguous sequences:
from windpred import load_erdc_data, find_sequences
alldata = load_erdc_data()
seqs = find_sequences(alldata, interval=pd.Timedelta(minutes=15))
# seqs is a list of dataframes like `dft` above
dft = seqs[0] # or any index in seqs

# Load the model
res = load_combined_model()
# Feed the data and the model to the prediction function, and get the updated
# model and the prediction dataframe. `pred` is a dataframe like `dft`
res, pred = combined_predict(dft, res)
# Repeat as needed
res, pred = combined_predict(new_dft, res)

# Simulate new timeseries of length 1000, where `simulation` is a dataframe like `dft`
res, simulation = combined_simulate(1000, dft, res)
```