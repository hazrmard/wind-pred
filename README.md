# wind-pred

## Getting started

Install python. Create a virtual environment. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### Training

Download the `ERDCSlow.hdf5` file and place in directory.

Then,

```bash
python windpred.py train
```

### Running the model


```python
from windpred import load_combined_model, combined_predict, INTERVAL

# make sure frequency of input data is the same as INTERVAL,
# that is, timestamps are INTERVAL apart
# dx, dy = cos(), sin() of direction
dft = pd.DataFrame(
    dict(Speed=[...], dx=[...], dy=[...]),
    index=[pd.Timestamp(2023,11,16,12,15,45), pd.Timestamp(2023,11,16,12,30,45), ..]
    )

res = load_combined_model()
res, pred = combined_predict(dft, res)
# ...
res, pred = combined_predict(new_dft, res)
```