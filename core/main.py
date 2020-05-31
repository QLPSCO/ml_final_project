import pickle

import pandas as pd

from pipeline.pipeline import Pipeline

# read datafame
df = pd.read_pickle("./datasets/h1b_2019.pkl")

# load into pipeline
pl = Pipeline()
pl.load_data(df)
pl.train_test_split("CASE_STATUS")  # TODO: redo to set_target





































