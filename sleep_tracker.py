import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#TODO Setup Github
#TODO CSV Format


def create_dummy_csv(filename):
    # create dummy csv
    pd.DataFrame({'date': [20171101, 20171102, 20171103],
                  'hour': [9, 10, 2],
                  'no_hours': [7, 8, 12],
                  'feel': [1, 2, 3]})\
      .to_csv('data/{}.csv'.format(filename), encoding='utf-8', index=False)
    print('saved file')













# create_dummy_csv('dummy_data')


