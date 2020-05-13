import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


st.title('County Level Covid-19 Forecast Model')
st.subheader('This is a demo of the dynamic SIR model with NYC Data')
"Here's the pre-processed data for NYC"


df = pd.read_pickle('new_york_data.pkl')
st.write(df)
st.write('However, this data is not ready to use yet. We"ll have to impute the realistic recovery data.')
#### data processing
# fix number problem
df.loc[0, 'New Cases'] = 2500
df.loc[0, 'New deaths'] = 30

df.loc[32, 'Confirmed'] = int((df.loc[33, 'Confirmed'] + df.loc[31, 'Confirmed']) / 2)
df.loc[32, 'Active'] = df.loc[32, 'Confirmed'] - df.loc[32, 'Deaths']
df.loc[32, 'New Cases'] = df.loc[32, 'Confirmed'] - df.loc[31, 'Confirmed']

# calculate and make up recovery data
# assume fatality rate of 30%, we fill recovery data

for i in range(27):
    df.loc[i + 14, 'New recovered'] = int(df.loc[i, 'New Cases'] * 0.7)

past_new_case = [21, 57, 70, 153, 355, 618, 642, 1028, 2115, 2446, 2948, 3677, 3983,
                 2600]  # data from nyc health https://github.com/nychealth/coronavirus-data

for i in range(14):
    df.loc[i, 'New recovered'] = int(past_new_case[i] * 0.9)

# calculate total recovery data
df.loc[0, 'Recovered'] = 2000
for i in range(1, 41):
    df.loc[i, 'Recovered'] = df.loc[i - 1, 'Recovered'] + df.loc[i, 'New recovered']

# calculate and rename cols
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
df['R'] = df['Recovered'] + df['Deaths']
df = df.rename(columns={'Active': 'I'})
df['Day'] = np.arange(41)

###############
'After imputation, our data is ready for the model! : )'
st.write(df)

## Model training
from SIR_Model import *

# split train dataset
train_df = df[df['Date'] < '2020-04-20']
test_df = df[(df['Date'] > '2020-04-20') & (df['Date'] < '2020-05-01')]

# initialize model
model = Train_Dynamic_SIR(epoch = 1000, data = train_df,
                             population = 8336817, gamma =1/18, c = 1, b = -10, a = 0.08)

# train the model
estimate_df = model.train()

#############
'After training our forecast modeld, here is the result of best fitted parameters'

model.plot_beta_R0(train_df)
st.pyplot()

'Let"s go head forecasting the future'


# initialize parameters for prediction
population = model.population
I0 = train_df['I'].iloc[-1]
R0 = train_df['R'].iloc[-1]
S0 = population - I0  - R0

est_beta = model.beta
est_alpha = model.a
est_b = model.b
est_c = model.c

forecast_period = st.slider("Forecast Length",0,60)

prediction = Predict_SIR(pred_period=forecast_period, S=S0, I=I0, R=R0, gamma=1/14,
                       a = est_alpha, c = est_c, b = est_b, past_days = train_df['Day'].max())

deaths = st.slider("Death Rate",0.0,0.2,0.03)

result = prediction.run(death_rate = deaths) # death_rate is an assumption

prediction.plot(start = train_df['Date'].max())
'**Here is the forecast result**'
st.pyplot()

'**Here is the accuracy test**'
prediction.MAPE_plot(test_df, result)
st.pyplot()

import os

with open(os.path.join('/Users/marshall/Desktop/covid-19 model/streamlit/', 'Procfile'), "w") as file1:
    toFile = 'web: sh setup.sh && streamlit run <app name>.py'

file1.write(toFile)

