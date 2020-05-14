import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns; sns.set()
from helpers import *
from SIR_Model import *
from scipy.integrate import odeint





def main():
    ## sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to",
                            ('Data Exploratory', 'Forecast Model', 'SIR Simulation'))

    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses JHU data available in [Github]"
        "(https://github.com/CSSEGISandData/COVID-19) repository.\n\n"
        "It is maintained by Marshall Zhao. \n\n"
        "Welcome to check out my flying videos [Here](https://www.bilibili.com/video/BV12Q4y1A7B1)"
    )

    if page == 'Data Exploratory':
        st.title('Explore County Level Data ')
        # load data
        total = load_data('total_data.pkl')
        # filter target county
        county_name = total['Combined_Key'].unique()
        county = st.selectbox("Select a County", county_name)
        df = total.loc[total['Combined_Key'] == county]

        # drawing
        base = alt.Chart(df).mark_bar().encode( x='monthdate(Date):O',).properties(width=500)

        red = alt.value('#f54242')
        a = base.encode(y='Confirmed').properties(title='Total Confirmed')
        st.altair_chart(a,use_container_width=True)

        b = base.encode(y='Deaths', color=red).properties(title='Total Deaths')
        st.altair_chart(b,use_container_width=True)

        c = base.encode(y='New Cases').properties(title='Daily New Cases')
        st.altair_chart(c,use_container_width=True)

        d = base.encode(y='New deaths', color=red).properties(title='Daily New Deaths')
        st.altair_chart(d,use_container_width=True)


    elif page == 'SIR Simulation':
        st.title('SIR Simulation')
        st.subheader('SIR simulation with customized parameters')
        N = st.slider('Input the populaiton', 100000,10000000, step = 100000,value = 3000000)
        I0 = st.slider('Input initial infection',1,5000,step = 5,value = 200)
        R0 = st.slider('Input initial removed',0,1000,step = 1,value = 0)
        beta = st.number_input('Input beta', min_value=0.0, max_value=10.0,value = 0.2)
        gamma = st.number_input('Input gamma',min_value = 0.0, max_value = 1.0, value = 0.08)

        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - R0
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        # A grid of time points (in days)
        t = np.linspace(0, 200, 500)

        # The SIR model differential equations.
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt

        # Initial conditions vector
        y0 = S0, I0, R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T

        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, S / N, 'b', lw=2, label='Susceptible')
        ax.plot(t, I / N, 'r', lw=2, label='Infected')
        ax.plot(t, R / N, 'g', lw=2, label='Recovered with immunity')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Number (1000s)')
        ax.set_ylim(0, 1.2)
        legend = ax.legend()
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()
        st.pyplot()\


    else:
        st.title('County Level Covid-19 Forecast Model')
        st.subheader('This is a demo of the dynamic SIR model')
        selected = st.radio('Select a county for demo',('NYC','Westerchester','Nassau'))
        df2 = load_data('{}.pkl'.format(selected.lower()))

        if st.checkbox('Show raw data'):
            st.write(df2)


        ## Model training

        # split train dataset
        train_df = df2[df2['Date'] < '2020-04-20']
        test_df = df2[(df2['Date'] > '2020-04-20') & (df2['Date'] < '2020-05-01')]

        # initialize model
        '## Training the Model'
        population = df2.Population[1]
        model = Train_Dynamic_SIR(epoch=1000, data=train_df,
                                  population=population, gamma=1 / 14, c=1, b=-10, a=0.08)

        # train the model
        estimate_df = model.train()

        # drawing
        'Training is completed, here is the result of the best fitted parameters'

        model.plot_beta_R0(train_df)
        st.pyplot()

        "## Let's go head forecasting the future"

        # initialize parameters for prediction
        population = model.population
        I0 = train_df['I'].iloc[-1]
        R0 = train_df['R'].iloc[-1]
        S0 = population - I0 - R0
        est_beta = model.beta
        est_alpha = model.a
        est_b = model.b
        est_c = model.c


        forecast_period = st.slider("Choose the forecast period", 5, 60, value=14)

        prediction = Predict_SIR(pred_period=forecast_period, S=S0, I=I0, R=R0, gamma=1 / 14,
                                 a=est_alpha, c=est_c, b=est_b, past_days=train_df['Day'].max())

        deaths = st.slider("Input a realistic death rate ", 0.0, 0.3, value = 0.1)

        result = prediction.run(death_rate=deaths)  # death_rate is an assumption

        'Prediction is completed'

        prediction.plot(start=train_df['Date'].max())
        '**Forecast for next {}** days'.format(forecast_period)
        st.pyplot()

        '**Accuracy: Real data VS Predicted**'
        prediction.MAPE_plot(test_df, result)
        st.pyplot()

main()

