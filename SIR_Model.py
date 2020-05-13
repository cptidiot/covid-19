import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import *



class Train_Dynamic_SIR:
    def __init__(self, data, population, epoch, gamma, c=1, b=-3, a=0.1):
        self.epoch = epoch
        self.steps = data.shape[0]

        # real data
        self.I = list(data['I'])
        self.R = list(data['R'])
        self.S = list(population - data['I'] - data['R'])

        # prediction
        self.I_pred = []
        self.R_pred = []
        self.S_pred = []
        self.past_days = data['Day'].min()

        #  parameters
        self.c = c
        self.b = b
        self.a = a

        self.beta = self._cal_beta(a=self.a, b=self.b, c=self.c, t=0)
        self.gamma = gamma
        self.population = population

        self.results = None
        self.estimation = None
        self.modelRun = False
        self.loss = None
        self.betalist = []

    def _cal_beta(self, a, b, c, t):
        """
        calculate a dynamic beta using logistic distribution
        """
        return c * exp(-a * (t + b)) * pow((1 + exp(-a * (t + b))), -2)

    def _cal_loss(self):
        return mean_squared_error(self.I, self.I_pred)

    def _cal_MAPE(self):
        y = np.array(self.I)
        y_pred = np.array(self.I_pred)
        return np.mean(np.abs((y - y_pred)) / np.abs(y))

    def _update(self):

        e = 2.71828

        # learning rate
        a_learn = 0.000000000000001
        b_learn = 0.00000000001
        c_learn = 0.0000000000001

        a_temp = b_temp = c_temp = 0

        for t in range(self.steps):
            f = (e ** (self.a * (t + self.b)))
            f2 = (e ** (-self.a * (t + self.b)))

            loss_beta = -2 * (self.I[t] - self.I_pred[t]) * (self.I_pred[t]) * t * self.S[t] / self.population
            beta_to_a = -self.c * f * (t + self.b) * (f - 1) * pow((1 + f), -3)
            beta_to_b = -self.c * f * self.a * (f - 1) * pow((1 + f), -3)
            beta_to_c = f2 * pow((1 + f2), -2)

            a_temp += loss_beta * beta_to_a  # new gradient
            b_temp += loss_beta * beta_to_b  # new gradient
            c_temp += loss_beta * beta_to_c  # new gradient

        self.a -= a_learn * a_temp;  # update values
        self.b -= b_learn * b_temp;
        self.c -= c_learn * c_temp;

    def train(self):
        for i in range(self.epoch):
            self.S_pred = []
            self.I_pred = []
            self.R_pred = []

            for t in range(self.steps):

                if t == 0:
                    self.S_pred.append(self.S[0])
                    self.I_pred.append(self.I[0])
                    self.R_pred.append(self.R[0])

                    self.beta = self._cal_beta(c=self.c, t=t, b=self.b, a=self.a)
                    # print("time {}, beta {}".format(t, self.rateSI))

                    # collect the optimal fitted beta
                    if i == (self.epoch - 1):
                        self.betalist.append(self.beta)

                else:
                    self.beta = self._cal_beta(c=self.c, t=t, b=self.b, a=self.a)
                    # print("time {}, beta {}".format(t, self.rateSI))

                    # collect the optimal fitted beta
                    if i == (self.epoch - 1):
                        self.betalist.append(self.beta)

                    # apply real data to SIR and calculate
                    S_to_I = (self.beta * self.S[t] * self.I[t]) / self.population
                    I_to_R = (self.I[t] * self.gamma)

                    self.S_pred.append(self.S[t] - S_to_I)
                    self.I_pred.append(self.I[t] + S_to_I - I_to_R)
                    self.R_pred.append(self.R[t] + I_to_R)

            # store the estimated number for the last iteration
            if i == (self.epoch - 1):
                self.estimation = pd.DataFrame.from_dict({'Time': list(range(len(self.S))),
                                                          'Estimated_Susceptible': self.S_pred,
                                                          'Estimated_Infected': self.I_pred,
                                                          'Estimated_Removed': self.R_pred},
                                                         orient='index').transpose()
                self.loss = self._cal_loss()
                MAPE = self._cal_MAPE()
            #            print("The loss in is {}".format(self.loss))
            #            print("The MAPE in the whole period is {}".format(MAPE))
            #            print("Optimial beta is {}".format(self.beta))

            ## calculate loss in each iteration
            self.loss = self._cal_loss()

            # print("The loss in iteration {} is {}".format(e, self.loss))
            # print("Current beta is {}".format(self.rateSI))

            ## ML optimization.
            self._update()  # Update parameters using Gradient Descent in each step
        return self.estimation  # the lastest estimation

    # ------------------------------------------ Plots -----------------------------------------------------------------
    def plot_beta_R0(self, train_df):

        fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex='col')

        ax[0].plot(self.estimation['Time'], self.betalist, color='green')
        ax[1].plot(self.estimation['Time'], [i / self.gamma for i in self.betalist], color='red')

        # date labels
        begin = train_df['Date'].min()
        num_days = train_df.shape[0]
        labels = list((begin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(num_days))
        plt.xticks(list(range(num_days)), labels, rotation=45, fontsize=12)

        # set titles
        ax[0].set_title('Fitted Dynamic Beta', fontsize=15)
        ax[1].set_title('Fitted Dynamic R0', fontsize=15)

        plt.show()

    def plot_fitting(self, train_df):

        fig, ax = plt.subplots(figsize=(16, 9))
        plt.plot(self.estimation['Time'], self.estimation['Estimated_Infected'], 'c--')
        plt.plot(self.estimation['Time'], train_df['I'], color='b')

        # set x tricks
        begin = train_df['Date'].min()
        num_days = train_df.shape[0]
        labels = list((begin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(num_days))
        plt.xticks(list(range(num_days)), labels, rotation=45, fontsize=12)

        # set title
        plt.title('Dynamic SIR Fitting', fontsize=15)
        plt.legend(['Estimated Infected', 'Real Infected'], fontsize='large')

        plt.show()

class Predict_SIR:
    """
    Added death_rate
    """

    def __init__(self, pred_period=150, S=1000000, I=1000, R=0, gamma=1 / 14,
                 a=0.3, c=5, b=-10, past_days=30):
        self.pred_period = pred_period  # number of prediction days
        self.S = S
        self.I = I
        self.R = R
        self.beta = None
        self.gamma = gamma

        self.population = S + I + R  # total population
        self.a = a
        self.c = c
        self.b = b
        self.past_days = past_days  # make prediction since the last observation
        self.results = None
        self.modelRun = False

    def _cal_beta(self, c: float, t: int, a: float, b: float, past_days: int):
        t = t + past_days
        return c * exp(-a * (t + b)) * pow((1 + exp(-a * (t + b))), -2)

    def run(self, death_rate):
        S = [self.S]
        I = [self.I]
        R = [self.R]

        for i in range(1, self.pred_period):
            self.beta = self._cal_beta(c=self.c, t=i, b=self.b,
                                       a=self.a, past_days=self.past_days)

            S_to_I = (self.beta * S[-1] * I[-1]) / self.population
            I_to_R = (I[-1] * self.gamma)

            S.append(S[-1])
            I.append(I[-1] + S_to_I - I_to_R)
            R.append(R[-1] + I_to_R)

        # deaths = death_rate * num_of_infections
        Death = list(map(lambda x: (x * death_rate), I))
        # heal = removed - deaths
        Heal = list(map(lambda x: (x * (1 - death_rate)), R))

        self.results = pd.DataFrame.from_dict({'Time': list(range(len(S))),
                                               'S': S, 'I': I,
                                               'R': R,
                                               'Death': Death, 'Heal': Heal},
                                              orient='index').transpose()
        self.modelRun = True
        return self.results

    # -------------------------------------- Plots ----------------------------------------------------------------
    def plot(self, start , S=False):

        print("Maximum Active case: ",
              format(int(max(self.results['I']))))

        fig, ax = plt.subplots(figsize=(16, 9))

        if S is True:
            plt.plot(self.results['Time'], self.results['S'], color='blue')
        plt.plot(self.results['Time'], self.results['I'], color='red')
        plt.plot(self.results['Time'], self.results['R'], color='palegreen')
        plt.plot(self.results['Time'], self.results['Heal'], color='green')
        plt.plot(self.results['Time'], self.results['Death'], color='grey')

        # set x trick
        begin = start
        num_days = len(self.results)
        labels = list((begin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(num_days))
        plt.xticks(list(range(num_days)), labels, rotation=45)

        # legend
        if S is True:
            plt.legend(['Susceptible', 'Infected', 'Removed', 'Heal', 'Death'], fontsize='large')
        else:
            plt.legend(['Infected', 'Removed', 'Heal', 'Death'], fontsize='large')

        plt.title('Prediction', fontsize=15)
        plt.show()

    # sensitivity analysis
    def MAPE_plot(self, test, predict_data):
        y = test["I"].reset_index(drop=True)
        y_pred = predict_data[:len(test)]['I'].reset_index(drop=True)
        mape = np.mean(np.abs((y - y_pred)) / np.abs(y))
        print("The MAPE is: ".format(mape))
        print(mape)

        fig, ax = plt.subplots(figsize=(16, 9))

        plt.plot(test['Date'], y, color='steelblue')
        plt.plot(pd.date_range('2020-04-20', periods=self.pred_period, freq='d'), predict_data['I'], color='orangered')

        plt.title('Predicted VS Real Infection', fontsize=15)
        plt.legend(['Observation', 'Prediction'], loc='upper left', prop={'size': 12},
                   fancybox=True, shadow=True)
        plt.show()