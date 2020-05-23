import streamlit as st
import pandas as pd

@st.cache
def load_data(name):
    return pd.read_pickle('./data/' + name)

@st.cache
def clean_country(df):
    us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'American Samoa': 'AS',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Guam': 'GU',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Northern Mariana Islands': 'MP',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Puerto Rico': 'PR',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virgin Islands': 'VI',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY'
    }

    mask = df.Combined_Key.str.split(',').str[1].str.lstrip()

    df = df.loc[~mask.isin(['US', 'Wuhan Evacuee'])]

    mask = df.Combined_Key.str.split(',').str[1].str.lstrip()

    df['code'] = [us_state_abbrev[i] for i in mask]

    df['state'] = df.Combined_Key.str.split(',').str[1].str.lstrip()

    df.loc[df.code.isin(['DC', 'MP', 'VI']), 'Population'] = 'Not Available'

    df = df[df['Date'] == df.iloc[-1].Date]

    df = df.groupby(['code', 'state'])[['Confirmed', 'Deaths', 'Population']].agg(
        {'Confirmed': 'sum', 'Deaths': 'sum', 'Population': 'max'}).reset_index()

    df['text'] = df.state + '<br>' + \
                 'Population:' + df['Population'].astype('str') + '<br>' + \
                 'Total Deaths:' + df['Deaths'].astype('str')
    return df
