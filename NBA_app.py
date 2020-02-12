
import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go



#regression packages
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


#for validating your classification model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# feature selection
from sklearn.feature_selection import RFE

# grid search
from sklearn.model_selection import GridSearchCV

from collections import deque

st.title('NBA Net Rating Machine')
st.markdown('_Please see left sidebar for more details._')

currentStats = pd.read_csv('https://raw.githubusercontent.com/neelganta/neel_project/master/updated2020.csv') 
regModel = pd.read_csv('https://raw.githubusercontent.com/neelganta/neel_project/master/githubRegression.csv')
regModel = regModel.fillna(0)

y = regModel['NET_RATING'] 
X = regModel.drop(['NET_RATING'], axis =1)
# Fit the model below
model1 =  lm.LinearRegression() #higher alpha (penality parameter), fewer predictors
model1.fit(X, y)
model1_y = model1.predict(X)

players = []
players = currentStats['Player']
players= deque(players) 
players.appendleft('2020 NBA Players') 
players = list(players) 

player1 = st.selectbox('Select first player:', players)
player2 = st.selectbox('Select second player:', players)
player3 = st.selectbox('Select third player:', players)
player4 = st.selectbox('Select fourth player:', players)
player5 = st.selectbox('Select fifth player:', players)


playerlist = [player1, player2, player3, player4, player5]

# playerdict = st.multiselect("Select 5 players for your lineup: ", players)

if(player1 != '2020 NBA Players' and player2 != '2020 NBA Players' and player3 != '2020 NBA Players' and player4 != '2020 NBA Players' and player5 != '2020 NBA Players'):
    userdf = pd.DataFrame(playerlist)
    userdf['Player'] = userdf[0]

    merged = userdf.merge(right = currentStats, on ='Player')
    merged['index'] = merged[0]
    merged['index'] = 'lineup'
    merged.set_index('index')
    merged.drop(['Player'], axis =1)
    dictionary = merged.groupby('index').apply(lambda dfg: dfg.drop('index', axis=1).to_dict(orient='list')).to_dict()
    converted = pd.DataFrame.from_dict(dictionary, orient= 'index')
    new_df = pd.concat([pd.DataFrame(col.tolist()).add_prefix(i) 
                        for i,col in converted.items()],axis = 1)

    new_df.index = converted.index
    new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='00')))]
    new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='01')))]
    new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='02')))]
    new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='03')))]
    new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='04')))]
    new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='5')))]
    new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='6')))]

    st.write('Lineup DataFrame:')
    st.write(new_df)
    st.write('           ')
    st.write('           ')
    st.write('           ')
    import itertools
    x = []
    average = []
    t=list(itertools.permutations(playerlist,len(playerlist)))
    for i in range(0,len(t)):
        x = t[i]
        userdf = pd.DataFrame(x)
        userdf['Player'] = userdf[0]

        merged = userdf.merge(right = currentStats, on ='Player')
        merged['index'] = merged[0]
        merged['index'] = 'lineup'
        merged.set_index('index')
        merged.drop(['Player'], axis =1)
        dictionary = merged.groupby('index').apply(lambda dfg: dfg.drop('index', axis=1).to_dict(orient='list')).to_dict()
        converted = pd.DataFrame.from_dict(dictionary, orient= 'index')
        new_df = pd.concat([pd.DataFrame(col.tolist()).add_prefix(i) 
                            for i,col in converted.items()],axis = 1)

        new_df.index = converted.index
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='00')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='01')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='02')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='03')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='04')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='5')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='6')))]
        new_df = new_df[new_df.columns.drop(list(new_df.filter(regex='Player')))]
        user_pred = model1.predict(new_df)
        num = int(user_pred)
        average.append(num)


    avg = sum(average) / len(average)

    string = str(round(avg, 2))


    if(avg < 0):
        st.error("The predicted Net Rating for this lineup is " + string +".")
    elif (avg > 10): 
        st.success("The predicted Net Rating for this lineup is " + string +".")
    else:
        st.write("The predicted Net Rating for this lineup is " + string +".")

st.markdown('_Presented by Neel Ganta._')
# st.sidebar.markdown()

st.sidebar.markdown('**ABOUT THE NBA NET RATING MACHINE:**  The _NBA Net Rating Machine_ was first incepted roughly one year ago while Neel Ganta was browsing through https://stats.nba.com/lineups/advanced/. He discovered a large set of lineup data, and a current lineup problem in the NBA. Should teams go small? Three shooters? Five? How can we see what our team would look like with a player _before_ trading for him? Seeing a problem and no publicly available solution, Neel decided to create what could be the next big GM tool. Please enjoy the _NBA Net Rating Machine_ which allows you to input **any** five players in the NBA, and predicts an overall Net Rating for the lineup.')
st.sidebar.markdown('**ABOUT NEEL GANTA**: Neel Ganta is graduating with a Finance and Computer Science degree from Kansas State, and completed internships at the Federal Reserve, JPMorgan Chase, the Boston Celtics, and currently sereves as an analytics consultant for Brad Underwood, Head Basketball Coach at University of Illinois. Neel grew up using his passion for basketball to connect with others, and can be found playing 5 on 5 in his local city league tournament or rec center. When he is taking a break from practicing dunks and _NBA_ three pointers, he is sharpening his machine learning skills and seeking new avenues to provide basketball insights.')
# st.sidebar.video('https://youtu.be/-OoM5XvLo20')
st.sidebar.markdown('**The Neel Ganta Fighting Illini Story:**')
st.sidebar.video(data = 'https://www.youtube.com/watch?v=Zfw0AevYR-4')
st.sidebar.markdown('**CONTACT:**')
st.sidebar.markdown('neelganta@gmail.com')
st.sidebar.markdown('https://www.linkedin.com/in/neelganta/')
