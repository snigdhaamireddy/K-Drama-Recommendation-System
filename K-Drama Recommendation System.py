
# # K-Drama Recommendation System from Top 100 K-Dramas

#Modules for EDA
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

#Modules for ML(Recommendation)
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

get_ipython().run_line_magic('matplotlib', 'inline')

# Importing dataset
kd = pd.read_csv('top100_kdrama.csv')

# Understanding the data
kd.info()

kd.describe()

kd.head()

kdrama_names = kd[['Name']]
kdrama_names.head()

cols_for_recommend = ['Year of release', 'Number of Episode', 'Network', 'Duration', 'Content Rating', 'Rating']
kd = kd[cols_for_recommend]
kd.head()

# # Feature Engineering
# # Removing duplicate values in Network column
networks = []
[networks.append(list(set(network.replace(' ','').split(',')))[0]) for network in kd['Network']]
networks[:5]

kd['Network'] = networks
kd['Network'].unique()

# # Network and Total K-Dramas
plt.figure(figsize=(7,7))

kd['Network'].value_counts().plot(kind='barh')

plt.gca().invert_yaxis()
plt.title("Networks of K-Dramas")
plt.xlabel('Frequency')
plt.show()

kd['Network'].value_counts()

kd['Duration'] = kd['Duration'].str.replace('[A-Za-z]\D+','',regex=True)
kd['Duration'].head()

kd['Duration'] = kd['Duration'].str.replace(' ','',regex=True)
kd['Duration'] = pd.to_numeric(kd['Duration'])
kd['Duration'].head()

plt.figure(figsize=(7,7))
sns.histplot(data=kd['Duration'])
plt.title('Duration in minutes')
plt.show()

plt.figure(figsize=(7,7))
kd['Content Rating'].value_counts().plot(kind='pie',autopct='%.2f%%')
plt.title("Content Rating")
plt.show()

kd['Content Rating'].value_counts()

# # Rating and Content Rating
sns.histplot(data=kd[['Rating','Content Rating']],x='Rating',hue='Content Rating')
plt.show()

kd[['Rating']].describe()

# # Number of K-Dramas released in a year
plt.figure(figsize=(7,7))

kd['Year of release'].value_counts().plot(kind='barh')

plt.gca().invert_yaxis()
plt.title("Number of K-Dramas released per Year")
plt.xlabel('Frequency')
plt.show()

kd['Year of release'].value_counts()

# # Number of Episodes Distribution
plt.figure(figsize=(7,7))

kd['Number of Episode'].value_counts().plot(kind='barh')

plt.gca().invert_yaxis()
plt.title("Number of Episodes in K-Dramas")
plt.xlabel('Frequency')
plt.show()

kd['Number of Episode'].value_counts()

# # One Hot Encoding
kd.head()

cols_to_encode = ['Network','Content Rating']
dummies = pd.get_dummies(kd[cols_to_encode],drop_first=True)
dummies.head()

kd.drop(cols_to_encode, axis=1,inplace=True)
kd.head()

# # Feature Scaling
scale = MinMaxScaler()
scalled = scale.fit_transform(kd)

i=0
for col in kd.columns:
    kd[col] = scalled[:,i]
    i += 1

kd.head()

new_kd = pd.concat([kd, dummies],axis=1)
new_kd.head()

synopsis = pd.read_csv('top100_kdrama.csv',usecols=['Synopsis'])
synopsis.head()

kdrama_names['Name'].loc[23]='kingdom'
new_kd.index = [drama for drama in kdrama_names['Name']]
synopsis.index = [drama for drama in kdrama_names['Name']]
new_kd.head()

def getRecommendation_dramas_for(drama_name,no_of_recommend=5,get_similarity_rate=False):
    
    kn = NearestNeighbors(n_neighbors=no_of_recommend+1,metric='manhattan')
    kn.fit(new_kd)
    
    distances, indices = kn.kneighbors(new_kd.loc[drama_name])
    
    print(f'Similar K-Dramas for "{drama_name[0]}":')
    nearest_dramas = [kdrama_names.loc[i][0] for i in indices.flatten()][1:]
    if not get_similarity_rate:
        return nearest_dramas
    sim_rates = []
    synopsis_ = []
    for drama in nearest_dramas:
        synopsis_.append(synopsis.loc[drama][0])
        sim = cosine_similarity(new_kd.loc[drama_name],[new_kd.loc[drama]]).flatten()
        sim_rates.append(sim[0])
    recommended_dramas = pd.DataFrame({'Recommended Drama':nearest_dramas,'Similarity':sim_rates,'Synopsis':synopsis_})
    recommended_dramas.sort_values(by='Similarity',ascending=True)
    return recommended_dramas

def print_similiar_drama_Synopsis(recommended_kd):
    rkd = recommended_kd
    rkd_cols = rkd['Synopsis']
    dramas = rkd['Recommended Drama']
    for i in range(5):
        print(dramas[i])
        print(rkd_cols[i])
        print('\n')


# # Predicting Drama Recommendation
rd1 = kdrama_names.loc[0]
rd1
getRecommendation_dramas_for(rd1,no_of_recommend=5)

rd2 = kdrama_names.loc[10]
rd2
getRecommendation_dramas_for(rd2,get_similarity_rate=True)

rd3 = kdrama_names.loc[1]
rd3
getRecommendation_dramas_for(rd3,no_of_recommend=10,get_similarity_rate=True)

rd4 = kdrama_names.loc[8]
rd4
rdf4 = getRecommendation_dramas_for(rd4,no_of_recommend=10,get_similarity_rate=True)
print_similiar_drama_Synopsis(rdf4)

rd5 = kdrama_names.loc[99]
rd5
rdf5=getRecommendation_dramas_for(rd5,no_of_recommend=5,get_similarity_rate=True)
print_similiar_drama_Synopsis(rdf5)
