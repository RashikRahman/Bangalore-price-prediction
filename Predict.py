import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


data = pd.read_csv('Bengaluru_House_Data.csv')
# print(data.groupby('area_type').area_type.agg('count')) #we grouped data by area_type and see the area_type counts .
data = data.drop(['area_type', 'society', 'balcony', 'availability'], axis=1) #dropping these columns from the dataset.
# print(data.isnull().sum()) #returns the sum of null values in each column that is returns the total number of empty cells in each columns

data = data.dropna() # We area dropping the rows which has null values. i.e bath column has 73 empty cell so we remove those 73 rows entirely as our data has 13k rows. If you dont want to remomve rows you can fill in those empty cells by the mean of the columns
# print(data.isnull().sum()) # now we again check and there's no empty cell in our data
# print(data['size'].unique()) #generally all the values are in BHK form but some are 1bedroom or 1RK form. There are total 13k  values in this row so we pick the unique values. like there maybe 300 2bhk values but it only returns 2BHK as a representative of all the 2BHK values

# now wee see that ['2 BHK' '4 Bedroom' '3 BHK' '4 BHK' '6 Bedroom' '3 Bedroom' '1 BHK'
#  '1 RK' '1 Bedroom' '8 Bedroom' '2 Bedroom' '7 Bedroom' '5 BHK' '7 BHK'
#  '6 BHK' '5 Bedroom' '11 BHK' '9 BHK' '9 Bedroom' '27 BHK' '10 Bedroom'
#  '11 Bedroom' '10 BHK' '19 BHK' '16 BHK' '43 Bedroom' '14 BHK' '8 BHK'
#  '12 Bedroom' '13 BHK' '18 Bedroom']
# the size contains these values. We need the 2 from the 2BHK and this applies to all the others. to do this we create another column named bhk and use a lambda function where x contains 2BHK and it splits and returns the 0 index that is 2 in int form.
# to apply this lambda function to the dataframe we use the apply() method

data['bhk'] = data['size'].apply(lambda x: int(x.split(' ')[0]))

# print(data[data.bhk>20]) #shows everything where bhk is greater than 20
#1718  2Electronic City Phase II      27 BHK       8000  27.0  230.0   27
#4684                Munnekollal  43 Bedroom       2400  40.0  660.0   43
#the print of data where bhk >20 shows these result here we can see 43 bedroom flat's square feet is 2400 which is an error cause it's impossible in real life scenario. to handle this error we explore the total_sqft column



'''############################################################## START cleaning and structuring data##############################################################'''

'''<-------------------------------------------------------------exploring total_sqft------------------------------------------------------------->'''
# print(data.total_sqft.unique()) # returns ['1056' '2600' '1440' ... '1133 - 1384' '774' '4689'] this . here we can see 1133 - 1384 this kind of value which is not a single value rather than it's a range. we'll convert it to single value.

def is_float(x):
    try:
        float(x)
        return True
    except :
        return False

# print(data['total_sqft'][~data['total_sqft'].apply(is_float)].unique())  #we use ~ to see only the values where the func returns false. it returns these 2100 - 2850, 34.46Sq. Meter, 4125Perch,5.31Acres, 1574Sq. Yards, 1Grounds and much more. So it's not uniformed and unstructured. It has outliers data errors loads of problems
# so we gonna clean the data where its a range like 1070 - 1315 this and we clean it by taking avg of it and for other formation we ignore them in calculation
#now we rite a function to clean the data

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens)==2:  #1133 - 1384 for this format returns average
        return (float(tokens[0])+float(tokens[1]))/2
    try: #1056 for this format returns the float value
        return float(x)
    except: #for all the other formats returns none
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
# print(data.loc[410])
# print(data.isnull().sum()) # there are total 46 empty cell in total_sqft
data.dropna(inplace=True)# we are dropping those empty cell
# print(data.isnull().sum()) #again check and didn't find any epmty cell

data['price_per_sqft'] = data['price']*100000/data['total_sqft'] #in real estate price per sqft is very imp and this feature will help us to do some outlier cleaning in later stage. so i am feature enggnring here and creating a new feature which can be helpful for outlier detection and removal in later stage. price*100000 cause the price is in lakh
# print(data)



'''<-------------------------------------------------------------exploring location------------------------------------------------------------->'''
data.location = data.location.apply(lambda x: x.strip()) # removing any trailing and ending spaces in each cell value of location column

# print(len (data.location.unique()))# see the location. There are total 1287 locations

#usually to handle the text data  we convert it into dummy columns using one hot encoding. And if we keep all the locations whats gonna happens is we gonna have 1287 coumns in our dataframe
#which is just too much , too many features. This is called dimensionality curse or a high dimensionality problem. There are techniques to reduce the dimension.
# one of the very effective technique is to come up with this other category. Other category means when you have 1287 locations you'll find that there will be
# many locations which wil have only 1 or 2 data points.

location_stat = data.groupby('location')['location'].agg('count').sort_values(ascending=False) # seeing how many locations have only 1 or 2 data points
# print(location_stat) # i.e  Whitefield has 533 data points Annasandrapalya  has 1 data point
# print(len(location_stat[location_stat<=10])) #since location stat is a series so we can directly apply this type of condition on it. We do this to check how many location have less than 10 data points
#out of 1287  there are 1047 locations which has less than 10 datapoints
location_stat_less_than_10 = location_stat[location_stat<=10]
# print(location_stat_less_than_10)
data.location = data.location.apply(lambda x : 'other' if x in location_stat_less_than_10 else x) # here we transform the column with a condition that in the location column the cell value will be other if the location is in location_stat_less_than_10 series else the cell value will contain x that is the location itself.
# print(len(data.location.unique())) # now we have 241 unique locations instead of 1287 locations this is pretty good because when i later on convert this into one hot encoding i'll only have to handle 241 columns. We transformed this column as we didnt want to lose 1045 locationpoints who has less than 10 datapoints.





'''<-------------------------------------------------------------Outlier detection and removal------------------------------------------------------------->'''

#outliers are the datapoints which are data errors or sometimes they are not data error but they just represent the extreme variation in the dataset
# although they are valid it makes sense to remove them otherwise they can create some issues later on. We can apply different techniques to detect the outliers
# and remove them. And these techniques are  either use like SD for example or we can use simple or domain knowledge. One of the things in realestate domain is that
# when you have lets say 2 bedroom apartment it can't be 500sqft in total area or have 43 Bedroom with 2400 sqft of total area. So we want to first look at data rows
# where the sqft per bedroom is less then some threshold. We goto the business manager and ask him what is the sqft per bedroom in avg and he will tell us it's around 300 minimum.
# so if you have nay case for example you have let's 2440/43=55sqft per bedroom, this seems very unusual as the manager says 300sqft per bedroom we take it as threshold
# using that criteria we'll examine our dataset and try to find out the properties where this threshold doesn't match

# print(data[(data.total_sqft/data.bhk)<300]) # here we see the datapoints which doesn't match the threshold. we take it to our manager to take him have a lookt at it and he says i.e. 6 Bedroom in 1020sqft, that is unusual and will tell us to remove that cause these are data outliers
# so we gonna remove them. But first lets see how many rows we have in our data frame which doesnt satisfy the threshold
# print(len(data[(data.total_sqft/data.bhk)<300])) # there are total 744 rows that doesnt match the threshold
# print(len(data))# we have 13200 total rows in the data set
#now let's drop them

data = data[~((data.total_sqft/data.bhk)<300)] # we ~ on the criteria cause the false ones will match the threshold . Removing outliers
# print(data.shape) #so after removing them we have 12456 rows in the dataset


#Now we can have more outliers for example price per sqft. So lets check that where the price per sqft is either very very high or very very low
# print(data.price_per_sqft.describe()) # This tells us about the properties of price_per_sqft. Here min is  267.8298 and max is 176470.588235
#in bangalore to get a property with 267Rs per sqft is very very rare and unlikely.and for max price its very high but its possible if the property is in very prime area
# but as we are making a generic model so it makes sense to remove this kind of extreme cases
# print(data.columns)

# So we gonna write a function that'll remove these extreme cases based on SD. i.e if the data has a normal distribution which we are assuming our dataset should have then most of the data points like around 68% data points should lie between mean and +-1SD
# so we gonna filter out anything which is beyond 1SD
# Now we gonna remove price_per_sqft outliers per location cause some location will have high price and some will have low price.so we'll remove outlier based on locations.So we need to find mean and SD per location and filter out datapoints which are beyond +-1SD


##########################pps outlier


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        # print(subdf)
        mn = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>mn-sd) & (subdf.price_per_sqft<mn+sd)]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

# print(remove_pps_outliers(data).shape) # after removing outliers we hace 10242 rows in the dataframe that is weremoved almost 2k outliers from our dataset
data =remove_pps_outliers(data)
# print(data.shape)

#now lets check if the property price for three bedroom is less than the property price of two bedroom where the size and location of the property is quite same
# in the data set we can see 3bedroom price is 81 lacks and 2bedroom price is 1.27crore. This maybe possible due to the 2bedroom property has premium features around it or belong to high class society butas we are building a generic model we cant overlook this matter.
# Let's see some visualization of how many such cases we have


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# plot_scatter_chart(data, 'Rajaji Nagar')
# plot_scatter_chart(data, 'Hebbal')

#from the visualization we can see that some location exists where there is still outlier we talked about above. Lets do some cleanup then
# this function will create per bedroom homes it will create some stats. The kind of stat it will create is like it will create a dictionary like this.
# What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.
#
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },
# }
# The function will work like if the price per sqft of of 2BHK is less than the mean price per sqft of 1BHK than that 2BHK is removed



##########################bhk outlier

def remove_bhk_outlier(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):#for every location_df we are creating new bhk_df
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }

        for bhk, bhk_df in location_df.groupby('bhk'): #again create bhk_df based on location and exclude indices based on bhk_stats
            stats = bhk_stats.get(bhk-1) #current BHK er ager BHK jabe stat e. i.e. current bhk 3 hoile 2bhk jabe stat e
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values) #current bhk er price_per_sqft jodi ager bhk er mean er choto hoy than sei index er value gula append koro

    return df.drop(exclude_indices,axis='index')



data = remove_bhk_outlier(data)
# plot_scatter_chart(data, 'Rajaji Nagar')
# plot_scatter_chart(data, 'Hebbal')
# from this plot we can see that bhk outliner is removed from these and all the other locations



'''
print(data.shape) #now we have only 7317 rows in the data
matplotlib.rcParams['figure.figsize'] = (20, 10)
plt.hist(data.price_per_sqft, rwidth=0.8) #rwidth is the width of the bath
plt.xlabel('Price Per Square Feet')
plt.ylabel('Counts')
plt.show() #how many properties i have based on price per sqft. By this we can see that from 0-10k price per sqft we have majority of our datapoints

matplotlib.rcParams['figure.figsize'] = (20, 10)
plt.hist(data.bath, rwidth=0.5) #rwidth is the width of the bath
plt.xlabel('Bathrooms')
plt.ylabel('Counts')
plt.show()
''' # there are 2 bathrooms in majority of the data point. lets say we have 4 bathroom for a 2bedroom apartment. we goto the business manager and see if there's any criteria for this like if this is unusual and if so should we remove it.

##########################bathroom outlier
#the business managers tells us that anytime we have number of bathroom which is  greater than equals number of bedroom+2 its unusual and remove these.
# print(data[data.bath>data.bhk+2]) #here we check and get some outliers
data = data[data.bath<data.bhk+2] #removing theoutliers
# print(data.shape) #now  we have 7239 rows in our dataset

#now our model is pretty much cleaned now we can prepare it for modeling or ML training and for that we have to drop some unnecessary columns
#sizecan be droppedas we have bhk column and price per sqft can also be dropped as we only used it for outlier detection
data = data.drop(['size', 'price_per_sqft'], axis='columns')
# print(data.shape)

'''############################################################## END cleaning and structuring data##############################################################'''












'''############################################################## START ML modeling ##############################################################'''

#we're gonna build a ML model and then use k-fold cross validation and GridSearchCV to come up with the best algorithm and parameters
#ML can't interpret text data so we have to convert locations into numeric column. To do this we will use one hot encoding it is laso called dummies so we gonna use pandas dummies method

dummies = (pd.get_dummies(data.location)) #created dummy variable
data = pd.concat([data, dummies.drop('other',axis='columns')], axis='columns') #concatinated with the data and dropped a dummy column to avoid the dummy variable trap issue.To avoid this issue you need to only drop only one column at random here we select to drop other column to drop
data = data.drop('location',axis='columns') # as we have encoded location no need to have location column
# print(data.shape)

#lets build ML

X = data.drop('price',axis='columns') # X should only contain independent variables. Here price is a dependent variable cause we gonna predict it depending on the independent variables
Y = data.price # Y should only contain dependent variable. here its price
# print(Y)

#we always divide our dataset in to two sets training and test. Training is to train the model and test is to evaluate the model

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10) #test_size=0.2 means we want 20% of the data to be test sample and 80% of the data to be model training samples. random_state=10 cause it will select same training data everytime rather then selecting randomly


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression() #make linearregression model , lr_clf is linear regression classifier
lr_clf.fit(X_train,Y_train) #train them
# print(lr_clf.score(X_test,Y_test)) #test them and our score is 86.29% but we can further improve this by using some algorithm

'''
#typically a data scientist will try couple models with couple of different parameters to come up with the best optimal model. Now we gonna try to comeup with an optimal model
# we gonna use k-fold cross validation for this. cross validation is a technique to choose the most optimal model. todo tutorial : https://www.youtube.com/watch?v=gJo0uNL-5Qw

from sklearn.model_selection import ShuffleSplit, cross_val_score

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=10) #cv is cross validation
print((cross_val_score(LinearRegression(), X, Y, cv=cv)).mean()) #we are getting scoe 84.37% but how about trying other linear regression techniques like lasso regression, decision tree regression etc
# There are various regression technique available so as a data scientist we want totry those different algorithms and figure out which one gives me the best score
# for that we use a method called grid search cv.

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,Y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {  #we also supply parameters with these algo to do hyper parameter tuning
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, Y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])



print(find_best_model_using_gridsearchcv(X, Y)) #from this wecan see that linearregression is best among all. so we gonna select that lr_clf
'''

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0] # [0] index e ekta list ase so oitar [0] index e column value ta ase. That what we need
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

# print(predict_price('1st Phase JP Nagar',8000, 3, 4))
# print(predict_price('Indira Nagar',8000, 3, 4))

import pickle #to export our model
with open('bangalore_home_price_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

import json
columns = {  #to predict we need the list of columns
    'data columns': [col.lower() for col in X.columns]
}

with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))