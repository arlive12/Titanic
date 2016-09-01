# Titanic

Here we are analyzing the Titanic data which Contains demographics and passenger information from 891 of the 2224 passengers and crew on board the Titanic. This data includes the following attributes of the passenger:
1. PassengerID - unique id of the passenger
2. Survived - do the passenger survived the Titanic shipwreck?                        0-survived
                                                                                      1-not survived

3. Pclass -  Passenger Class (it is a proxy for socio economic status)                1-first class
                                                                                      2-second class
                                                                                      3-third class

4. Name 
5. Sex 
6. Age
7. sibsp - Number of sibling/spouses aboard
8. parch - Number of parents/children aboard
9. ticket - Ticket number
10.fare - Passenger fare
11.cabin
12.embarked                                                                            C = Cherbourg
                                                                                       Q = Queenstown
                                                                                       S = Southampton
Through this analysis we will try to answer the following question
     1. What factors made people more likely to survive?
     2. How the Fare differs for the people of different class?

Note: The results here are tentative as we will not use statistics or machine learning to prove our answer.
Data Wrangling Phase
In [89]:
%pylab inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
titanic_data = pd.read_csv('/Users/ar50645/desktop/titanic_data.csv',index_col = 'PassengerId')
Populating the interactive namespace from numpy and matplotlib
In [90]:
def checkMissingValues(field):
    return np.count_nonzero(field.isnull().values)

print titanic_data.apply(checkMissingValues)
cleaned_data_by_age =titanic_data[np.isfinite(titanic_data['Age'])]
Survived      0
Pclass        0
Name          0
Sex           0
Age         177
SibSp         0
Parch         0
Ticket        0
Fare          0
Cabin       687
Embarked      2
dtype: int64
Data acquisition: We had acquired data through https://www.kaggle.com/c/titanic/data.
Data Cleaning: From the above code , We can say that the variable which had missing values are as follows:
                     1. Age which has 177 missing values
                     2. Cabin which has 686 missing values
                     3. Embarked which has 2 missing values
We have created new dataframe named cleaned_data_by_age for the passenger which has valid age values. Missing Values of Cabin and Embarked will not impact our analysis.In our analysis, we will use both raw data and cleaned data in our analysis.
Exploration Phase:
In [91]:
print titanic_data['Survived'].value_counts()
0    549
1    342
Name: Survived, dtype: int64
Note that in our sample 38.38% of the passenger survived
Histogram of Age
In [92]:
titanic_data.Age.hist(bins = 70)
plt.title('Distribution of Age')
plt.xlabel('Ages')
plt.ylabel('Passenger Counts')
Out[92]:
<matplotlib.text.Text at 0x11d97c2d0>

We can observe that most of the people onboard was of the age somewhere between 17 to 38.
Histogram of Age by Survival
In [93]:
titanic_data.groupby('Survived')['Age'].get_group(0).hist(label = 'Died', bins = 70)
titanic_data.groupby('Survived')['Age'].get_group(1).hist(label = 'Survived', bins = 70)
plt.legend()
plt.title('Histogram of Age\nby\nSurvival')
plt.xlabel('Age')
plt.ylabel('Passengers')
Out[93]:
<matplotlib.text.Text at 0x120363f10>

Histogram of Fare
In [94]:
titanic_data['Fare'].hist(bins=10)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Passenger Counts')
Out[94]:
<matplotlib.text.Text at 0x12064e510>

Here we can observe that most of the population spends less than 50 bucks as a fare.
In [95]:
group_by_Pclass = titanic_data.groupby('Pclass').mean()
print group_by_Pclass
plot_group_by_Pclass= group_by_Pclass['Fare'].plot(kind='bar',title="Fare vs Pclass")
plot_group_by_Pclass.set_ylabel("Fare")
        Survived        Age     SibSp     Parch       Fare
Pclass                                                    
1       0.629630  38.233441  0.416667  0.356481  84.154687
2       0.472826  29.877630  0.402174  0.380435  20.662183
3       0.242363  25.140620  0.615071  0.393075  13.675550
Out[95]:
<matplotlib.text.Text at 0x120b9a610>

We can observe that fare for the first class passenger is the highest follwed by second class passenger followed by third class passenger. This can be understable as the first class passenger enjoys many luxury in ships.
In [96]:
group_by_Sex_and_Pclass = titanic_data.groupby(['Sex','Pclass']).mean()
print group_by_Sex_and_Pclass
plot_group_by_Sex_and_Pclass= group_by_Sex_and_Pclass["Survived"].plot(title= "Survival vs Sex,PClass")
plot_group_by_Sex_and_Pclass.set_ylabel("Survival")
               Survived        Age     SibSp     Parch        Fare
Sex    Pclass                                                     
female 1       0.968085  34.611765  0.553191  0.457447  106.125798
       2       0.921053  28.722973  0.486842  0.605263   21.970121
       3       0.500000  21.750000  0.895833  0.798611   16.118810
male   1       0.368852  41.281386  0.311475  0.278689   67.226127
       2       0.157407  30.740707  0.342593  0.222222   19.741782
       3       0.135447  26.507589  0.498559  0.224784   12.661633
Out[96]:
<matplotlib.text.Text at 0x1209e4b10>

By the above data, we can conclude that chance for survival can be arranged as follows: first class female > second class female > third class female > first class male > second class male > third class male We can see that the factors made people more likely to survive are : sex and Pclass.
Now we analyse the effect of age on the survival. Here we will use the cleaned data.
In [97]:
#function take columns of the dataframe and converts each column in its standard form

def standarize(col):
    return (col-col.mean())/col.std(ddof=0)
standarized_value_survived = standarize(cleaned_data_by_age['Survived'])
standarized_value_age = standarize(cleaned_data_by_age['Age'])
pearson_r = (standarized_value_survived  * standarized_value_age).mean()
print pearson_r
-0.0772210945722
We can see that the pearson's r is negative . This means age will be negatively related to the probability of survival.
Thus,we can finally conclude the following from our analysis:
The factors that made people more likely to survive are Gender and class of the passenger.
First class people pays a lot higher for ticket than second class followed by third class people.
Note: one change causes another based solely on a correlation.All the findings are tentative as we are not using any statistical methods to prove it. By not using statistical methods, we cannot be certain of our findings as we cannot state the confidence without t-tests.We also have 177 missing values of the ages of the passenger, so accurateness of the calculation of the pearson'r above also affected.Another possible explanatory variable is whether or not a passenger got on a lifeboat or not. This seems to be a significant determinant of survival. In addition, one could control for the cabin of the passenger. It would also be interesting to investigate why upper class men had a lower probability of survival than lower class women. It might also be interesting to have a separate category for children. However, from the analysis that was performed, a person was lucky to survive and had the best chance of survival if they were a young, upper class woman.
In [ ]:
 
