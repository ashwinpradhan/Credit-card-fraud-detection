import pandas as pd
import seaborn as sns

dataset = pd.read_csv("creditcard.csv")

x = dataset.iloc[:, :-1]
y = dataset["Class"]

#visualizing the data
sns.countplot(x = "Class", data = dataset)

fraud = dataset[dataset["Class"] == 1]
normal = dataset[dataset["Class"] == 0]


#balancing the imbalanced dataset.
from imblearn.over_sampling import RandomOverSampler

os = RandomOverSampler(sampling_strategy = 1, random_state = 0)
x_reshape, y_reshape = os.fit_sample(x, y)


#spliting the dataset into train and test set.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_reshape, y_reshape, 
                                                    test_size = 0.3, random_state = 0)


from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression(random_state = 0)
regressor.fit(x_train, y_train)

#predicting the froud.
y_pred = regressor.predict(x_test)


#checking the accuracy.
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

print(f"accuracy of the model is :- {round(accuracy_score(y_test, y_pred) * 100)}%")
