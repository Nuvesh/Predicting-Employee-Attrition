import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle

# Set display options for Pandas
pd.options.display.max_columns = None

# Load and shuffle the dataset
attrition = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
attrition = shuffle(attrition)

# Separate the data based on the 'Attrition' column
attrition_no = attrition[attrition['Attrition'] == 'No'].iloc[:550]
attrition_yes = attrition[attrition['Attrition'] == 'Yes']
attrition = pd.concat([attrition_yes, attrition_no])

# Check for null values
print(attrition.isnull().any())

# Create subplots
f, axes = plt.subplots(5, 3, figsize=(30, 50), sharex=False, sharey=False)

# Set color palette
color_start = 0.0
cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)
axes_x = 0
axes_y = 0

def plot_xy(x_label, y_label):
    global color_start, cmap, axes, axes_x, axes_y
    x = attrition[x_label].values
    y = attrition[y_label].values
    sns.kdeplot(x=x, y=y, cmap=cmap, fill=True, ax=axes[axes_x, axes_y])
    axes[axes_x, axes_y].set(title='{} vs. {}'.format(x_label, y_label))
    axes_y = axes_y + 1
    if axes_y == 3:
        axes_y = 0
        axes_x = axes_x + 1
    color_start = color_start + 0.33
    cmap = sns.cubehelix_palette(start=color_start, light=1, as_cmap=True)

# Plot multiple KDE plots
plot_xy('Age', 'TotalWorkingYears')
plot_xy('Age', 'DailyRate')
plot_xy('YearsInCurrentRole', 'Age')
plot_xy('DailyRate', 'DistanceFromHome')
plot_xy('DailyRate', 'JobSatisfaction')
plot_xy('YearsAtCompany', 'JobSatisfaction')
plot_xy('YearsAtCompany', 'DailyRate')
plot_xy('RelationshipSatisfaction', 'YearsWithCurrManager')
plot_xy('WorkLifeBalance', 'JobSatisfaction')
plot_xy('Age', 'JobLevel')
plot_xy('Age', 'MonthlyIncome')
plot_xy('MonthlyIncome', 'JobLevel')
plot_xy('NumCompaniesWorked', 'MonthlyIncome')
plot_xy('StockOptionLevel', 'EmployeeNumber')

# Show the plots
plt.tight_layout()
plt.show()

# Convert attrition to numerical values
target_map = {'Yes': 1, 'No': 0}
attrition['Attrition_num'] = attrition['Attrition'].apply(lambda x: target_map[x])

# Plot countplot
plt.figure(figsize=(10, 6))
sns.countplot(x='Attrition', data=attrition)
plt.show()

# Remove non-numerical columns and get the correlation matrix
attrition_numerical_cols = attrition._get_numeric_data()
corr = attrition_numerical_cols.corr()

# Graph heatmap to visualize correlations between data
fig, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(240, 5, as_cmap=True)
sns.heatmap(corr, cmap=cmap, square=True, ax=ax)
plt.show()

# Define numerical columns and plot pairplot to visualize correlations
numerical = ['Age', 'DailyRate', 'JobSatisfaction', 'JobLevel', 'MonthlyIncome', 'PerformanceRating', 'WorkLifeBalance', 'YearsAtCompany', 'Attrition_num']
g = sns.pairplot(attrition[numerical], hue='Attrition_num')
g.set(xticklabels=[])
plt.show()

# Get categorical data and perform one-hot encoding
categorical = attrition.select_dtypes(include=['object'])
numerical = attrition_numerical_cols.drop('Attrition_num', axis=1)
categorical = pd.get_dummies(categorical)

# Concatenate numerical and categorical data
input_data = pd.concat([numerical, categorical], axis=1)

# Normalize the data
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
np_scaled = min_max_scaler.fit_transform(input_data)
input_data_normalized = pd.DataFrame(np_scaled, columns=input_data.columns.values)

# Define training and test data
input_data_labels = attrition['Attrition_num']
input_data_nums = input_data_normalized

training_data = input_data_nums[:600]
training_labels = input_data_labels[:600]
test_data = input_data_nums[600:]
test_labels = input_data_labels[600:]

# Split data into batches and print each batch
training_data_batches = np.array_split(training_data, 10)
training_labels_batches = np.array_split(training_labels, 10)

for (data_batch, label_batch) in zip(training_data_batches, training_labels_batches):
    for (data, label) in zip(data_batch.values, label_batch.values):
        print(data, label)
        
