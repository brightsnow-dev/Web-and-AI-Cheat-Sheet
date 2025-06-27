import streamlit as st
import base64
import requests

# Initial page config
st.set_page_config(
    page_title='Web+AI Personal Cheat Sheet',
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    ds_sidebar()
    ds_body()

# Function to convert image to base64 bytes (for logo)
def img_to_bytes(img_url):
    try:
        response = requests.get(img_url)
        img_bytes = response.content
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except:
        return ''

# Sidebar content
def ds_sidebar():
    # logo_url = 'https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png'
    # logo_encoded = img_to_bytes(logo_url)
    
    # st.sidebar.markdown(
    #     f"""
    #     <a href="https://ahammadmejbah.com/">
    #         <img src='data:image/png;base64,{logo_encoded}' class='img-fluid' width=100>
    #     </a>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.sidebar.header('🧰 Web+AI Personal Cheat Sheet')

    st.sidebar.markdown('''
    <small>Comprehensive summary of essential Web Dev + Data Science concepts, libraries, and tools.</small>
    ''', unsafe_allow_html=True)

    st.sidebar.markdown('__🔑 Key Libraries__')
    st.sidebar.code('''
$ pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pytorch nltk spacy sqlalchemy airflow boto3
    ''')

    st.sidebar.markdown('__💻 Common Commands__')
    st.sidebar.code('''
$ jupyter notebook
$ python script.py
$ git clone https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet
$ streamlit run app.py
    ''')

    st.sidebar.markdown('__🔄 Data Science Workflow__')
    st.sidebar.code('''
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Building
6. Evaluation
7. Deployment
    ''')

    st.sidebar.markdown('__💡 Tips & Tricks__')
    st.sidebar.code('''
- Use virtual environments
- Version control with Git
- Document your code
- Continuous learning
- Utilize Jupyter Notebooks for exploration
    ''')

    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Data Science Cheat Sheet v1.0](https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet) | Nov 2024 | [Shemanto Sharkar](https://ahammadmejbah.com/)<div class="card-footer">Shemanto Sharkar © 2024</div></small>''', unsafe_allow_html=True)

# Main body of cheat sheet
def ds_body():
    # Header
    st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #FF4B4B;">🚩 Web Development + Data Science Cheat Sheet By Shemanto Sharkar</h1>
        </div>
    """, unsafe_allow_html=True)

    # Define main categories and their subtopics
    sections = {
        "🐍 Python": {
            "Importing Libraries": '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import requests
import json
import re
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from pathlib import Path
            ''',
            "Data Structures": '''
# List
my_list = [1, 2, 3, 4]

# Tuple
my_tuple = (1, 2, 3, 4)

# Dictionary
my_dict = {'key1': 'value1', 'key2': 'value2'}

# Set
my_set = {1, 2, 3, 4}

# List Comprehension
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary Comprehension
square_dict = {x: x**2 for x in range(10)}
            ''',
            "Control Flow": '''
# If-Else
if condition:
    # do something
elif another_condition:
    # do something else
else:
    # default action

# For Loop
for i in range(10):
    print(i)

# While Loop
while condition:
    # do something
    break
            ''',
            "Functions": '''
def my_function(param1, param2):
    """
    Function description.
    """
    result = param1 + param2
    return result

# Lambda Function
add = lambda x, y: x + y
print(add(5, 3))

# Decorators
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
            ''',
            "Exception Handling": '''
try:
    # code that may raise an exception
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Execution complete.")
            ''',
            "File I/O": '''
# Reading a file
with open('file.txt', 'r') as file:
    data = file.read()

# Writing to a file
with open('file.txt', 'w') as file:
    file.write('Hello, World!')

# Reading CSV with Pandas
df = pd.read_csv('data.csv')

# Writing DataFrame to CSV
df.to_csv('output.csv', index=False)
            ''',
            "Modules and Packages": '''
# Importing a module
import math

# Using a function from a module
result = math.sqrt(16)
print(result)

# Creating a package
# Directory structure:
# mypackage/
#     __init__.py
#     module1.py
#     module2.py

# Importing from a package
from mypackage import module1, module2

# Using functions from modules
module1.function_a()
module2.function_b()
            '''
        },
        "📁 Data Manipulation": {
            "Pandas Basics": '''
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
})

# Read CSV
df = pd.read_csv('data.csv')

# View DataFrame
df.head()

# Information about DataFrame
df.info()

# Summary statistics
df.describe()
            ''',
            "Data Selection": '''
# Select column
df['Age']

# Select multiple columns
df[['Name', 'Age']]

# Select rows by index
df.iloc[0:5]

# Select rows by condition
df[df['Age'] > 30]

# Select rows using loc
df.loc[df['City'] == 'New York']

# Select rows using iloc
df.iloc[[0, 2, 4]]
            ''',
            "Data Cleaning": '''
# Handle missing values
df.dropna(inplace=True)
df.fillna(value=0, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Data type conversion
df['Age'] = df['Age'].astype(int)

# Rename columns
df.rename(columns={'Name': 'Full Name'}, inplace=True)

# Replace values
df['City'].replace({'New York': 'NY', 'Los Angeles': 'LA'}, inplace=True)

# Filtering out outliers
df = df[df['Salary'] < df['Salary'].quantile(0.95)]
            ''',
            "Data Transformation": '''
# Apply function
df['Age'] = df['Age'].apply(lambda x: x + 1)

# Vectorized operations
df['Age'] = df['Age'] + 1

# Mapping
df['City'] = df['City'].map({'New York': 'NY', 'Los Angeles': 'LA', 'Chicago': 'CHI'})

# Binning
df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])

# Creating new columns
df['Salary_Per_Age'] = df['Salary'] / df['Age']

# String operations
df['Name'] = df['Name'].str.upper()
            ''',
            "Merging & Joining": '''
# Merge DataFrames
merged_df = pd.merge(df1, df2, on='Key', how='inner')

# Concatenate DataFrames
concatenated_df = pd.concat([df1, df2], axis=0)

# Join DataFrames
joined_df = df1.join(df2, how='inner')

# Merge on multiple keys
merged_df = pd.merge(df1, df2, on=['Key1', 'Key2'], how='outer')

# Merge with indicator
merged_df = pd.merge(df1, df2, on='Key', how='outer', indicator=True)
            ''',
            "Grouping & Aggregation": '''
# Group by
grouped = df.groupby('City')

# Aggregation
grouped['Age'].mean()

# Multiple aggregations
grouped.agg({'Age': ['mean', 'sum'], 'Salary': 'median'})

# Group by with multiple columns
grouped = df.groupby(['City', 'Age Group'])

# Aggregation with custom functions
grouped.agg({
    'Salary': ['mean', 'sum'],
    'Experience': lambda x: x.max() - x.min()
})
            ''',
            "Pivot Tables": '''
# Create pivot table
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', fill_value=0)

# Multiple aggregation functions
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc=['sum', 'mean'], fill_value=0)

# Adding margins
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', margins=True, fill_value=0)
            '''
        },
        "📈 Data Visualization": {
            "Matplotlib": '''
import matplotlib.pyplot as plt

# Line Plot
plt.figure(figsize=(10,5))
plt.plot(x, y, label='Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.legend()
plt.grid(True)
plt.show()

# Bar Chart
plt.figure(figsize=(10,5))
plt.bar(categories, values, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Scatter Plot
plt.figure(figsize=(10,5))
plt.scatter(x, y, color='red')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()

# Histogram
plt.figure(figsize=(10,5))
plt.hist(data, bins=10, color='green', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Pie Chart
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart')
plt.axis('equal')
plt.show()
            ''',
            "Seaborn": '''
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter Plot with Regression Line
sns.lmplot(x='Age', y='Salary', data=df, aspect=1.5)
plt.title('Age vs Salary with Regression Line')
plt.show()

# Heatmap
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='City', y='Salary', data=df)
plt.title('Salary Distribution by City')
plt.show()

# Pairplot
sns.pairplot(df, hue='City')
plt.show()

# Violin Plot
plt.figure(figsize=(10,6))
sns.violinplot(x='City', y='Salary', data=df)
plt.title('Salary Distribution by City')
plt.show()
            ''',
            "Plotly": '''
import plotly.express as px

# Scatter Plot
fig = px.scatter(df, x='Age', y='Salary', color='City', title='Age vs Salary by City')
fig.show()

# Bar Chart
fig = px.bar(df, x='City', y='Sales', color='City', barmode='group', title='Sales by City')
fig.show()

# Line Chart
fig = px.line(df, x='Date', y='Sales', title='Sales Over Time')
fig.show()

# Histogram
fig = px.histogram(df, x='Age', nbins=10, title='Age Distribution')
fig.show()

# Pie Chart
fig = px.pie(df, names='Product', values='Sales', title='Sales Distribution by Product')
fig.show()
            ''',
            "Altair": '''
import altair as alt
import matplotlib.pyplot as plt

# Simple Line Chart
chart = alt.Chart(df).mark_line().encode(
    x='Date:T',
    y='Sales:Q'
).properties(
    title='Sales Over Time'
).interactive()
chart.display()

# Interactive Scatter Plot
chart = alt.Chart(df).mark_circle(size=60).encode(
    x='Age:Q',
    y='Salary:Q',
    color='City:N',
    tooltip=['Name', 'Age', 'Salary']
).interactive().properties(
    title='Age vs Salary by City'
)
chart.display()

# Bar Chart
chart = alt.Chart(df).mark_bar().encode(
    x='City:N',
    y='Sales:Q',
    color='City:N'
).properties(
    title='Sales by City'
).interactive()
chart.display()

# Heatmap
heatmap = alt.Chart(df).mark_rect().encode(
    x='City:N',
    y='Product:N',
    color='Sales:Q'
).properties(
    title='Sales Heatmap'
).interactive()
heatmap.display()

# Multi-line Chart
chart = alt.Chart(df).mark_line().encode(
    x='Date:T',
    y='Sales:Q',
    color='City:N'
).properties(
    title='Sales Over Time by City'
).interactive()
chart.display()
            ''',
            "Plotly Express Example": '''
# Interactive Scatter Plot
fig = px.scatter(df, x='Age', y='Salary', color='City', hover_data=['Name'], title='Interactive Age vs Salary')
st.plotly_chart(fig)

# Interactive Bar Chart
fig = px.bar(df, x='City', y='Sales', color='City', barmode='group', title='Interactive Sales by City')
st.plotly_chart(fig)

# Interactive Line Chart
fig = px.line(df, x='Date', y='Sales', title='Interactive Sales Over Time')
st.plotly_chart(fig)

# Interactive Histogram
fig = px.histogram(df, x='Age', nbins=10, title='Interactive Age Distribution')
st.plotly_chart(fig)

# Interactive Pie Chart
fig = px.pie(df, names='Product', values='Sales', title='Interactive Sales Distribution by Product')
st.plotly_chart(fig)
            '''
        },
        "🤖 Machine Learning": {
            "Scikit-learn Basics": '''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X = df[['Age', 'Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'MSE: {mse}, R2: {r2}')
            ''',
            "Classification Example": '''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize and train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(report)
            ''',
            "Cross-Validation": '''
from sklearn.model_selection import cross_val_score

# 5-Fold Cross-Validation
scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Scores: {scores}')
print(f'Average CV Score: {scores.mean()}')
            ''',
            "Hyperparameter Tuning with GridSearchCV": '''
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit GridSearch
grid_search.fit(X_train, y_train)

# Best parameters
print(grid_search.best_params_)

# Best score
print(grid_search.best_score_)
            ''',
            "Feature Scaling": '''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
            ''',
            "Handling Categorical Variables": '''
# One-Hot Encoding
X = pd.get_dummies(X, columns=['Category'])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Category'] = le.fit_transform(X['Category'])
            ''',
            "Model Persistence": '''
import joblib

# Save the model
joblib.dump(model, 'linear_regression_model.joblib')

# Load the model
loaded_model = joblib.load('linear_regression_model.joblib')
print(loaded_model.predict([[25, 5]]))
            '''
        },
        "🧠 Deep Learning": {
            "TensorFlow/Keras Basics": '''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
            ''',
            "PyTorch Basics": '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MyModel(input_dim=10, hidden_dim=64, output_dim=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoader
dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(100):
    for data, targets in loader:
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            ''',
            "Convolutional Neural Networks (CNN)": '''
from tensorflow.keras import layers, models

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channels)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
            ''',
            "Recurrent Neural Networks (RNN)": '''
from tensorflow.keras import layers, models

# Define RNN model
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.SimpleRNN(128, return_sequences=True),
    layers.SimpleRNN(128),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
            ''',
            "LSTM Networks": '''
from tensorflow.keras import layers, models

# Define LSTM model
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
            ''',
            "Model Saving and Loading": '''
# Saving the model
model.save('my_model.h5')

# Loading the model
loaded_model = keras.models.load_model('my_model.h5')

# Using the loaded model for predictions
predictions = loaded_model.predict(X_test)
print(predictions)
            '''
        },
        "📊 Statistical Analysis": {
            "Descriptive Statistics": '''
# Summary statistics
df.describe()

# Mean, Median, Mode
df['Age'].mean()
df['Age'].median()
df['Age'].mode()

# Variance and Standard Deviation
df['Age'].var()
df['Age'].std()

# Quantiles
df['Age'].quantile([0.25, 0.5, 0.75])
            ''',
            "Probability Distributions": '''
import numpy as np
import matplotlib.pyplot as plt

# Normal Distribution
data = np.random.normal(loc=0, scale=1, size=1000)
plt.figure(figsize=(10,6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Normal Distribution')
plt.show()

# Binomial Distribution
data = np.random.binomial(n=10, p=0.5, size=1000)
plt.figure(figsize=(10,6))
plt.hist(data, bins=30, color='salmon', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Binomial Distribution')
plt.show()

# Poisson Distribution
data = np.random.poisson(lam=3, size=1000)
plt.figure(figsize=(10,6))
plt.hist(data, bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Poisson Distribution')
plt.show()
            ''',
            "Hypothesis Testing": '''
from scipy import stats

# T-Test
t_stat, p_val = stats.ttest_ind(group1, group2)
print(f'T-statistic: {t_stat}, P-value: {p_val}')

# Chi-Square Test
chi2, p, dof, ex = stats.chi2_contingency(table)
print(f'Chi-square: {chi2}, P-value: {p}')

# ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f'F-statistic: {f_stat}, P-value: {p_val}')

# Mann-Whitney U Test
u_stat, p_val = stats.mannwhitneyu(group1, group2)
print(f'U-statistic: {u_stat}, P-value: {p_val}')

# Kruskal-Wallis Test
h_stat, p_val = stats.kruskal(group1, group2, group3)
print(f'H-statistic: {h_stat}, P-value: {p_val}')
            ''',
            "Correlation Analysis": '''
# Pearson Correlation
pearson_corr = df['A'].corr(df['B'])
print(f'Pearson Correlation: {pearson_corr}')

# Spearman Correlation
spearman_corr = df['A'].corr(df['B'], method='spearman')
print(f'Spearman Correlation: {spearman_corr}')

# Kendall Correlation
kendall_corr = df['A'].corr(df['B'], method='kendall')
print(f'Kendall Correlation: {kendall_corr}')
            ''',
            "Confidence Intervals": '''
import scipy.stats as st

# 95% Confidence Interval for the mean
confidence = 0.95
n = len(data)
mean = np.mean(data)
stderr = st.sem(data)
h = stderr * st.t.ppf((1 + confidence) / 2., n-1)
print(f'Confidence Interval: {mean-h} to {mean+h}')

# 99% Confidence Interval
confidence = 0.99
h = stderr * st.t.ppf((1 + confidence) / 2., n-1)
print(f'99% Confidence Interval: {mean-h} to {mean+h}')
            ''',
            "Regression Analysis": '''
import statsmodels.api as sm

# Define independent variables (add constant)
X = sm.add_constant(df[['Age', 'Experience']])
y = df['Salary']

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Predictions
predictions = model.predict(X)
print(predictions)
            ''',
            "Bayesian Statistics": '''
import pymc3 as pm
import numpy as np

# Sample data
data = df['Salary'].values
age = df['Age'].values

# Define the model
with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=(1,))
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value
    mu = alpha + beta[0] * age

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=data)

    # Inference
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)

# Summary
print(pm.summary(trace))
            '''
        },
        "🔧 Data Engineering": {
            "SQL Basics": '''
-- Select statement
SELECT column1, column2 FROM table_name;

-- Where clause
SELECT * FROM table_name WHERE condition;

-- Join
SELECT a.column1, b.column2
FROM table_a a
JOIN table_b b ON a.id = b.a_id;

-- Group By
SELECT column, COUNT(*)
FROM table
GROUP BY column;

-- Order By
SELECT * FROM table ORDER BY column DESC;

-- Inner Join
SELECT a.column1, b.column2
FROM table_a a
INNER JOIN table_b b ON a.id = b.a_id;

-- Left Join
SELECT a.column1, b.column2
FROM table_a a
LEFT JOIN table_b b ON a.id = b.a_id;

-- Right Join
SELECT a.column1, b.column2
FROM table_a a
RIGHT JOIN table_b b ON a.id = b.a_id;

-- Full Outer Join
SELECT a.column1, b.column2
FROM table_a a
FULL OUTER JOIN table_b b ON a.id = b.a_id;
            ''',
            "Database Connections with SQLAlchemy": '''
from sqlalchemy import create_engine
import pandas as pd

# Create engine
engine = create_engine('postgresql://user:password@localhost:5432/mydatabase')

# Read SQL query into DataFrame
df = pd.read_sql('SELECT * FROM table_name', engine)

# Write DataFrame to SQL
df.to_sql('table_name', engine, if_exists='replace', index=False)

# Execute a raw SQL query
with engine.connect() as connection:
    result = connection.execute("SELECT COUNT(*) FROM table_name")
    count = result.fetchone()[0]
    print(f'Total records: {count}')
            ''',
            "ETL Processes": '''
# Extract, Transform, Load (ETL) example using Pandas
import pandas as pd

# Extract
df = pd.read_csv('data.csv')

# Transform
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].apply(lambda x: x * 1.1)  # Increase salary by 10%
df['Name'] = df['Name'].str.title()

# Load
df.to_csv('clean_data.csv', index=False)
            ''',
            "Data Pipelines with Airflow": '''
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def extract():
    # Extraction logic
    pass

def transform():
    # Transformation logic
    pass

def load():
    # Loading logic
    pass

default_args = {
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('etl_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    extract_task = PythonOperator(task_id='extract', python_callable=extract)
    transform_task = PythonOperator(task_id='transform', python_callable=transform)
    load_task = PythonOperator(task_id='load', python_callable=load)

    extract_task >> transform_task >> load_task
            ''',
            "Data Warehousing with Redshift": '''
import psycopg2

# Connect to Redshift
conn = psycopg2.connect(
    dbname='dev',
    user='username',
    password='password',
    host='redshift-cluster.amazonaws.com',
    port='5439'
)

# Create cursor
cur = conn.cursor()

# Execute a query
cur.execute("SELECT * FROM sales_data LIMIT 10;")

# Fetch results
results = cur.fetchall()
print(results)

# Close connection
cur.close()
conn.close()

# Loading data into Redshift
import pandas as pd
from sqlalchemy import create_engine

# Create engine
engine = create_engine('postgresql://username:password@redshift-cluster.amazonaws.com:5439/dev')

# Load data
df = pd.read_csv('sales_data.csv')
df.to_sql('sales_data', engine, if_exists='append', index=False)
            ''',
            "Data Lakes with Hadoop": '''
# Install Hadoop
!apt-get update
!apt-get install -y openjdk-8-jdk-headless wget
!wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz
!tar -xzf hadoop-3.3.1.tar.gz
!mv hadoop-3.3.1 /usr/local/hadoop

# Set environment variables
import os
os.environ['HADOOP_HOME'] = '/usr/local/hadoop'
os.environ['PATH'] += ':/usr/local/hadoop/bin:/usr/local/hadoop/sbin'

# Start Hadoop services
!start-dfs.sh
!start-yarn.sh

# Create HDFS directories
!hdfs dfs -mkdir /data
!hdfs dfs -mkdir /data/raw
!hdfs dfs -mkdir /data/processed

# Upload data to HDFS
!hdfs dfs -put local_data.csv /data/raw/

# List HDFS directories
!hdfs dfs -ls /data/
            '''
        },
        "🛠 Tools & Utilities": {
            "Virtual Environments with venv": '''
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows
myenv\\Scripts\\activate
# On macOS/Linux
source myenv/bin/activate

# Deactivate
deactivate
            ''',
            "Package Management with pip": '''
# Install a package
pip install package_name

# List installed packages
pip list

# Freeze requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Upgrade a package
pip install --upgrade package_name

# Uninstall a package
pip uninstall package_name
            ''',
            "Docker Basics": '''
# Pull an image
docker pull python:3.8

# Run a container
docker run -it python:3.8 bash

# Build an image from Dockerfile
docker build -t myimage .

# List running containers
docker ps

# Stop a container
docker stop container_id

# Remove a container
docker rm container_id

# Remove an image
docker rmi myimage
            ''',
            "Jupyter Notebook Shortcuts": '''
# Create a new notebook
jupyter notebook

# Keyboard Shortcuts
- Shift + Enter: Run cell and move to next
- Ctrl + Enter: Run cell
- A: Insert cell above
- B: Insert cell below
- M: Convert to Markdown
- Y: Convert to Code
- D + D: Delete cell
- Z: Undo cell deletion
- H: Show help
            ''',
            "Git Commands": '''
# Initialize repository
git init

# Clone repository
git clone https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet.git

# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Commit message"

# Push to remote
git push origin main

# Pull from remote
git pull origin main

# View commit history
git log

# View branches
git branch

# Create a new branch
git branch feature-branch

# Switch to a branch
git checkout feature-branch
            ''',
            "Git Branching": '''
# Create a new branch
git branch feature-branch

# Switch to the branch
git checkout feature-branch

# Create and switch
git checkout -b new-feature

# Merge branch into main
git checkout main
git merge feature-branch

# Delete branch
git branch -d feature-branch

# Rename branch
git branch -m old-name new-name

# List all branches
git branch -a
            ''',
            "Git Stashing": '''
# Stash changes
git stash

# Apply stashed changes
git stash apply

# List stashes
git stash list

# Drop a stash
git stash drop stash@{0}

# Pop the latest stash
git stash pop
            ''',
            "Git Conflict Resolution": '''
# After a merge conflict, edit the files to resolve

# Add resolved files
git add conflicted_file.py

# Commit the merge
git commit -m "Resolved merge conflict in conflicted_file.py"

# Continue rebase
git rebase --continue

# Abort rebase
git rebase --abort
            ''',
            "Git Rebasing": '''
# Start rebase
git checkout feature-branch
git rebase main

# Continue rebase after resolving conflicts
git add .
git rebase --continue

# Abort rebase
git rebase --abort

# Interactive rebase
git rebase -i HEAD~3
            ''',
            "Git Cherry-Picking": '''
# Cherry-pick a commit
git cherry-pick commit_hash

# Cherry-pick a range of commits
git cherry-pick start_commit^..end_commit
            '''
        },
        "☁️ Cloud Services": {
            "AWS Basics": '''
# Install AWS CLI
pip install awscli

# Configure AWS CLI
aws configure

# List S3 buckets
aws s3 ls

# Upload a file to S3
aws s3 cp local_file.txt s3://mybucket/

# Download a file from S3
aws s3 cp s3://mybucket/file.txt .

# Sync local directory with S3
aws s3 sync ./local_folder s3://mybucket/folder
            ''',
            "Google Cloud Platform (GCP) Basics": '''
# Install Google Cloud SDK
# Visit https://cloud.google.com/sdk/docs/install for installation steps

# Initialize
gcloud init

# List projects
gcloud projects list

# Deploy to App Engine
gcloud app deploy

# List Compute Engine instances
gcloud compute instances list

# Create a new Compute Engine instance
gcloud compute instances create my-instance --zone=us-central1-a
            ''',
            "Microsoft Azure Basics": '''
# Install Azure CLI
# Visit https://docs.microsoft.com/en-us/cli/azure/install-azure-cli for installation steps

# Login
az login

# List resource groups
az group list

# Create a resource group
az group create --name myResourceGroup --location eastus

# List virtual machines
az vm list

# Create a virtual machine
az vm create --resource-group myResourceGroup --name myVM --image UbuntuLTS --admin-username azureuser --generate-ssh-keys
            ''',
            "Deploying Models to AWS SageMaker": '''
import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define model
from sagemaker.sklearn import SKLearnModel
model = SKLearnModel(model_data='s3://path-to-model/model.tar.gz',
                    role=role,
                    entry_point='inference.py')

# Deploy model
predictor = model.deploy(instance_type='ml.m4.xlarge', initial_instance_count=1)

# Make predictions
response = predictor.predict({'data': [sample_data]})
print(response)
            ''',
            "Azure Machine Learning": '''
from azureml.core import Workspace, Dataset

# Connect to workspace
ws = Workspace.from_config()

# Access dataset
dataset = Dataset.get_by_name(ws, name='my_dataset')

# Load dataset into pandas DataFrame
df = dataset.to_pandas_dataframe()

# Register a new dataset
new_dataset = Dataset.Tabular.register_pandas_dataframe(df, ws, 'new_dataset_name')
            ''',
            "Google Cloud AI Platform": '''
from google.cloud import aiplatform

# Initialize AI Platform
aiplatform.init(project='my-project', location='us-central1')

# Deploy model
model = aiplatform.Model.upload(
    display_name='my_model',
    artifact_uri='gs://my-bucket/model/',
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-3:latest'
)

endpoint = model.deploy(
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=2
)

# Make predictions
response = endpoint.predict(instances=[sample_instance])
print(response)
            '''
        },
        "🔍 Natural Language Processing (NLP)": {
            "Text Preprocessing": '''
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

sample_text = "Data Science is amazing! Let's explore its potentials."
print(preprocess(sample_text))
            ''',
            "Bag of Words with Scikit-learn": '''
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample documents
documents = [
    "Data Science is fascinating.",
    "Machine Learning is a subset of Data Science.",
    "Natural Language Processing is a part of Machine Learning."
]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform
X = vectorizer.fit_transform(documents)

# Get feature names
features = vectorizer.get_feature_names_out()
print(features)

# Convert to DataFrame
df_bow = pd.DataFrame(X.toarray(), columns=features)
print(df_bow)
            ''',
            "TF-IDF Vectorization": '''
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = [
    "Data Science is fascinating.",
    "Machine Learning is a subset of Data Science.",
    "Natural Language Processing is a part of Machine Learning."
]

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer()

# Fit and transform
X = tfidf.fit_transform(documents)

# Get feature names
features = tfidf.get_feature_names_out()
print(features)

# Convert to DataFrame
df_tfidf = pd.DataFrame(X.toarray(), columns=features)
print(df_tfidf)
            ''',
            "Word Embeddings with Gensim": '''
from gensim.models import Word2Vec

# Sample documents
documents = [
    "Data Science is fascinating.",
    "Machine Learning is a subset of Data Science.",
    "Natural Language Processing is a part of Machine Learning."
]

# Tokenize sentences
sentences = [doc.split() for doc in documents]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get vector for a word
vector = model.wv['Data']
print(vector)

# Find similar words
similar = model.wv.most_similar('Data', topn=3)
print(similar)
            ''',
            "Sentiment Analysis with NLTK": '''
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize
sia = SentimentIntensityAnalyzer()

# Analyze sentiment
sentiment = sia.polarity_scores("I love Data Science! It's absolutely amazing.")
print(sentiment)

# Example output:
# {'neg': 0.0, 'neu': 0.392, 'pos': 0.608, 'compound': 0.6696}
            ''',
            "Named Entity Recognition with SpaCy": '''
import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Apple is looking at buying U.K. startup for $1 billion."

# Process text
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)
            ''',
            "Topic Modeling with LDA": '''
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "Data Science is an interdisciplinary field.",
    "Machine Learning is a subset of Data Science.",
    "Natural Language Processing deals with text data.",
    "Deep Learning models are powerful for image recognition."
]

# Vectorize text
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Initialize LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)

# Fit LDA
lda.fit(X)

# Display topics
for index, topic in enumerate(lda.components_):
    print(f'Topic #{index +1}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])
            ''',
            "Text Classification with Scikit-learn": '''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Sample data
documents = [
    "I love machine learning.",
    "Data science is fascinating.",
    "Natural language processing is a part of AI.",
    "Deep learning models are complex."
]
labels = [1, 1, 1, 0]  # 1: Positive, 0: Negative

# Vectorize text
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documents)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Initialize and train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predictions
predictions = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))
            '''
        },
        "📅 Time Series": {
            "Time Series Decomposition": '''
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('timeseries.csv', parse_dates=['Date'], index_col='Date')

# Decompose
decomposition = seasonal_decompose(df['Value'], model='additive')
fig = decomposition.plot()
plt.show()
            ''',
            "ARIMA Modeling": '''
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(df['Value'], order=(1,1,1))
model_fit = model.fit()

# Summary
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=10)
print(forecast)
            ''',
            "Prophet Forecasting": '''
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Prepare data
df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Value': 'y'})

# Initialize and fit
model = Prophet()
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=30)

# Predict
forecast = model.predict(future)

# Plot
fig = model.plot(forecast)
plt.show()

# Plot components
fig2 = model.plot_components(forecast)
plt.show()
            ''',
            "Rolling Statistics": '''
# Moving Average
df['MA'] = df['Value'].rolling(window=12).mean()

# Moving Standard Deviation
df['STD'] = df['Value'].rolling(window=12).std()

# Plot rolling statistics
plt.figure(figsize=(12,6))
plt.plot(df['Value'], label='Original')
plt.plot(df['MA'], label='Moving Average')
plt.plot(df['STD'], label='Moving Std Dev')
plt.legend()
plt.title('Rolling Statistics')
plt.show()
            ''',
            "Seasonal ARIMA (SARIMA)": '''
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Define SARIMA model
model = SARIMAX(df['Value'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Summary
print(model_fit.summary())

# Forecast
forecast = model_fit.get_forecast(steps=12)
print(forecast.predicted_mean)

# Plot forecast
forecast.plot()
plt.show()
            ''',
            "Exponential Smoothing": '''
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Fit model
model = ExponentialSmoothing(df['Value'], trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Summary
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=12)
print(forecast)

# Plot forecast
plt.figure(figsize=(10,6))
plt.plot(df['Value'], label='Original')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.title('Exponential Smoothing Forecast')
plt.show()
            '''
        },
        "🔄 Data Pipelines": {
            "Scikit-learn Pipelines": '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
print(predictions)
            ''',
            "FeatureUnion for Parallel Processing": '''
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

# Define FeatureUnion
features = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('select', SelectKBest(k=1))
])

# Integrate into pipeline
pipeline = Pipeline([
    ('features', features),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
            ''',
            "Custom Transformers": '''
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1):
        self.param = param
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Custom transformation logic
        return X * self.param

# Use in pipeline
pipeline = Pipeline([
    ('custom', CustomTransformer(param=2)),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
            ''',
            "Pipeline with Multiple Steps": '''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Define numeric and categorical features
numeric_features = ['Age', 'Experience']
categorical_features = ['City']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression())
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
print(predictions)
            ''',
            "Pipeline with Feature Selection": '''
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=2)),
    ('clf', LogisticRegression())
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
print(predictions)
            '''
        },
        "🚀 Deployment": {
            "Saving and Loading Models": '''
import joblib
import pickle

# Save with joblib
joblib.dump(model, 'linear_regression_model.joblib')

# Load with joblib
loaded_model = joblib.load('linear_regression_model.joblib')
print(loaded_model.predict([[25, 5]]))

# Save with pickle
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load with pickle
with open('linear_regression_model.pkl', 'rb') as f:
    loaded_model_pickle = pickle.load(f)
print(loaded_model_pickle.predict([[30, 10]]))
            ''',
            "Deploying with Streamlit": '''
# Create a simple Streamlit app to deploy the model
import streamlit as st
import joblib

# Load model
model = joblib.load('linear_regression_model.joblib')

st.title('Salary Prediction App')

# Input features
age = st.number_input('Age', min_value=18, max_value=100, value=25)
experience = st.number_input('Years of Experience', min_value=0, max_value=80, value=5)

# Predict
if st.button('Predict'):
    prediction = model.predict([[age, experience]])
    st.success(f'Predicted Salary: ${prediction[0]:.2f}')
            ''',
            "Deploying with Flask": '''
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('linear_regression_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    age = data['Age']
    experience = data['Experience']
    prediction = model.predict([[age, experience]])
    return jsonify({'Prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
            ''',
            "Deploying with Docker": '''
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
            ''',
            "Deploying to AWS Elastic Beanstalk": '''
# Initialize Elastic Beanstalk
eb init -p python-3.8 my-data-science-app

# Create environment and deploy
eb create my-data-science-env

# Open the app
eb open

# Update application
eb deploy
            ''',
            "Deploying to Heroku": '''
# Create a Procfile
echo "web: streamlit run app.py" > Procfile

# Initialize git repository
git init
git add .
git commit -m "Initial commit"

# Login to Heroku
heroku login

# Create a new Heroku app
heroku create my-data-science-app

# Push code to Heroku
git push heroku main

# Open the app
heroku open
            ''',
            "Containerizing with Docker Compose": '''
# docker-compose.yml
version: '3'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
    environment:
      - STREAMLIT_SERVER_PORT=8501
    command: streamlit run app.py
            '''
        }
    }

    # Split main categories into two rows
    main_categories = list(sections.keys())
    mid_point = len(main_categories) // 2
    row1_categories = main_categories[:mid_point]
    row2_categories = main_categories[mid_point:]

    # Function to render a row of tabs
    def render_tab_row(categories):
        tab_titles = categories
        tabs = st.tabs(tab_titles)
        for category, tab in zip(categories, tabs):
            with tab:
                subtopics = sections[category]
                sub_tabs = st.tabs(list(subtopics.keys()))
                for sub_tab, (sub_title, code) in zip(sub_tabs, subtopics.items()):
                    with sub_tab:
                        # Determine language based on sub_title
                        language = 'python'
                        if 'bash' in sub_title.lower() or 'shell' in sub_title.lower():
                            language = 'bash'
                        elif 'sql' in sub_title.lower():
                            language = 'sql'
                        elif 'dockerfile' in sub_title.lower():
                            language = 'dockerfile'
                        elif 'yaml' in sub_title.lower():
                            language = 'yaml'
                        elif 'text' in sub_title.lower():
                            language = 'text'
                        elif 'docker' in sub_title.lower() and 'compose' in sub_title.lower():
                            language = 'yaml'
                        st.code(code, language=language)

    # Render first row of tabs
    render_tab_row(row1_categories)

    # Render second row of tabs
    render_tab_row(row2_categories)

    # Footer with social media links
    st.markdown(f"""
        <div style="background-color: #FFFFFF; color: black; text-align: center; padding: 20px; margin-top: 50px; border-top: 2px solid #000000;">
            <p>Connect with me:</p>
            <div style="display: flex; justify-content: center; gap: 20px;">
                <a href="https://github.com/shemanto27" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" alt="GitHub" width="30" style="transition: transform 0.2s;">
                </a>
                <a href="https://www.linkedin.com/in/shemanto/" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/919/919827.png" alt="Portfolio" width="30" style="transition: transform 0.2s;">
                </a>
            </div>
            <br>
            Data Science Cheat Sheet v1.0 | Nov 2024 | <a href="" style="color: #000000;">Shemanto Sharkar</a>
            <div class="card-footer">Shemanto Sharkar © 2024</div>
        </div>
    """, unsafe_allow_html=True)

    # Optional: Add some spacing at the bottom
    st.markdown("<br><br><br>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
