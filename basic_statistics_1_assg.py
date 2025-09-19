
"""1.Descriptive Analytics for Numerical Columns"""
#Loading the Dataset
import pandas as pd
df=pd.read_csv('/content/sales_data_with_discounts.csv')

#Identifying the Numeical columns
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
print(numerical_columns)

#Calculating the Statistics
mean=df[numerical_columns].mean()
median=df[numerical_columns].median()
mode=df[numerical_columns].mode().iloc[0]
std_dev=df[numerical_columns].std()
print("Mean:\n",mean)
print("\nMedian:\n",median)
print("\nMode:\n",mode)
print("\nStandard Deviation:\n",std_dev)

"""Interpretation of Statistic
1.   Mean: The average value of the data. It gives a central tendency of the data but can be affected by outliers.
2.   Median:The middle value when the data is sorted.The median is less sensitive to outliers than the mean
3. Mode: The most frequently occuring value in the dataset.
4. Standard Deviation:A measure of the amount of variation or dispersion in the data.A higher std dev indicates that the values are more spread out from the mean,while the lower value means they are closer to the mean.

**2.Data Visualization**

**HISTOGRAM**
"""

#Plotting Histograms, Analyzing Skewness and Detecting Outliers
import matplotlib.pyplot as plt
import seaborn as sns
def outlier_detection(df,col):
  q1=df[col].quantile(0.25)
  q3=df[col].quantile(0.75)
  iqr=q3-q1
  upper_extreme=q3+(1.5 * iqr)
  lower_extreme=q1-(1.5 * iqr)
  outliers=df[(df[col]<lower_extreme) | (df[col]>upper_extreme)][col]
  return outliers
plt.figure(figsize=(12,8))
for i,col in enumerate(numerical_columns,1):
   plt.subplot(2,3,i)
   skewness=df[col].skew()
   print("Skewness of", col,":" ,round(skewness))
   sns.histplot(df[col], bins=20,kde=True,color="indigo")
   outliers= outlier_detection(df,col)
   if not outliers.empty:
    plt.scatter(outliers,[0]* len(outliers), color="Red", edgecolors="Black", s=80, label="Outliers",zorder=3)
   plt.xlabel(col)
   plt.ylabel("Frequency")
   plt.legend()

plt.tight_layout()
plt.show()

"""Inferences of Skewness (KDE Curve is showing)

Skewness=0 Symmetric Distribution     
Skewness>0 Right-skewed   
Skewness<0 Left-skewed

Inferences of Outliers (Used IQR Method)

Right-skewed -> Presence of high value outliers on the right  
Left-skewed -> Presence of low value outliers on the left

**BOXPLOT**
"""

#Create Boxplot to visualize the IQR and detect Outliers
plt.figure(figsize=(8,6))
sns.boxplot(data=df[numerical_columns],palette="coolwarm")
plt.title("Boxplot of Numerical Columns")
plt.xlabel("Numerical Columns")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()

"""Findings, Extreme Values or Unusual Distributions

1.   Some columns have extreme values(outliers) appearing as points outside the whiskers, indicating unusual transaction or data entry error.
2.  Skewed Distribution
*   Right-skewed -> Median is closer to the lower quartile and the right whisker is longer.
*  Left-skewed -> Median is closer to the upper quartile and left whisker is longer.
3.  Wide vs Narrow IQR
*   A wider box indicates high variability in the data.
*  A Narrow box suggests that most values are closely packed around the median
4.  A column without outliers is well-distributed, while a short or mixing box indicates low variability or a dominant value.

**BARPLOT**
"""

categorical_columns=df.select_dtypes(include=['object']).columns.tolist()
print(categorical_columns)

num_rows = int(len(categorical_columns) / 3) + (len(categorical_columns) % 3 > 0)
plt.figure(figsize=(12,8))
for i,col in enumerate(categorical_columns):
  plt.subplot(num_rows, 3, i + 1)
  sns.countplot(data=df,x=col,hue=col,legend=False,palette="viridis")
  plt.xlabel(col)
  plt.ylabel("Count")
  plt.title(f"Distribution of {col}")
  plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""Distribution of categories and Insights

1. Balanced Categories: Data is well distributed when categories have bar heights that are approximately equal.  
2. Dominant Categories :Bars that are extremely tall suggest that some groups pre-dominant in the dataset.
3. Rare Categories: Very short bars indicate irregular or uncommon events.

**3.Standardization of Numerical Variables**

Concept of standardization (z-score normalization)   

Data is transformed using Z-score normalization, also known as standardization, to have a mean (average) of 0 and a standard deviation of 1.  This procedure modifies data values according to their standard deviation, which is a measure of how far they differ from the mean.  

Z = X−μ / σ

(Z) is the Z-score.  
(X) is the value of the data point.  
(μ) is the mean of the dataset.  
(σ) is the standard deviation of the dataset.
"""

plt.figure(figsize=(12,8))
mean=df[numerical_columns].mean()
std_dev=df[numerical_columns].std()
standardized_df=(df[numerical_columns]-mean)/std_dev
for i,col in enumerate(numerical_columns,1):
   plt.subplot(3,3,i)
   sns.histplot(df[col], bins=20,kde=True,color="brown",label="Before std")
   sns.histplot(standardized_df[col], bins=20,kde=True,color="green",label="After std")
   plt.title(f"Distribution Before and After Std: {col}")
   plt.legend()
plt.tight_layout()
plt.show()

"""**4.Conversion of Categorical Data into Dummy Variables**

One-Hot Encoding

Categorical data often contains non-numeric values (like city names, product categories, or gender) that cannot be directly used in mathematical models. Since most machine learning algorithms and statistical models require numerical input, we must convert these text categories into numbers.

1.No ordinal misinterpretation  
2.Improved model accuracy  
3.Versatility
"""

categorical_columns=df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df[categorical_columns])
print("Transfored dataset(One-Hot Encoded):\n")
print(df_encoded.head())

"""**5.Descriptive Analytics and Data Visualizations**

1. Data in numerical columns like "Total Sales Value" and "Discount Amount" show right-skewed distributions, which suggests that there are high outlier values.
2. Standardization (Z-score normalization) helped reduce the impact of outliers, centering the data around zero.
3. With a few categories controlling the data, some categorical variables (such as "City" and "Brand") exhibit a notable imbalance.
4. Some models and brands have far greater numbers than others, according to visualizations.
5. After encoding, the converted dataset has a much higher number of columns, which increases dimensionality.

**6.Standardization and One-Hot Encoding**

1. Standardization scales numerical data to a common range, reducing the impact of outliers and improving model performance.
2. One-hot encoding converts categorical variables into numerical format, enabling models to process non-numeric data effectively.
3. These techniques enhance model accuracy and interpretability.
"""