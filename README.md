# PYSAPRK-CLASSIFCATION-MODEL

Leveraging big data insights using the normal sklearn libraries is at times not advisable ,this is beacuse big data needs more computing ,using the normal libraries will take a much longer time and the computing accuracy will definitely not be at its besttats why we need to use **pyspark** to analyze these big datasets and create ml models to cumpute this
for faste rpreprocessing and faster creation of thse model
### Project Overview
In the following model we will be implementing a classfication model wwhere we will be using pyspark to preprocess the data,eda ,model creation,we will try different  classification models and when we settle on our idela model we will tune it also using pyspark and lets see what accuracy we will get get our best perfoming model and try to tune it,its a begginner friendly model where we start from the basics to advances cocpts all in one

we will be doing a classifcation algorithm problem is hwether a customer wil cimmit to a term deposit .This can be used to asssess the credit risk of a customer.

### Data ingestion and data cleaning
We will be  importing some few libraries
```
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('bank.csv', header = True, inferSchema = True)
df.printSchema()
```
*Dataset Column Names and Data Types*

- squareMeters: int
- numberOfRooms: int
- hasYard: int
- hasPool: int
- floors: int
- cityCode: int
- cityPartRange: int
- numPrevOwners: int
- made: int
- isNewBuilt: int
- hasStormProtector: int
- basement: int
- attic: int
- garage: int
- hasStorageRoom: int
- hasGuestRoom: int
- price: double


Input variables: age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome. taget variable :deposit(Yes/No)

## EDA
we will bes looking at the **first five columns** . we will be using the pandas dataframe instead f the spark freame.show()
```
import pandas as pd
pd.DataFrame(df.take(5), columns=df.columns)
```
in the following we will be assessing the traget column variable where we will see if our **target column** is balanced if not then we will need to balance the taget variable as our model would then be biased to a certain class

`df.groupby('deposit').count().toPandas()`

deposit	count
0	no	5873
1	yes	5289


Here we will be looking at the summary **statistics** of our numeric columns . This comes in handy when dealing with outliers and also looking at the distribution of the various columns.
```
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().toPandas()
```
```
import pandas as pd
import matplotlib.pyplot as plt

numeric_data = df.select(numeric_features).toPandas()

axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8))

# Rotate axis labels and remove axis ticks
n = len(numeric_data.columns)
for i in range(n):
    for j in range(n):
        ax = axs[i, j]
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
        ax.set_yticks(())
        ax.xaxis.label.set_rotation(90)
        ax.set_xticks(())

plt.show()
```
![a](https://github.com/stilinsk/PYSAPRK-CLASSIFCATION-MODEL/assets/113185012/5c277fa6-5b27-43c0-8b64-c1015cda1262)

We can see that there is not that concrete correlation between the columns above Therefore we can keeep all the variables for our modelexcept for the time (dayand month)basicallly a month will not likely influemvce whether a deposit will be made or not


### Data preprocessing and model building
We will start with category indexing ,**One Hot encoding** and **Vector Assembler**-a feature transformer that merges multiple columns into a vector column The code is available at the databricks site and it indexes each categorical columnusing the **string Indexer** and converts the indexed categories into one hot encoded varibales.
The resulting output has the binary vectors appended to the end of each row. We use the **StringIndexer** again to encode our labels to label indices. Next, we use the **VectorAssembler** to combine all the feature columns into a single vector column.

```
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]

numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
```

**Vector Assembler**: A VectorAssembler is created to assemble all the feature columns, including the **one-hot encoded categorical*8 columns and the numeric columns, into a single feature vector named **'features'**. The input columns are specified as **assemblerInputs*8, which is a concatenation of the one-hot encoded categorical column names and the numeric column names. The **VectorAssembler is added to the stages list**.

The stages list will contain all the necessary stages of the**data preprocessing pipeline, including string indexing, one-hot encoding, label string indexing, and vector assembling**. These stages can then be used in a Pipeline for further processing or training a machine learning model


