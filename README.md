# Predictive Analysis for Uncovering the Causes of Major Power Outages

**Authors**: Jessica Zhang, Stephanie Wang

---

## Project Overview

This is a data science project on predicting the causes of a major power outage. The dataset used to explore the topic can be find [here](https://engineering.purdue.edu/LASCI/research-data/outages). This project is for DSC80 at UCSD.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Introduction](#introduction)
    - [Introduction to the Dataset in the Study](#introduction-to-the-dataset-in-the-study)
3. [Cleaning and Exploratory Data Analysis](#cleaning-and-exploratory-data-analysis)
    - [Data Cleaning](#data-cleaning)
    - [Univariate Analysis](#univariate-analysis)
        - [Distribution of Duration in Hours](#distribution-of-duration-in-hours)
        - [Distribution of Anomaly Level](#distribution-of-anomaly-level)
    - [Bivariate Analysis](#bivariate-analysis)
    - [Interesting Aggregates](#interesting-aggregates)
4. [Assessment of Missingness](#assessment-of-missingness)
    - [NMAR Analysis](#nmar-analysis)
    - [Missingness Dependency](#missingness-dependency)
5. [Hypothesis Testing](#hypothesis-testing)
    - [Framing a Prediction Problem](#framing-a-prediction-problem)
        - [At the "Time of Prediction"](#at-the-time-of-prediction)
6. [Baseline Model](#baseline-model)
    - [Feature Engineering](#feature-engineering)
7. [Final Model](#final-model)
    - [Feature Selection](#feature-selection)
    - [Modeling Selection](#modeling-selection)
    - [Finding Hyperparameters](#finding-hyperparameters)
    - [KFold Cross Validation](#kfold-cross-validation)
    - [Comparison with Baseline Model](#comparison-with-baseline-model)
8. [Fairness Analysis](#fairness-analysis)
9. [References](#references)

---

## Introduction

[Back to Top](#table-of-contents)

Power exists everywhere in our lives, woven into the fabric of our daily routines and essential activities. We use power to brighten our homes, to charge the devices that keep us connected to the world, and to preserve the food that nourishes us. Imagine this: You're about to sit down for a family dinner or dive into a critical work project when suddenly, everything goes dark. The power's out, again. Frustration kicks in, plans get disrupted, and the uncertainty of when things will return to normal looms over you. This scenario is far too common for many of us and highlights a pressing issue in our modern lives—the vulnerability of our power systems to outages. The Department of Energy defines large outages as those that impact at least 50,000 customers or result in an unscheduled firm load loss of at least 300 MW (Mukherjee et al. 2018).  *In order to minimize the inconveniences associated with power outages, we will uncover the cause of a major power outage in U.S.*

### Introduction to the Dataset in the Study

[Back to Top](#table-of-contents)

The dataset for power outages contains *1534* rows, each representing information on major outages experienced by different states from 2016 to 2020, with *55* columns regarding to the following information.

| Column Name | Description |
| --- | --- |
| `YEAR` | Year when the outage occurred |
| `MONTH` | Month when the outage occurred |
| `U.S._STATE` | States in U.S. |
| `POSTAL.CODE` | Postal code of the states |
| `NERC.REGION` | North American Electric Reliability Corporation regions involved in the outage |
| `CLIMATE.REGION` | U.S. climate regions |
| `ANOMALY.LEVEL` | Oceanic El Niño/La Niña (ONI) index |
| `CLIMATE.CATEGORY` | Climate episodes |
| `OUTAGE.START.DATE` | Day of the year when the outage started |
| `OUTAGE.START.TIME` | Time of the day when the outage started |
| `OUTAGE.RESTORATION.DATE` | Day of the year when power was restored |
| `OUTAGE.RESTORATION.TIME` | Time of the day when power was restored |
| `CAUSE.CATEGORY` | Category causing major power outages |
| `CAUSE.CATEGORY.DETAIL` | Description of categories |
| `HURRICANE.NAMES` | Hurricane name |
| `OUTAGE.DURATION` | Duration of outage |
| `DEMAND.LOSS.MW` | Amount of peak demand lost |
| `CUSTOMERS.AFFECTED` | Number of customers affected by the power outages |
| `RES.PRICE` | Monthly electricity price in the residential sector |
| `COM.PRICE` | Monthly electricity price in the commercial sector |
| `IND.PRICE` | Monthly electricity price in the industrial sector |
| `TOTAL.PRICE` | Average monthly electricity price |
| `RES.SALES` | Residential electricity consumption |
| `COM.SALES` | Commercial electricity consumption |
| `IND.SALES` | Industrial electricity consumption |
| `TOTAL.SALES` | Total electricity consumption |
| `RES.PERCEN` | Percentage of residential electricity consumption |
| `COM.PERCEN` | Percentage of commercial electricity consumption |
| `IND.PERCEN` | Percentage of industrial electricity consumption |
| `RES.CUSTOMERS` | Annual number of customers served in the residential electricity sector |
| `COM.CUSTOMERS` | Annual number of customers served in the commercial electricity sector |
| `IND.CUSTOMERS` |Annual number of customers served in the industrial electricity sector |
| `TOTAL.CUSTOMERS` | Annual number of total customers |
| `RES.CUST.PCT` | Percent of residential customers |
| `COM.CUST.PCT` | Percent of commerical customers |
| `IND.CUST.PCT` | Percent of industrial customers |
| `PC.REALGSP.STATE` | Per capita real gross state product in the states |
| `PC.REALGSP.USA` | Per capita real gross state product in U.S. |
| `PC.REALGSP.REL` | Relative per capita real gross state product |
| `PC.REALGSP.CHANGE` | Percentage change of per capita real GSP from the previous year |
| `UTIL.REALGSP` | Real capita real gross state product contributed by Utility industry |
| `TOTAL.REALGSP` | Real capita real gross state product contributed by all industries |
| `UTL.CONTRI` | Utility industry's contribution to the total capita real gross state product in the state |
| `PI.UTIL.OFUSA` | State utility sector׳s income as a percentage of the total earnings of the utility sector's income |
| `POPULATION` | Population in the state |
| `POPPCT_URBAN` |Percentage of the total population of the state represented by the urban population |
| `POPPCT_UC` | Percentage of the total population of the state represented by the population of the urban clusters |
| `POPDEN_URBAN` | Population density of the urban areas |
| `POPDEN_UC` | Population density of the urban clusters |
| `POPDEN_RURAL` | Population density of the rural areas |
| `AREAPCT_URBAN` | Percentage of the land area of state represented by the land area of the urban areas |
| `AREAPCT_UC` | Percentage of the land area of state represented by the land area of the urban clusters |
| `PCT_LAND` | Percentage of land area in state as compared to the overall land area |
| `PCT_WATER_TOT` | Percentage of water area in the state as compared to the overall water area |
| `PCT_WATER_INLAND` | Percentage of inland water area in the state as compared to the overall inland water area |

In the project, we used several key columns from the dataset to understand the factors leading to power outages, particularly focusing on those caused by severe weather. The columns of interest spanned various dimensions, including electricity pricing (e.g., `RES.PRICE` for residential, `COM.PRICE` for commercial, `IND.PRICE` for industrial sectors), sales volumes (`RES.SALES`, `COM.SALES`, `IND.SALES`), customer demographics (`RES.CUSTOMERS`, `COM.CUSTOMERS`, `IND.CUSTOMERS`), and broader economic indicators such as state and national GDP (`PC.REALGSP.STATE`, `PC.REALGSP.USA`). 

---

## Cleaning and Exploratory Data Analysis

[Back to Top](#table-of-contents)

### Data Cleaning

First, we read the Excel file and loaded into notebook. In order to increase the readability of the data and performance of our model, we converted the dataset into a dataframe called "*power*" and conducted data cleaning through the following steps:

- **Drop meaningless columns and reset the index starting from 0**: The original Excel file includes a row that illustrates the unit of measurement for some columns, as well as a column labeled OBS indicating the sequence of cases as 1, 2, 3, 4, and so on.
	
- **Replace missing values with np.nan or 0**: It  ensures a consistent representation of missing values across different columns and data types, making it easier to identify, count, and handle these missing entries. For columns `MONTHS` and `YEAR`, we filled null values with 0.

- **Check the datatypes of each column, especially the columns related to datetime**: The `OUTAGE.START.DATE`, `OUTAGE.START.DATE`, `OUTAGE.RESTORATION.DATE`, `OUTAGE.RESTORATION.TIME` columns in the dataframe are stored as objects. We converted it into datetime data type so that we can apply basic operations later. Convert numerical columns such as `ANOMALY.LEVEL`, `OUTAGE.DURATION`, `DEMAND.LOSS.MW`, etc., into float or int types.

- **Merging date and time into a single column**: Since the columns (`OUTAGE.START.DATE`, `OUTAGE.START.DATE`, `OUTAGE.RESTORATION.DATE`, `OUTAGE.RESTORATION.TIME`) have redudancy. This process creates a more comprehensive and accurate representation of specific events or records in a dataset. The resulting singular datetime column then allows for more efficient sorting, filtering, and time series analysis within the data. Then, we dropped the original columns from the "power" dataframe.

To explore further about the question, we decided to assign some new columns:

- **Adding time duration in hours**: By extracting the day of the week from the outage start and restoration times, we can observe and analyze patterns in outage occurrences and recovery times across different days.

- **Adding day for start and restoration**: By identifying the specific days on which power outages begin and end, we can better understand the temporal patterns of outages and assess their impact more accurately."

- **Adding climate severity value and severity category columns**: By quantifying the anomaly level of weather conditions (how much hotter, colder, wetter, or drier it is than usual), we can directly link the severity of weather conditions to the frequency and duration of power outages. The value in the "Climate Severity" column represents the degree of anomaly in weather conditions at the time of the outage. Higher values indicate more extreme weather conditions relative to the historical norm for that region and time of year. While the "Climate Severity" provides a quantitative measure of weather conditions' extremity, the "Severity Category" simplifies this information into qualitative categories. This categorization makes it easier to analyze and communicate the data, especially for non-technical stakeholders. It helps in quickly identifying patterns and trends without needing to interpret specific numerical values.

Overall, adding new columns facilitates the comparison of outage characteristics across different severity levels, enabling targeted investigations into the resilience of power systems under various weather scenarios.

After cleaning the dataframe, the dataframe looks like ths following (only showing the first 5 rows for illustration):

|   YEAR |   MONTH | U.S._STATE   | POSTAL.CODE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   HURRICANE.NAMES |   OUTAGE.DURATION |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |   RES.PRICE |   COM.PRICE |   IND.PRICE |   TOTAL.PRICE |   RES.SALES |   COM.SALES |   IND.SALES |   TOTAL.SALES |   RES.PERCEN |   COM.PERCEN |   IND.PERCEN |   RES.CUSTOMERS |   COM.CUSTOMERS |   IND.CUSTOMERS |   TOTAL.CUSTOMERS |   RES.CUST.PCT |   COM.CUST.PCT |   IND.CUST.PCT |   PC.REALGSP.STATE |   PC.REALGSP.USA |   PC.REALGSP.REL |   PC.REALGSP.CHANGE |   UTIL.REALGSP |   TOTAL.REALGSP |   UTIL.CONTRI |   PI.UTIL.OFUSA |   POPULATION |   POPPCT_URBAN |   POPPCT_UC |   POPDEN_URBAN |   POPDEN_UC |   POPDEN_RURAL |   AREAPCT_URBAN |   AREAPCT_UC |   PCT_LAND |   PCT_WATER_TOT |   PCT_WATER_INLAND | OUTAGE.START        | OUTAGE.RESTORATION   |   OUTAGE.DURATION.HOUR | OUTAGE.START.DAY   | OUTAGE.RESTORATION.DAY   |   Absolute Anomaly Level |   Anomaly Frequency Rank |   Climate Severity | Severity Category   |
|-------:|--------:|:-------------|:--------------|:--------------|:-------------------|----------------:|:-------------------|:-------------------|:------------------------|------------------:|------------------:|-----------------:|---------------------:|------------:|------------:|------------:|--------------:|------------:|------------:|------------:|--------------:|-------------:|-------------:|-------------:|----------------:|----------------:|----------------:|------------------:|---------------:|---------------:|---------------:|-------------------:|-----------------:|-----------------:|--------------------:|---------------:|----------------:|--------------:|----------------:|-------------:|---------------:|------------:|---------------:|------------:|---------------:|----------------:|-------------:|-----------:|----------------:|-------------------:|:--------------------|:---------------------|-----------------------:|:-------------------|:-------------------------|-------------------------:|-------------------------:|-------------------:|:--------------------|
|   2011 |       7 | Minnesota    | MN            | MRO           | East North Central |            -0.3 | normal             | severe weather     | nan                     |               nan |              3060 |              nan |                70000 |       11.6  |        9.18 |        6.81 |          9.28 | 2.33292e+06 | 2.11477e+06 | 2.11329e+06 |   6.56252e+06 |      35.5491 |      32.225  |      32.2024 |     2.30874e+06 |          276286 |           10673 |       2.5957e+06  |        88.9448 |        10.644  |       0.411181 |              51268 |            47586 |          1.07738 |                 1.6 |           4802 |          274182 |       1.75139 |             2.2 |  5.34812e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2011-07-01 17:00:00 | 2011-07-03 20:00:00  |             51         | Friday             | Sunday                   |                      0.3 |                      191 |          0.130435  | Very Common         |
|   2014 |       5 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | intentional attack | vandalism               |               nan |                 1 |              nan |                    0 |       12.12 |        9.71 |        6.49 |          9.28 | 1.58699e+06 | 1.80776e+06 | 1.88793e+06 |   5.28423e+06 |      30.0325 |      34.2104 |      35.7276 |     2.34586e+06 |          284978 |            9898 |       2.64074e+06 |        88.8335 |        10.7916 |       0.37482  |              53499 |            49091 |          1.08979 |                 1.9 |           5226 |          291955 |       1.79    |             2.2 |  5.45712e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2014-05-11 18:38:00 | 2014-05-11 18:39:00  |              0.0166667 | Sunday             | Sunday                   |                      0.1 |                       61 |          0.0138857 | Very Common         |
|   2010 |      10 | Minnesota    | MN            | MRO           | East North Central |            -1.5 | cold               | severe weather     | heavy wind              |               nan |              3000 |              nan |                70000 |       10.87 |        8.19 |        6.07 |          8.15 | 1.46729e+06 | 1.80168e+06 | 1.9513e+06  |   5.22212e+06 |      28.0977 |      34.501  |      37.366  |     2.30029e+06 |          276463 |           10150 |       2.58690e+06 |        88.9206 |        10.687  |       0.392361 |              50447 |            47287 |          1.06683 |                 2.7 |           4571 |          267895 |       1.70627 |             2.1 |  5.3109e+06  |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2010-10-26 20:00:00 | 2010-10-28 22:00:00  |             50         | Tuesday            | Thursday                 |                      1.5 |                        8 |          0.0273162 | Rare                |
|   2012 |       6 | Minnesota    | MN            | MRO           | East North Central |            -0.1 | normal             | severe weather     | thunderstorm            |               nan |              2550 |              nan |                68200 |       11.79 |        9.25 |        6.71 |          9.19 | 1.85152e+06 | 1.94117e+06 | 1.99303e+06 |   5.78706e+06 |      31.9941 |      33.5433 |      34.4393 |     2.31734e+06 |          278466 |           11010 |       2.60681e+06 |        88.8954 |        10.6822 |       0.422355 |              51598 |            48156 |          1.07148 |                 0.6 |           5364 |          277627 |       1.93209 |             2.2 |  5.38044e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2012-06-19 04:30:00 | 2012-06-20 23:00:00  |             42.5       | Tuesday            | Wednesday                |                      0.1 |                       61 |          0.0138857 | Very Common         |
|   2015 |       7 | Minnesota    | MN            | MRO           | East North Central |             1.2 | warm               | severe weather     | nan                     |               nan |              1740 |              250 |               250000 |       13.07 |       10.16 |        7.74 |         10.43 | 2.02888e+06 | 2.16161e+06 | 1.77794e+06 |   5.97034e+06 |      33.9826 |      36.2059 |      29.7795 |     2.37467e+06 |          289044 |            9812 |       2.67353e+06 |        88.8216 |        10.8113 |       0.367005 |              54431 |            49844 |          1.09203 |                 1.7 |           4873 |          292023 |       1.6687  |             2.2 |  5.48959e+06 |          73.27 |       15.28 |           2279 |      1700.5 |           18.2 |            2.14 |          0.6 |    91.5927 |         8.40733 |            5.47874 | 2015-07-18 02:00:00 | 2015-07-19 07:00:00  |             29         | Saturday           | Sunday                   |                      1.2 |                       27 |          0.0737537 | Uncommon            |


### Univariate Analysis

After cleaning the dataframe, we analyzed the distribution of outage duration in hours and anomaly level.

#### Distribution of Duration in Hours

Here is a histogram on the distribution of the duration in hours of power outages in U.S.

Since the data is right-skewed, most outages are short-lived, but there are a few that last much longer and can skew averages. For a thorough risk assessment, an energy company should use median or a range of percentiles for more representatie comparison, and analyze outliers separately to understand the causes behind prolonged outages and to identify any patters in these extreme cases.

<iframe src="figures/duration_dist.html" width=800 height=600 frameBorder=0></iframe>

#### Distribution of Anomaly Level

Here is a histogram on the distribution of the anomaly level in our dataset.

This figure suggests that the distribution could be approximated as a Gaussian distribution, slightly skewed to the right, with a break between 1.7 and 2. We would say that the figure centered around -0.35, meaning that most power outages have anomaly level at -0.35.

<iframe src="figures/anomaly_dist.html" width=800 height=600 frameBorder=0></iframe>

### Bivariate Analysis

To identify potential correlations within the data, bivariate analysis is essential for exploring and understanding the relationship between two variables. In the figure, we used boxplot to illustrate the relationship between cause category and outrage duration in hours.

<iframe src="figures/cause_duration.html" width=800 height=600 frameBorder=0></iframe>

From the plot, ***fuel supply emergency*** has the widest range of outage durations, indicating high variablity, with some outages lasting significantly longer than those caused by other factors. ***Severe weather*** is another major cause, with a higher median outage duration than most categories but fewer outliers that *fuel supply emergency*. Overall, the plot indicates that the cause of an outage is a strong indicator of its potential duration and variability.

### Interesting Aggregates

To construct a more reliable analysis, we created a pivot table illustrating the average outage duration for each combination of climate severity category and cause of outage. This can help us to identify if certain causes of outages are more prevalent or more severe under certain climate conditions.

| Severity Category   |   equipment failure |   fuel supply emergency |   intentional attack |   islanding |   public appeal |   severe weather |   system operability disruption |
|:--------------------|--------------------:|------------------------:|---------------------:|------------:|----------------:|-----------------:|--------------------------------:|
| Common              |             376.364 |                19424    |              404.083 |     220.25  |         1498.38 |          4649.85 |                        1010.15  |
| Rare                |            1080     |                 5370    |               44     |      15     |         1844    |          3081.37 |                         280.444 |
| Uncommon            |             400.5   |                 5163.75 |              541.161 |     163     |          293.2  |          3678.8  |                         425.353 |
| Very Common         |            3177.43  |                14217.7  |              433.684 |     225.083 |         1526.19 |          3699.91 |                         755.38  |

For example, we found that 'equipment failure' as a cause has a higher average outage duration in the 'Very Common' climate severity category.

Also, a sample grouped table is shown below. By looking at the sum of outage durations per year per state, we can analyze trends over time, which could reveal whether there has been an improvement in utility.

|   YEAR |   Alabama |   Alaska |    Arizona |   Arkansas |   California |   Colorado |   Connecticut |   Delaware |   District of Columbia |    Florida |   Georgia |   Hawaii |    Idaho |   Illinois |    Indiana |     Iowa |   Kansas |    Kentucky |   Louisiana |       Maine |   Maryland |   Massachusetts |   Michigan |   Minnesota |   Mississippi |   Missouri |   Montana |    Nebraska |     Nevada |   New Hampshire |   New Jersey |   New Mexico |   New York |   North Carolina |   North Dakota |       Ohio |   Oklahoma |    Oregon |   Pennsylvania |   South Carolina |   South Dakota |   Tennessee |     Texas |      Utah |   Vermont |   Virginia |   Washington |   West Virginia |   Wisconsin |   Wyoming |
|-------:|----------:|---------:|-----------:|-----------:|-------------:|-----------:|--------------:|-----------:|-----------------------:|-----------:|----------:|---------:|---------:|-----------:|-----------:|---------:|---------:|------------:|------------:|------------:|-----------:|----------------:|-----------:|------------:|--------------:|-----------:|----------:|------------:|-----------:|----------------:|-------------:|-------------:|-----------:|-----------------:|---------------:|-----------:|-----------:|----------:|---------------:|-----------------:|---------------:|------------:|----------:|----------:|----------:|-----------:|-------------:|----------------:|------------:|----------:|
|   2000 |  74.9     |        0 |    1.1     |    0       |       0      |    0       |       0       |   0        |                  0     |    0       |    0      |   0      |  0       |   20       |   0        |   0      |   0      |   0         |    0        |   0         |    0       |         0       |     0      |     0       |     0         |   0        |  0        |   0         |  0         |         0       |      0       |      0       |   11.35    |         229.5    |              0 |   0        |     0      |   0       |        0       |         234      |              0 |      0      |   45.15   |   0       | 0         |    0       |      0       |       0         |      0      |   0       |
|   2001 |   0       |        0 |    0       |    0       |      87.0667 |    0       |       0       |   0        |                  0     |    0       |    0      |   0      |  0       |    0       |   0        |   0      |   0      |   0         |    0        |   0         |    0       |         1.71667 |     0      |     0       |     0         |   0        |  0        |   0         |  0         |         0       |      0       |      0       |    8.23333 |           0      |              0 |   0        |     0      |   0       |        0       |           0      |              0 |      0      |  195.783  |   0       | 0         |    4.01667 |      0       |       0         |      0      |   0       |
|   2002 |   0       |        0 |    0       |  136       |     252.383  |    0       |      98       |   0        |                  0     |    3.83333 |    0      |   0      |  0       |    0       |   0        |   0      |   0      |   0         |    0        |   0         |    0       |         0       |    60      |     0       |     0         | 257        |  0        |   0         |  0         |         0       |      0       |      0       |    0       |           0      |              0 |   0        |   198      |   0       |       58.5     |           0      |              0 |      0      |    0      |   0       | 0         |   44.85    |      0       |       0         |      0      |   0       |
|   2003 |   0       |        0 |    2.25    |    0       |     717.667  |    0       |       0       |   0        |                241.667 |   13.6     |    0      |   0      | 25.8     |   24       |   0        |   0      |   0      |   0         |    0        |   0         |  518.967   |         1.91667 |  1193.43   |     0       |     0         |   0        |  0        |   0         |  0         |         0       |      0       |      0       |  190.567   |          82.1833 |              0 | 160.283    |     0      |   0       |       72.3     |           0      |              0 |      0      |  173.817  |   0       | 0         |    4.1     |    108       |       0         |     36.3167 |   0       |
|   2004 |   0       |        0 | 1650.97    |    0       |     181.883  |    0       |       0       |   0        |                  0     | 1275.43    |   81.5    |   0      |  1.58333 |   47.5     |  87.5      |   0      |   0      |   0         |  198.483    |   0         |   65.8333  |         6.23333 |   454.333  |     0       |     0         |   0        |  0        |   0.0666667 |  0         |         0       |      0       |      0       |  170       |          31.5    |              0 | 164        |   111.417  |   0       |       38.65    |          66.1333 |              0 |      0      |  365.9    |   0       | 0         |   10.4     |    160.417   |       0         |      0      |   0       |
|   2005 |   0       |        0 |    0       |    2.1     |     234.367  |    0       |       0       |   0        |                  0     | 1498.33    |   54.0833 |   0      |  0       |    2.48333 | 265.833    |   0      | 234      |   0         |  488.933    |   0         |  215.083   |         0       |   597.983  |   214       |     0         |   0        |  0        |   0         |  0         |         0       |      0       |      0       |   54       |          28      |              0 | 434        |     0      |   0       |       57.85    |           0      |              0 |      0      |  255.5    |   0       | 0         |    0       |      0       |       0         |    123.5    |   0       |
|   2006 |   0       |        0 |    0       |    6.9     |     335.717  |   42.9833  |       2.41667 |   0        |                  0     |    0       |    2      |  44.7667 |  0       |   91.5     |  67.4167   |   0      |   0      |   0         |    0        |   3.3       |  183.133   |         0       |   225      |     0       |     0         |   0        |  0        | 160         |  0         |         0       |    131       |      0       |  672.933   |          19.5    |              0 |  30.6667   |     5      | 162.383   |      357.1     |           0      |              0 |      0      |   42.5167 |   0       | 0         |   28.75    |   1047.5     |       0         |      0      |   0       |
|   2007 |   0       |        0 |    0       |    0       |     246.783  |    0       |      71       |   0        |                  0     |    0       |    0      |   0      |  0       |  166.817   |   0.783333 |  23.1833 | 227.5    |   0         |    0        | 103.833     |  147.5     |         0       |   516.917  |     0       |     0         |  34.5      |  0        |   0         |  0         |        11       |      0       |      0       |   56.6167  |           0      |              0 |   0        |   232.867  |   0       |        1.3     |           0      |              0 |      0      |   99.0667 |   4.71667 | 0         |   20.0167  |    138.6     |       0         |      0      |   0       |
|   2008 |   0       |        0 |    0       |    0       |     686.917  |    0       |      45.8333  |   0        |                  0     |  196.9     |   32.5    |  22.7833 |  0       |  109       | 246.933    | 144.5    |   0      | 592.983     | 1339.8      |  88.7       |   96.5     |         0       |   755.7    |     0       |     0         |  14.5      |  0        |   1         |  0         |         0       |    189.783   |      0       |  407.833   |          36.7167 |              0 | 502.433    |     0      |   3.33333 |      371.217   |           0      |              0 |     14      | 1707.18   |  14.5167  | 0         |   47       |      4.13333 |       0         |      0      |   0       |
|   2009 |   0       |        0 |   70       |  216.233   |      90.2833 |   45.1667  |      76       |   0        |                  0     |   23.65    |   20      |   0      |  0       |  164.817   | 805.3      |   0      |   0      | 459.567     |   35.7      |  49.8167    |   47.5833  |         0       |  1845.23   |     0       |     0         | 161.667    |  0        |   0         |  0         |         0       |      0       |      0       |    0       |          24.3333 |              0 | 104.883    |     0      |   0       |       86       |           0      |              4 |     15.5    |  264.567  |   0.25    | 0         |   66.2833  |     20.0667  |       0         |      0      |   1.76667 |
|   2010 |   0       |        0 |    0       |    3.43333 |     603.517  |    6.51667 |      57.35    |   0.833333 |                185.517 |    6.2     |    0      |   0      |  0       |  174.667   | 428        |   0      |   0      |   0         |  269.317    |  11.2167    |  179       |         0       |   503.983  |   130.5     |     0         |   0        |  0.216667 |   2.65      |  0         |         0       |    528.983   |      0       |  892.35    |           0.7    |              0 |  30.7167   |   122.467  |   0       |      681.4     |           0      |              0 |      0      |  173.783  |  37.9167  | 0         |   62.9167  |     86       |       0         |     14.4667 |   1.01667 |
|   2011 |   0       |        0 |   12.8333  |  174.567   |     697.717  |   38.7333  |      27.5167  |   7.61667  |                151.417 |   43.35    |   69.1833 |   2.9    | 20.5     |  151.417   | 153.367    |   0      |  15.2167 |  50.3167    |   28.2833   |   3.25      |  365.983   |         8.01667 |   877.533  |    73.0333  |     5         | 260.867    |  0        |   0         | 57.15      |         7.83333 |    608.417   |      0       | 1580.83    |         164.1    |             12 | 210.55     |   138.383  |  32.7333  |     1000.18    |          40.25   |              0 |    223.783  |  406.617  | 100.667   | 0.85      |   53.3     |    197.933   |       0         |      0      |   0       |
|   2012 |   0       |        0 |    0       |   19.8333  |     239.767  |    5.16667 |       1       |   0.783333 |                134.617 |    4       |    1.8    |   0      |  0       |   92.6667  |  41.5167   |  36.0167 |   0      |   0         |  168.717    |  19.7833    |  313.333   |       118.15    |   282.75   |    42.5     |     0         |   0        |  0        |   0         |  1.81667   |        45.1333  |    735.267   |      5.23333 | 1005.28    |          27.0833 |              0 | 330.533    |    41.2    |   0       |      689       |           0      |              0 |     66      |  154.933  |   0       | 4.43333   |  188.067   |    139.867   |     447.583     |      0      |   0       |
|   2013 |   1.28333 |        0 |  139.2     |    7.35    |     163.167  |    2.95    |       5.53333 |   2        |                  3.15  |    4.31667 |    0      |   0      |  0       |   36.25    | 305.783    |  27.2833 |   0      |   0         |    3.78333  |  49.2       |   19.5     |       129.417   |   668.133  |   189.35    |     0         |  93.5      |  1.55     |   0         |  4.56667   |         1.28333 |      0       |     13.4833  |  227.4     |          60.55   |              0 |  27.75     |   167.367  |  53.0167  |        6.33333 |           0      |              0 |     57.2833 |  851.15   |   1.85    | 0.0166667 |   60.4333  |     58.35    |       0         |     23.4667 |   0.55    |
|   2014 |   6.5     |        0 |   14.8833  |    4.75    |      30.8833 |    0       |       0       |   1.38333  |                  0.9   |    0       |   86.8167 |   0      |  0       |   92.5167  |  60.7167   | 408.183  |   0      |   0.0166667 |    0.866667 |   0         |   50.6167  |         0       |   156.25   |     1.01667 |     0.516667  |   0.416667 |  0.933333 |   0         |  0         |         0       |      5.88333 |      0       | 1441.02    |         185.2    |              0 |  10.5667   |     0      |  16.05    |      162.05    |          77.6167 |              0 |    134.6    |  111.617  |   0.3     | 0.0166667 |    0       |     28.8667  |      16.6667    |   2280.47   |   0       |
|   2015 |  13.3833  |        0 |    2.03333 |   59.8167  |     102.883  |    1.75    |       0       |  70.9      |                  0     |    0       |   33.3167 |   0      |  0       |    1.5     |   2        |   0      |  33.85   |   1.8       |    0        |   0.0166667 |    5.06667 |         0       |   229.917  |    31.5833  |     0.0833333 |   1.08333  |  0        |   0         |  0.0166667 |         0       |    170.2     |      0       |    7.01667 |          32.5    |              0 |   0.116667 |    77.0333 |   6.05    |       28.8667  |           0      |              0 |     27.1833 |  457.8    |   4.76667 | 0         |   40.5833  |    209.133   |       0.0166667 |      0      |   0       |
|   2016 |   0       |        0 |    3.78333 |    0       |     828.917  |   66.9833  |       0       |  13.1      |                  0     |    1.38333 |    0      |   0      |  7.4     |    0       |   0        |   0      |   0      |   0         |   53        |   0         |   27.8833  |        17.8     |    30.2167 |     0       |     0         |  19.9833   |  0        |   0         |  1         |         0       |      1.28333 |      0       |  316.35    |          25.3667 |              0 |   0        |    13.2667 |  45.8833  |        7.36667 |           0      |              0 |      0      |  193.417  |   6       | 0         |    0       |     37.2167  |       0         |     26.75   |   0       |

These states show notable figures in certain years (e.g., Florida in 2005 and Arizona in 2004), which might correspond to specific large-scale outages or reporting changes.

---

## Assessment of Missingness

[Back to Top](#table-of-contents)

To explore whether missing values are Missing At Random (MAR) or Not Missing At Random (NMAR), we typically look for patterns in the missing data. If the presence of missing data is related to observed data, it might be MAR; if it's related to unobserved data, it might be NMAR. We visualized the number of missing values in each column. This allows us to focus more specifically on exploring the patterns of missingness within certain columns.

<iframe src="figures/missing_values_proportion.html" width=800 height=600 frameBorder=0></iframe>

Assuming significant missing values refer to a missing proportion greater than 0.2, the chart indicates several columns with a significant number of missing values: `CAUSE.CATEGORY.DETAIL`, `HURRICANE.NAMES`, `DEMAND.LOSS.MW`, `CUSTOMERS.AFFECTED`.

### NMAR Analysis

One of the column in our dataset with missing values that is possibly NMAR is the `CUSTOMERS.AFFECTED` column. In large-scale outages, especially those caused by natural disasters or catastrophic events, the precise number of customers affected can be challenging to ascertain. Utilities may face difficulties in:

- *Real-time Data Collection*: The rapid onset of a crisis may overwhelm systems designed to monitor outages, leading to undercounting.
- *Infrastructure Damage*: The damage to infrastructure may prevent accurate data transmission regarding the number of customers affected.

In addition, people might be reluctant to report the full extent of an outage's impact due to:

- *Public Relations*: Admitting to large numbers of customers affected can lead to negative public perception and damage to the utility's reputation.
- *Market Sensitivity*: Shareholders and investors may react negatively to news of widespread outages, affecting stock prices and investment.

However, the `CUSTOMERS.AFFECTED` column could be MAR if we have internal reports or assessments which contain more accurate data that were not released publicly. In addition, detailed maps of outage locations and affected areas could help estimate the number of customers affected based on population density.


### Missingness Dependency

Since we are interested in the characteristics of major power outages, we belive that one possible feature could be represented by the `CAUSE.CATEGORY.DETAIL` column. Intuitively, the `CAUSE.CATEGORY.DETAIL` column depends on the `CAUSE.CATEGORY` column, which, notably, does not have any missing values. Hence, we want to know whether the missingness of `CAUSE.CATEGORY.DETAIL` depends on other columns or not. To validate our hypothesis, we conducted pertumatition tests to examine the relationship. Specifically, we chose to investigate the dependency of the missingness of `CAUSE.CATEGORY.DETAIL` on two columns: `NERC.REGION` and `U.S._STATE`. 

1.`CAUSE.CATEGORY.DETAIL` and `NERC.REGION` (NMAR)

**Null Hypothesis**: The missingness of `CAUSE.CATEGORY.DETAIL` *does not* depend on `NERC.REGION`.

**Alternative Hypothesis**: The missingness of `CAUSE.CATEGORY.DETAIL` depends on `NERC.REGION`.

Since `NERC.REGION` is categorical, we should implement *total variance distance(TVD)* in our permutation test. 

We created a new column `is_missing` indicating the missingness status of the `CAUSE.CATEGORY.DETAIL` and then shuffled the `NERC.REGION` column for our permutation. 

Below shows the empirical distribution of our test statistics in 10,000 permutations, the red line indicates the observed test statistic.

<iframe src="figures/NERC_CAUSE.html" width=800 height=600 frameBorder=0></iframe>

Since the p value after running permutation is test is 0.3587 which is greater than our chosen significance level of 5%, we failed to reject the null hypothesis that the missingness of `CAUSE.CATEGORY.DETAIL` *does not* depend on `NERC.REGION`. Therefore, we conclude that **it is highly possible that the missingness of `CAUSE.CATEGORY.DETAIL` does not depend on the `NERC.REGION` column**.

2.`CAUSE.CATEGORY.DETAIL` and `U.S._STATE` (MAR)

**Null Hypothesis**: The missingness of `CAUSE.CATEGORY.DETAIL` *does not* depend on `U.S._STATE`.

**Alternative Hypothesis**: The missingness of `CAUSE.CATEGORY.DETAIL` depends on `U.S._STATE`.

Since `U.S._STATE` is categorical, we should also implement *total variance distance(TVD)* in our permutation test. 

Similarly, we created a new column `is_missing` indicating the missingness status of the `CAUSE.CATEGORY.DETAIL` and then shuffled the `U.S._STATE` column for our permutation. 

Below shows the empirical distribution of our test statistics in 10,000 permutations, the red line indicates the observed test statistic.

<iframe src="figures/STATE_CAUSE.html" width=800 height=600 frameBorder=0></iframe>

Since the p value after running permutation is test is 0.0049 which is smaller than our chosen significance level of 5%, we rejected the null hypothesis that the missingness of `CAUSE.CATEGORY.DETAIL` *does not* depend on `U.S._STATE`. Therefore, we conclude that **it is highly possible that the missingness of `CAUSE.CATEGORY.DETAIL` depends on the `U.S._STATE` column**.

---

## Hypothesis Testing

[Back to Top](#table-of-contents)

To unravel the characteristics of major power outages with heightened severity, we have formulated a research question that focuses on the relationship between outage severity, as indicated by the number of customers affected. and total electricity sales. For our analysis, we've posited the following hypotheses:

**Null Hypothesis**: The severity of major power outages ***is not related*** to the `TOTAL.SALES`.
**Alternative Hypothesis**: The severity of major power outages ***is related*** to the `TOTAL.SALES`.

To test these hypotheses, we've implemented a permutation test that used to investigate the relationship between the severity of power outages and total electricity sales. The method does not rely on any assumptions about the distribution of the data, making it robust to data that do not follow a normal distribution. Since the distribution is numerical, our test statistic is the **absolute difference in the average number of customers affected for outages above and below the median of `TOTAL.SALES`**. By randomly shuffling the `TOTAL.SALES` data and recalculating the difference in means 10,000 times, we simulate the distribution of our test statistic under the null hypothesis.

Below is the empirical distribution of our test statistic, derived from these permutations, and the red line indicates the observed test statistic:

<iframe src="figures/Customer_Sales.html" width=800 height=600 frameBorder=0></iframe>

**Significance Level**: 5%

Since the p value after running permutation is test is 0.0 which is smaller than our chosen significance level of 5%, we **rejected the null hypothesis**, which states that the severity of major power outages ***is not related*** to the `TOTAL.SALES`. 

Our statistical analysis, through the application of permutation tests, indicates that there is a statistically relationship between TOTAL.SALES and the severity of power outages, as measured by the number of customers affected. However, it's important to acknowledge that this finding doesn't confirm a causal relationship. To move closer to establishing causality, additional research incorporating a broader range of data and potentially experimental designs would be essential.

---

## Framing a Prediction Problem

[Back to Top](#table-of-contents)

Building on our comprehensive analysis of the factors affecting power outages and their impacts, we delve deeper into the nuanced interactions between these elements. Specifically, this study aims to ***discern whether a major power outage was caused by "severe weather"***—a classification task that entails predicting a binary outcome, leveraging the wealth of data available on past incidents. Understanding the primary causes of outages is paramount for devising more effective mitigation strategies, enhancing grid resilience, and ensuring the reliable delivery of electricity—a resource integral to the fabric of modern society. Accurate predictions can empower utility providers and policymakers with the insights needed to preemptively address vulnerabilities and manage resources adeptly.

The focal point of our prediction is a binary classification of the `CAUSE.CATEGORY`, specifically determining if an outage was caused by *"severe weather"* as opposed to other causes. This binary approach streamlines our analysis, allowing for a targeted investigation into one of the most common and important causes of outages.

Our evaluation strategy employs accuracy as the primary metric, complemented by a confusion matrix and a comprehensive classification report. The use of accuracy helps us directly assess the proportion of correct predictions made by the model, offering an intuitive understanding of its effectiveness.

### At the "Time of Prediction"

In our predictive model for power outage causes, we meticulously select features that are known or estimable at the time of prediction, ensuring the model's utility in real-world scenarios. For instance:

- **Price and Sales Related Features** (`RES.PRICE`, `COM.PRICE`, `IND.PRICE`, `TOTAL.PRICE`, `RES.SALES`, `COM.SALES`, `IND.SALES`, `TOTAL.SALES`):  These features reflect the economic conditions leading up to an outage, which can be a factor in its cause.
- **Percentage and Customer Count Features** (`RES.PERCEN`, `COM.PERCEN`, `IND.PERCEN`, `RES.CUSTOMERS`, `COM.CUSTOMERS`, `IND.CUSTOMERS`, `TOTAL.CUSTOMERS`): These features represent the distribution and number of customers impacted by outages.
- **Economic Indicators** (`PC.REALGSP.STATE`, `PC.REALGSP.USA`, `PC.REALGSP.REL`, `PC.REALGSP.CHANGE`, `UTIL.REALGSP`, `TOTAL.REALGSP`, `UTIL.CONTRI`, `PI.UTIL.OFUSA`): Economic health could be related to infrastructure investment and maintenance, which in turn could impact the likelihood of different outage causes.
- **Demographic and Geographic Features** (`POPULATION`, `POPPCT_URBAN`, `POPPCT_UC`, `POPDEN_URBAN`, `POPDEN_UC`, `POPDEN_RURAL`, `AREAPCT_URBAN`, `AREAPCT_UC`, `PCT_LAND`, `PCT_WATER_TOT`, `PCT_WATER_INLAND`): Population density and urbanization can influence the complexity of the power network and the susceptibility to certain types of outages.

The above columns are all available at the time of the prediction.

---

## Baseline Model

[Back to Top](#table-of-contents)

Our baseline model employs a *HistGradientBoostingClassifier*, a robust machine learning algorithm suited for classification tasks, to discern whether a power outage's cause was 'severe weather'. The model considers a wide array of features—**21 quantitative variables** such as `RES.PRICE`, `COM.PRICE`, and `POPULATION`, which reflect economic and demographic aspects without requiring preprocessing. Furthermore, temporal data from `OUTAGE.START` and `OUTAGE.RESTORATION` are treated as *ordinal* and undergo a transformation to extract year, month, day, and hour components, ensuring temporal nuances are captured. The response variable `CAUSE.CATEGORY`, are considered *nominal*.

### Feature Engineering

To increase the performance of our baseline model, we did feature engineering on some columns.

- **Column Transformation**: The `ColumnTransformer` applies the helper_function to the specified datetime columns while passing through the other columns unchanged. This step ensures that the model can use both the extracted time features and any other relevant features in the dataset.

- **Imputation**: The `SimpleImputer` is used to fill in any missing values in the dataset. This is important because machine learning models require complete data to make accurate predictions.

To facilitate model learning, the data were shuffled to ensure randomness and split into training and testing sets with the first 1000 rows for training and the rest for testing. Then, we fit the model to the training data set and used it for evaluation. 

Below is our baseline model:

```py
Pipeline(steps=[('ct',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('timestrap',
                                                  FunctionTransformer(func=<function helper_function at 0x2880ea9d0>),
                                                  ['OUTAGE.START',
                                                   'OUTAGE.RESTORATION'])])),
                ('impute', SimpleImputer()),
                ('gbdt', HistGradientBoostingClassifier())])
```

This model actually got an overall accuracy of 0.79, indicating that it correctly predicts the cause of power outages 79% of the time. However, accuracy does not tell everything. We can also take look at other indicators such as precision, recall, and F1-score for both classes. 

The confusion matrix for this base model is illustrated as below:

<iframe src="figures/confusion_matrix1.html" width=800 height=600 frameBorder=0></iframe>

And below is the classfication report of our baseline model:

|            | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| False      | 0.75      | 0.81   | 0.77     | 260     |
| True       | 0.83      | 0.76   | 0.80     | 274     |
|            |           |        |          |         |
| Accuracy   |           |        | 0.78     | 534     |
| Macro Avg  | 0.78      | 0.79   | 0.78     | 534     |
| Weighted Avg | 0.78    | 0.79   | 0.78     | 534     |

We can observe that the scores for both classes ('False' and 'True') are relatively close, with scores around 0.77 to 0.80. These scores suggest a reasonably good baseline performance. However, there's still room for improvement, as we strive for higher precision and recall values which would indicate a more accurate and reliable model.

---

## Final Model

[Back to Top](#table-of-contents)

Our final model was meticulously crafted to provide the most accurate predictions possible for the causes of severe weather-related power outages. It incorporates some categorical variables, which are dependent on other variables. We want to see if these categorical variables can increase the accuracy of our final model.

### Feature Selection

- **U.S._STATE and NERC.REGION**: These features introduce a geographic dimension to the model, considering that different states may have varying patterns that can influence the likelihood of outages.

- **CLIMATE.CATEGORY and CLIMATE.REGION**: Including climate-related variables allows the model to account for environmental factors that significantly impact power outage causes and severities.

For those categorical features, they are one-hot encoded to convert these nominal variables into a format that can be provided to the machine learning algorithms, enhancing the model's interpretability of geographical and climate influences.

Also, we maintained the same feature engineering methods as the baseline model, such as transformations for time features, imputation, and column transformers.

### Modeling Selection

Selected Model: HistGradientBoostingClassifier

- The model operates by converting continuous features into discrete bins, a process that significantly enhances computational efficiency and allows the model to scale with large datasets—a crucial capability given the diverse and voluminous data involved in power outage analysis. Through its gradient boosting framework, the algorithm iteratively builds a series of decision trees, each designed to correct the residuals or mistakes of the previous trees. This sequential correction process, grounded in gradient descent, refines the model's accuracy with each step.

Our model benefits immensely from this approach. The ensemble technique, which aggregates predictions from multiple trees, ensures a robust final prediction that accounts for various nuances and patterns in the data. This is particularly advantageous for our task, where factors influencing power outages can be subtle and multifaceted, ranging from geographical and temporal to climatic variables.

### Finding Hyperparameters

Hyperparameter tuning for this classifier is a nuanced process, aimed at refining the model to achieve a delicate balance between learning from the data and generalizing well to unseen data. 

- **Max Depth**: It is the maximum number of levels allowed in each decision tree. Setting the max depth too low can prevent the model from capturing complex patterns in the data, leading to underfitting. Conversely, a very high max depth can result in overly complex models that overfit the training data.

To choose the best hyperparameter, we implemented GridSearchCV with a 5-fold cross-validation. It carefully tests several combinations of hyperparameters to determine which combination is the most effective. By utilizing a range of values for max_depth (from 10 to 20, stepping by 2), we allow GridSearchCV to assess how different tree depths affect performance. 

Below is our final model:

```py
Pipeline(steps=[('ct',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('timestrap',
                                                  FunctionTransformer(func=<function pl_final.<locals>.helper_function at 0x288e3c670>),
                                                  ['OUTAGE.START',
                                                   'OUTAGE.RESTORATION']),
                                                 ('onehot',
                                                  OneHotEncoder(handle_unknown='ignore'),
                                                  ['U.S._STATE', 'NERC.REGION',
                                                   'CLIMATE.CATEGORY',
                                                   'CLIMATE.REGION'])])),
                ('impute', SimpleImputer()),
                ('gbdt', HistGradientBoostingClassifier(max_depth=18))])
```

### KFold Cross Validation

We employed a KFold cross-validation strategy with 10 splits, a rigorous method ensuring that every observation in our dataset is used for both training and validation. This technique provides a comprehensive assessment of our model's predictive capabilities across different subsets of data.

After iteratively training and testing the model on different subsets of the dataset, it returns an average accuracy of **0.8455097190391306**. Hence, we can conclude that our final model has an average accuracy of approximately 83%, which is a 5% improvement over the baseline model. The final model’s enhanced feature set, coupled with a rigorous hyperparameter optimization process, delivers a more reliable and robust tool for predicting the causes of major power outages, thereby improving upon the baseline model's initial groundwork.

### Comparison with Baseline Model

We fit the final model to the same training data set. Compared to the baseline model, we have higher accuracy scores which shows that our final model has improved.

The confusion matrix for the final model is illustrated as below:

<iframe src="figures/confusion_matrix2.html" width=800 height=600 frameBorder=0></iframe>

And below is the performance report of our final model:

|            | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| False      | 0.91      | 0.85   | 0.88     | 75      |
| True       | 0.87      | 0.92   | 0.89     | 78      |
|            |           |        |          |         |
| Accuracy   |           |        | 0.89     | 153     |
| Macro Avg  | 0.89      | 0.89   | 0.89     | 153     |
| Weighted Avg | 0.89    | 0.89   | 0.89     | 153     |

In conclusion, our final model has shown significant improvements in all metrics, including accuracy, recall, and f1 score. Therefore, it indicates that our final model has made great progress compared to the baseline model. Additionally, since our model's predictive accuracy has reached over 80%, we believe that our model's predictive performance is very well.

---

## Fairness Analysis

[Back to Top](#table-of-contents)

Given the different climate types present in the various states of the United States, we hypothesize that climate may impact the accuracy of our model. We roughly categorize the `U.S._STATE` into a temperate and subtropical group, which includes states with a humid continental climate in the Northeast and Midwest, as well as states with a humid subtropical climate in the South; and an arid, semi-arid, and Mediterranean climate group, which includes states with arid and semi-arid climates in the West and Southwest, as well as California with its Mediterranean climate and the Pacific Northwest with its oceanic climate.

```py
temperate_subtropical_states = ['Massachusetts', 'Connecticut', 'Rhode Island', 'New Hampshire', 'Vermont',
                                'Maine', 'Ohio', 'Michigan', 'Indiana', 'Illinois', 'Wisconsin', 'Minnesota',
                                'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas',
                                'Virginia', 'North Carolina', 'South Carolina', 'Georgia', 'Florida', 'Kentucky',
                                'Tennessee', 'Alabama', 'Mississippi', 'Arkansas', 'Louisiana', 'West Virginia',
                                'New York', 'New Jersey', 'Pennsylvania', 'Maryland', 'Delaware',
                                'District of Columbia']

arid_mediterranean_states = ['Nevada', 'Arizona', 'Utah', 'New Mexico', 'Texas', 'Oklahoma', 'Colorado',
                             'Washington', 'Oregon', 'Idaho', 'Montana', 'Wyoming', 'California', 'Alaska',
                             'Hawaii']
```

**Group 1**: States with Temperate and subtropical climate

**Group 2**: Stetes with Arid, semi-arid, and Mediterranean climate

**Null hypothesis**: Our model is fair. The accuracy in states with temperate and subtropical climates (Group 1) and states with arid, semi-arid, and Mediterranean climates (Group 2) is roughly the same, and any differences are due to chance.

**Alternative hypothesis**: Our model is not fair. The accuracy in states with temperate and subtropical climates (Group 1) and states with arid, semi-arid, and Mediterranean climates (Group 2) is not the same.

**Test statistic**: The difference in accuracy rates between states with temperate and subtropical climates (Group 1) and states with arid, semi-arid, and Mediterranean climates (Group 2).

**Significance level**: 0.05

Then we conducted a permutation test to examine whether there's a statistically significant difference in predictions between two groups of states, classified as either 'Temperate/Subtropical' or 'Arid/Mediterranean', based on their climate. We shuffled the accuracy and then computed the difference in mean accuracy for the shuffled data.

Below shows the empirical distribution of our test statistics in 10,000 permutations, and the red line indicates the observed test statistic:

<iframe src="figures/fairness.html" width=800 height=600 frameBorder=0></iframe>

Since the p value after running permutation is test is 0.0 which is smaller than our chosen significance level of 5%, we **rejected the null hypothesis**, which states that the model is fair. One possible reason is that different climate types indeed lead to differences in prediction accuracy, but more specific reasons require further analysis. 

---

## References

[Back to Top](#table-of-contents)

Mukherjee, S., Nateghi, R., & Hastak, M. (2018). Data on major power outage events in the continental U.S. Data in Brief, 19, 2079–2083. https://doi.org/10.1016/j.dib.2018.06.067
