# Time Series Analysis Toolkit

## Overview

This toolkit provides a comprehensive suite of Python utilities for performing advanced feature engineering and data preparation in time series analysis. Designed to facilitate the analysis of temporal data, it incorporates a range of techniques to enhance model accuracy and insight generation. The toolkit is especially useful for projects requiring detailed consideration of time-dependent patterns, calendar events, and historical data augmentation.

## Key Features

- **Time Series Feature Engineering**: Implements functions for creating lagged variables, rolling statistics, and other temporal features essential for capturing time-dependent patterns in data.
- **Calendar Event Integration**: Provides utilities for preparing and augmenting data with calendar events, allowing models to account for the impact of holidays and special events on time series metrics.
- **Data Augmentation and Preparation**: Includes tools for efficiently preparing time series data, managing memory usage, and ensuring data is ready for analysis.
- **Tutorial Notebooks**: Comes with tutorial notebooks that demonstrate practical applications of the toolkit's features, offering users a guided experience on leveraging the toolkit for their time series analysis projects.

## Usage

To begin using the toolkit, import the necessary modules in your Python script or Jupyter notebook. For example, to use the feature engineering functionalities:

```python
from ts_feature_engineering import create_lags, create_rolling
```
Refer to the tutorial notebooks ts_helpers1_tutorial.ipynb and feature_engineering_tutorial.ipynb for comprehensive examples on applying the toolkit's features to real-world datasets.

## Author
This toolkit is developed and maintained by Jarred Bultema, PhD