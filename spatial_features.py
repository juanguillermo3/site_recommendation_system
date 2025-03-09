"""
title: Spatial Features
description: Provides a dataframe with zip level spatial features from the US
"""
import os
import pandas as pd
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
APP_HOME = os.getenv("APP_HOME")

if not APP_HOME:
    raise ValueError("APP_HOME environment variable not found. Please set it in the .env file.")

# Change working directory
os.chdir(APP_HOME)

# Load the dataset
zip_features = pd.read_csv('data/income_new.csv')

# Standardizing ZIP code
zip_features["zip_code"] = zip_features["ZIP"].astype(str)

# Selecting relevant columns
zip_features = zip_features[
    [col for col in zip_features.columns if re.search("(zip_code)|(Hous)|(Fam)|(Marr)|(Non)", col)]
]

# Display DataFrame information
print(zip_features.info())

# Dictionary for term replacements
abbreviations = {
    'Households': 'Hs',
    'Less Than': 'LT',
    'Nonfamily Households': 'NFHs',
    'Married-Couple Families': 'MCF',
    'Families': 'Fam',
    'Median Income (Dollars)': 'MedInc',
    'Mean Income (Dollars)': 'MeanInc',
    'Income in the Past 12 Months': 'Inc12M',
    ' to ': '-',
    '$': '',
}

# Function to replace all terms and abbreviate numbers
def abbreviate_names(name):
    for long, short in abbreviations.items():
        name = name.replace(long, short)
    name = name.replace('100,000', '100k')
    name = name.replace('150,000', '150k')
    name = name.replace('200,000', '200k')
    return name

# Apply abbreviation function
zip_features.columns = [abbreviate_names(col) for col in zip_features.columns]

# Save the processed file
#zip_features.to_csv("data/processed_income_features.csv", index=False)
#print("Processed file saved.")
