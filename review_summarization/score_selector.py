#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import fractions

# Source: https://stackoverflow.com/questions/39759867/splitting-a-mixed-number-string-from-a-dataframe-column-and-converting-it-to-a-f
def get_sign(num_str):
  """
  Verify the sign of the fraction
  """
  return 1-2*num_str.startswith('-')

def is_valid_fraction(text_str):
  """
  Check if the string provided is a valid fraction.
  Here I just used a quick example to check for something of the form of the fraction you have. For something more robust based on what your data can potentially contain, a regex approach would be better.
  """
  return text_str.replace(' ', '').replace('-', '').replace('/', '').isdigit()

def convert_to_float(text_str):
  """
  Convert an incoming string to a float if it is a fraction
  """
  if isinstance(text_str, str):
    if is_valid_fraction(text_str):
      sgn = get_sign(text_str)
      return sgn*float(sum([abs(fractions.Fraction(x)) for x in text_str.split()]))
    else:
      return pd.np.nan # Insert a NaN if it is invalid text
  else:
    return text_str


if __name__ == "__main__":

  # Exiting the script via code  
  sys.exit("This script is not meant to be run like this, but to give an impression of how the score-normalisation has been done")

  # Read data
  data = pd.read_csv("/home/christianschuler/data/NLP-Project-Review-Summaries/data-sentiment-normalised.csv", encoding='unicode_escape')

  print("Initial distribution of review_score values")
  #pd.set_option('display.max_rows', None)
  print(data['review_score'].value_counts())


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Schema for normalisation:
  #
  # P/N       5*  5     10      4     A-F   1     P/N/N
  # ============================================================================
  # Positive  5*  5/5   10/10   4/4   A     1.0   Positive
  # Positive  5*        9/10                0.9   Positive
  # Positive  4*  4/5   8/10    3/4   B     0.8   Positive
  # Positive  4*        7/10                0.7   Positive
  # Positive  3*  3/5   6/10          C     0.6   Neutral
  # Positive  3*        5/10    2/4         0.5   Neutral
  # Negative  2*  2/5   4/10          D     0.4   Neutral
  # Negative  2*        3/10                0.3   Negative
  # Negative  1*  1/5   2/10    1/4   F     0.2   Negative
  # Negative  1*        1/10                0.1   Negative
  # Negative  1*  0/5   0/10    0/4   E     0.0   Negative
  #
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  # Normalise to 1-5 stars
  data['normalised_score'] = data['review_score']

  # From Letter-Scores to Fractions (strings)
  data.loc[data['normalised_score'] == "A", 'normalised_score'] = "5/5"
  data.loc[data['normalised_score'] == "A+", 'normalised_score'] = "5/5"
  data.loc[data['normalised_score'] == "A-", 'normalised_score'] = "5/5"
  data.loc[data['normalised_score'] == "B+", 'normalised_score'] = "5/5"
  data.loc[data['normalised_score'] == "B", 'normalised_score'] = "4/5"
  data.loc[data['normalised_score'] == "B-", 'normalised_score'] = "4/5"
  data.loc[data['normalised_score'] == "C+", 'normalised_score'] = "4/5"
  data.loc[data['normalised_score'] == "C", 'normalised_score'] = "3/5"
  data.loc[data['normalised_score'] == "C-", 'normalised_score'] = "3/5"
  data.loc[data['normalised_score'] == "D+", 'normalised_score'] = "3/5"
  data.loc[data['normalised_score'] == "D", 'normalised_score'] = "2/5"
  data.loc[data['normalised_score'] == "D-", 'normalised_score'] = "2/5"
  data.loc[data['normalised_score'] == "F+", 'normalised_score'] = "2/5"
  data.loc[data['normalised_score'] == "F", 'normalised_score'] = "1/5"
  data.loc[data['normalised_score'] == "F-", 'normalised_score'] = "1/5"
  data.loc[data['normalised_score'] == "E+", 'normalised_score'] = "1/5"
  data.loc[data['normalised_score'] == "E", 'normalised_score'] = "0/5"
  data.loc[data['normalised_score'] == "E-", 'normalised_score'] = "0/5"

  # Selection of viable values
  viable_values = ["5/5", "4/5", "3/5", "2/5", "1/5", "0/5"]
  data_selected = data.loc[data['normalised_score'].isin(viable_values)]

  print("review_scores after letter to fraction")
  #pd.set_option('display.max_rows', None)
  print(data_selected['normalised_score'].value_counts())

  # From Fraction-Scores (strings) to Number-Scores (float)
  #data_selected['normalised_score'] = data_selected['normalised_score'].apply(lambda frac: convert_to_float(frac))
  
  # From Fraction-Scores (strings) to Number-Scores (int)
  data_selected.loc[data['normalised_score'] == "5/5", 'normalised_score'] = 5
  data_selected.loc[data['normalised_score'] == "4/5", 'normalised_score'] = 4
  data_selected.loc[data['normalised_score'] == "3/5", 'normalised_score'] = 3
  data_selected.loc[data['normalised_score'] == "2/5", 'normalised_score'] = 2
  data_selected.loc[data['normalised_score'] == "1/5", 'normalised_score'] = 1
  data_selected.loc[data['normalised_score'] == "0/5", 'normalised_score'] = 1
  
  print("review_scores after fraction to integer")
  #pd.set_option('display.max_rows', None)
  print(data_selected['normalised_score'].value_counts())

  # From Fraction-Scores (strings) to Number-Scores (int)
  data_selected.loc[data['bertbasemultiuncased_label'] == "5 stars", 'bertbasemultiuncased_label'] = 5
  data_selected.loc[data['bertbasemultiuncased_label'] == "4 stars", 'bertbasemultiuncased_label'] = 4
  data_selected.loc[data['bertbasemultiuncased_label'] == "3 stars", 'bertbasemultiuncased_label'] = 3
  data_selected.loc[data['bertbasemultiuncased_label'] == "2 stars", 'bertbasemultiuncased_label'] = 2
  data_selected.loc[data['bertbasemultiuncased_label'] == "1 star", 'bertbasemultiuncased_label'] = 1

  print("Final distribution of bertbasemultiuncased_label values after normalisation")
  #pd.set_option('display.max_rows', None)
  print(data_selected['bertbasemultiuncased_label'].value_counts())


  # Normalise to Positive/Negative (based on the normalised scores)
  data_selected['normalised_pone'] = data_selected['normalised_score']

  data_selected.loc[data_selected['normalised_pone'] == 5, 'normalised_pone'] = "POSITIVE"
  data_selected.loc[data_selected['normalised_pone'] == 4, 'normalised_pone'] = "POSITIVE"
  data_selected.loc[data_selected['normalised_pone'] == 3, 'normalised_pone'] = "POSITIVE"
  data_selected.loc[data_selected['normalised_pone'] == 2, 'normalised_pone'] = "NEGATIVE"
  data_selected.loc[data_selected['normalised_pone'] == 1, 'normalised_pone'] = "NEGATIVE"
  


  data_selected.to_csv("/home/christianschuler/data/NLP-Project-Review-Summaries/data-sentiment-normalised2.csv")

  # Exiting the script via code  
  sys.exit("(==== Test Fullstop ===)")
  
################################################################################  
  
################################################################################  
  # TEMP: Looking at initial data
  data = pd.read_csv("/home/christianschuler/data/NLP-Project-Review-Summaries/data.csv", encoding='unicode_escape')
  #data = data[:10]
  scores = data['review_score'].value_counts().sort_index()
  scores.to_csv("data-initial-scores.csv")

  with open("data-initial-scores.txt", "w") as text_file:
    print(scores)
    print(type(scores))
    #print(type(scores.to_string()))
    text_file.write(scores.to_string())

    #for col in scores.columns:
    	#scores[col].to_csv('scores_'+col+'.txt', index=False, header=False)
    
    #for score in scores:
    #	print(score)
    #	text_file.write(score)
    
  #print("Initial distribution of review_score values")
  #pd.set_option('display.max_rows', None)
  #print(data['review_score'].value_counts())

  # Exiting the script via code  
  sys.exit("(==== Test Fullstop ===)")  
################################################################################  


