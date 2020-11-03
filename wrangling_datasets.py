#Data wrangling


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing datasets and saving them as a csv file
dataset = pd.ExcelFile('study_problem.xlsx')
sheet1 = dataset.parse('Existing employees')
sheet1.to_csv('Existing_employees.csv', sep=',')
print(sheet1.head())

sheet2 = dataset.parse('Employees who have left')
sheet2.to_csv('Ex_employees.csv', sep=',')
print(sheet2.head())
 
#adding a new column to the datasets in order to merge them and know which employees have left and which ones have not
#sheet1['left_company'] = 0  #value '0' stands for not left
#sheet1.info()
#sheet2['left_company'] = 1  # value '1' stands for left
#sheet2.info()

#merge sheet1 and sheet 2 together for easy exploration
#data = pd.concat([sheet2,sheet1],ignore_index=True) 
#data.to_csv('data.csv')
#print(data.head(10))

#after tweaking the data dataset using excel
emp_dataset = pd.read_csv('data.csv')
print(emp_dataset.info())













































































