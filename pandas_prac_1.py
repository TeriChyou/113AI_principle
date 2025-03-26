from modules import *



# pd.DF
"""dataAmt = 3
data = {
    "a": [np.random.randint(300, 500) for _ in range(dataAmt)],
    "b": [np.random.randint(40, 50) for _ in range(dataAmt)],
    "c": [np.random.randint(3500, 5000) for _ in range(dataAmt)],
    "d": [np.random.randint(400, 600) for _ in range(dataAmt)]
}

myvar = pd.DataFrame(data)
print(myvar)
print(myvar.loc[0])
print(myvar.loc[0][1:2])"""

# pd.read
"""
df = pd.read_xml("https://www.w3schools.com/xml/books.xml")
print(df.to_string())
"""

# 處理髒資料
df = pd.read_csv("testData/csvData/dirtyData.csv")
print(df.to_string())

# Bad data includes:
# Empty row or column
# Data in wrong format
# Wrong data
# Duplicates

# df.dropna() # remove empty
# df['Calories'].fillna(130, inplace=True) # replace Nan
# df['Calories'].mean() # get median of the col

# get the date to correct format
df['Date'] = pd.to_datetime(df['Date'], format='mixed')
df.dropna(subset=['Date'], inplace=True)
print(df.to_string())