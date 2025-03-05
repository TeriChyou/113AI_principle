from modules import *
def show():
    mpl.pyplot.show()
# probability practice 一個陣列＋一個機率分佈 去跑count看結果
# arr_1 = [1,3,5,7,9]
# prob_1 = [0.1, 0.2, 0, 0.45, 0.25]
# prob_arr_1 = np.random.choice(arr_1, p=prob_1, size=(1000))
# Count occurrences of each number
# unique, counts = np.unique(prob_arr_1, return_counts=True)
# Display results
# count_dict = dict(zip(unique, counts))
# print(count_dict)

# sns.displot(prob_arr_1, kind='hist')
# show()
"""
{1: 106, 3: 164, 7: 471, 9: 259}
"""

# sns.displot([0,1,2,3,4,5])
# mpl.pyplot.show()

# Normal distribution 常態分佈討論
# x = np.random.normal(loc=1,scale=2,size=(2,3))
# print(x)

# 以下會印出常態分佈圖 normal(Gauss) distribution => e^((-(x-σ)^2)/(a^2))
# sns.displot(np.random.normal(size=1000), kind='kde')
# mpl.pyplot.show()

# 以下會印出二項式分佈 binomial distribution => (x+y)^n
# sns.displot(np.random.binomial(n=10, p=0.5, size=1000), kind='hist')
# show()

# 以下會印出波以鬆分佈 poisson distribution
# sns.displot(np.random.poisson(lam=2, size=1000), kind='hist')
# show()

# 以下會印出連續型均勻分佈 uniform distribution
# sns.displot(np.random.uniform(size=(1000)), kind='kde')
# show()

# 以下會印出logistic distribution
# sns.displot(np.random.logistic(size=(1000)), kind='kde')
# show()

# 多項式分佈
# sns.displot(np.random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]), kind='ecdf')
# show()

# 指數分佈
# sns.displot(np.random.exponential(size=1000), kind='kde')
# show()

# 卡方檢定 Chi Square Distribution
# sns.displot(np.random.chisquare(df=1, size=(1000)), kind='kde')

""" # These code will show different df of Chisquare distribution
#  Generate Chi-square distributed data
data1 = np.random.chisquare(df=1, size=1000)
data2 = np.random.chisquare(df=2, size=1000)
data3 = np.random.chisquare(df=3, size=1000)

# Plot all distributions on the same figure
sns.kdeplot(data1, label="df=1", linewidth=2)
sns.kdeplot(data2, label="df=2", linewidth=2)
sns.kdeplot(data3, label="df=3", linewidth=2)

# Add legend and title
mpl.pyplot.legend()
mpl.pyplot.title("Chi-square Distributions with Different Degrees of Freedom")
mpl.pyplot.show()
"""

# Rayleigh distibution => {\displaystyle f(x;\sigma )={\frac {x}{\sigma ^{2}}}e^{-x^{2}/2\sigma ^{2}},\quad x\geq 0,}
# sns.displot(np.random.rayleigh(size=1000),kind='kde')
# show()

# Pareto distribution (八二法則)
# sns.displot(np.random.pareto(a=2, size=1000), kind='hist')
# show()

# Zipf Distribution 在自然語言的語料庫裡，一個單詞出現的頻率與它在頻率表里的排名成反比
# x = np.random.zipf(a=2, size=1000)
# sns.displot(x[x<10], kind='hist')
# show()

