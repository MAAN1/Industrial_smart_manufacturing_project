import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from datetime import timedelta
from matplotlib.dates import DateFormatter
sns.set()

data = pd.read_csv("smart_industry.csv")
print(" first print",data)

# convert data str to "datetime" data type, yyyy_MM_DD
data["start"] = pd.to_datetime(data["start"], infer_datetime_format=True, format = "%Y-%m-%d")
data["end"] = pd.to_datetime(data["end"], infer_datetime_format=True, format = "%Y-%m-%d") #%d %H:%M

data.sort_values("start", axis=0, ascending=True, inplace=True)
data.reset_index(drop=True, inplace=True)

# data duration column and its very important for us
data["Duration"] = data["end"] - data["start"]
data["PastTime"] = data["start"] - data["start"][0]
data["Day_of_week"] = data["start"].dt.strftime("%A")
data["Months"] = data["start"].dt.strftime("%B")



data.isnull().sum()
data["Date"] = data["end"] - data["start"]
data["PastTime"] = data["start"] - data["start"][0]

#data["time"] = data["start"].dt.strftime("%M:%H")

print(data)



# Create figure and plot space
fig, ax = plt.subplots(figsize=(12, 12))
"""
# Add x-axis and y-axis
ax.bar(data.index.values,
       data['start'],
       data["end"],
       color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Task",
       title="SESAME Work schedule")

# Define the date format
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)

plt.show()

"""

# start drawing
nrow= len(data)
plt.figure(num =1, figsize=[8,5], dpi=108)
bar_width = 0.9
for i in range(nrow):
    i_rev = nrow-1-i
    # plot the last task as first task
    plt.broken_barh([(data["start"][i_rev], data["Duration"][i_rev])], (i-bar_width/2,bar_width), color="b")
    plt.broken_barh([(data["start"][0], data["PastTime"][i_rev])], (i-bar_width/2,bar_width), color="#f2f2f2")

#------------------# Data prepration


#-----------------#
y_pos= np.arange(nrow)
plt.yticks(y_pos, labels=reversed(data["Job"]), )
# Xticks
#plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
#plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(fmt="%m-%d"))
#plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1))
#date_form = DateFormatter("%d")
#ax.xaxis.set_major_formatter(date_form)

# grid
plt.grid(axis="x", which="major", lw=1)
plt.grid(axis="x", which="minor", ls="--", lw=1 )

# date rotation

#plt.gcf().autofmt_xdate(rotation=30)
plt.xlim(data["start"][0])
plt.xlabel("Execution time ", fontsize=12, weight="bold")
plt.ylabel("Job IDs ", fontsize=12, weight="bold")
plt.title("Project Schedule", fontsize=12, weight="bold")
plt.tight_layout(pad=1.8)
plt.show()

"""
# dictionary of lists
#d = {'Car': ['BMW', 'Lexus', 'Audi', 'Mercedes', 'Jaguar', 'Bentley'], 'Date_of_purchase': ['2020-10-10', '2020-10-12', '2020-10-17', '2020-10-16', '2020-10-19', '2020-10-22']
#}
d = {'Job': ['1', '2', '3', '4', '5', '6'], 'start': ['2020-10-10', '2020-10-12', '2020-10-17', '2020-10-16', '2020-10-19', '2020-10-22'], 'end': ['2021-10-10', '2021-10-12', '2021-10-17', '2021-10-16', '2021-10-19', '2021-10-22']
}

# creating dataframe from the above dictionary of lists
dataFrame = pd.DataFrame(d)
print("DataFrame...\n",dataFrame)

# write dataFrame to SalesRecords CSV file
dataFrame.to_csv("C:\\Users\\imran\\PycharmProjects\\Industrial_smart_manufacturing_project\\jobs.csv")

# display the contents of the output csv
print("The output csv file written successfully and generated...")
"""
"""
"""










"""

#data = np.zeros((2,3))
#print("data",data, type(data))

#data1= np.array([[1,2,4], [4,5,3]])

#print("data1",data1, type(data1) )
list = [1,2,3,45,5]
list1= [5,6,7,89,90]
l_a = np.array(list)
l_a1 = np.array(list1)

print("list",list, type(list), "array",l_a, type(l_a))
def data2():
    for i in range(len(data)):
        print("the value of i", data1[i])

#data2()
"""
#--------#
"""

sns.set()
rewards = np.array([0, 0.1,0,0.2,0.4,0.5,0.6,0.9,0.9,0.9])
sns.lineplot(x=range(len(rewards)),y=rewards)
# sns.relplot(x=range(len(rewards)),y=rewards,kind="line") #  Equivalent to the line above
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("data")
plt.show()
"""
#-----------#
"""
rewards1 = np.array([0, 0.1,0,0.2,0.4,0.5,0.6,0.9,0.9,0.9])
rewards2 = np.array([0, 0,0.1,0.4,0.5,0.5,0.55,0.8,0.9,1])
rewards3 = np.vstack((rewards1,rewards2)) #  Merge into a two-dimensional array
rewards4 = np.concatenate((rewards1,rewards2)) #  Merge into a one-dimensional array
print(np.shape(rewards3))
print(rewards3)
print(np.shape(rewards4))
print(rewards4)
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("data")
plt.show()
"""


#----------#

"""
rewards1 = np.array([0, 0.1,0,0.2,0.4,0.5,0.6,0.9,0.9,0.9])
rewards2 = np.array([0, 0,0.1,0.4,0.5,0.5,0.55,0.8,0.9,1])
rewards=np.concatenate((rewards1,rewards2)) #  Merge array
episode1=range(len(rewards1))
episode2=range(len(rewards2))
episode=np.concatenate((episode1,episode2))
sns.lineplot(x=episode,y=rewards)
plt.xlabel("episode")
plt.ylabel("reward")
#plt.plot(rewards)
plt.show()
"""
""""

def get_data():
    ''' get data  '''
    bace= np.array((3,6))
    basecond = np.array([[18, 20, 19, 18, 13, 4, 1],[20, 17, 12, 9, 3, 0, 0],[20, 20, 20, 12, 5, 3, 0]])
    cond1 = np.array([[18, 19, 18, 19, 20, 15, 14],[19, 20, 18, 16, 20, 15, 9],[19, 20, 20, 20, 17, 10, 0]])
    cond2 = np.array([[20, 20, 20, 20, 19, 17, 4],[20, 20, 20, 20, 20, 19, 7],[19, 20, 20, 19, 19, 15, 2]])
    cond3 = np.array([[20, 20, 20, 20, 19, 17, 12],[18, 20, 19, 18, 13, 4, 1], [20, 19, 18, 17, 13, 2, 0]])
    return bace, cond1, cond2, cond3

data = get_data()
print(type(data))
label = ['algo1', 'algo2', 'algo3', 'algo4']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='loss'))
    df[i]['algo']= label[i]
df=pd.concat(df) #  Merge
print(df)
sns.lineplot(x="episode", y="loss", hue="algo", style="algo",data=df)
plt.title("some loss")
plt.show()
"""





"""
def get_data():
    base_cond = [[1, 20, 19, 18, 13, 4, 1],
            [20, 17, 12, 9, 3, 0, 0],
            [20, 20, 20, 12, 5, 3, 0]]
    cond1 = [[18, 19, 18, 19, 20, 15, 14],
             [19, 20, 18, 16, 20, 15, 9],
             [19, 20, 20, 20, 17, 10, 0],
             [20, 20, 20, 20, 7, 9, 1]]
    cond2 = [[20, 20, 20, 20, 19, 17, 4],
             [20, 20, 20, 20, 20, 19, 7],
             [19, 20, 20, 19, 19, 15, 2]]
    cond3 = [[20, 20, 20, 20, 19, 17, 12],
             [18, 20, 19, 18, 13, 4, 1],
             [20, 19, 18, 17, 13, 2, 0],
             [19, 18, 20, 20, 15, 6, 0]]
    return base_cond, cond1, cond2, cond3

results =get_data()
print("results datset" ,type(results))
#tips2 = pd.read_csv('results'
fig = plt.figure()
xdata = np.array([0, 1, 2, 3, 4])
print(type(xdata))
sns.tsplot(time=xdata, data = results[3], color = "r", linestyle ="-")
#sns.tsplot(time=xdata, data=results[1], color="g", linestyle="--")
#sns.tsplot(time=xdata, data=results[2], color="b", linestyle=":")
#sns.tsplot(time=xdata, data=results[3], color="k", linestyle="-")
#sns.tsplot(time=xdata, data=results[3], color="y", linestyle="-")
# y label success rate
plt.ylabel("Success Rate", fontsize=10)
# our x-axis
plt.xlabel("Iteration Number ", fontsize=10, labelpad=-4)
plt.title("Assambly line environment Agent performance", fontsize=15)
plt.legend(loc="lower left")
plt.show()
"""
