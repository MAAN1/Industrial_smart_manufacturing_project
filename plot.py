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
       data['end'],
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
plt.yticks(y_pos, labels=reversed(data["Job"]))
# Xticks
#plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
#plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator())
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(fmt="%m-%d"))
#plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1))
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)

# grid
plt.grid(axis="x", which="major", lw=1)
plt.grid(axis="x", which="minor", ls="--", lw=1 )

# date rotation

#plt.gcf().autofmt_xdate(rotation=30)
plt.xlim(data["start"][0])
plt.xlabel("Date", fontsize=12, weight="bold")
plt.title("Project Schedule", fontsize=12, weight="bold")
plt.tight_layout(pad=1.8)
plt.show()
