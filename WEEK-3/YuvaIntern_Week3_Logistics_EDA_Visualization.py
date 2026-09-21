"""
Yuva Intern - Week 3
Advanced Data Analysis and Visualization in Logistics
Author: Aman Tiwari
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logistics_data_cleaned.csv")
df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

print(df.shape)
print(df.describe())
print(df.isna().sum())

print("On-time delivery rate:", df["on_time"].mean()*100)
print("Delay rate:", df["delayed"].mean()*100)
print("Average delivery time:", df["delivery_time_hours"].mean())
print("Average transportation cost:", df["transportation_cost"].mean())

region_summary = df.groupby("region").agg(
    shipments=("shipment_id","count"),
    avg_delivery_time=("delivery_time_hours","mean"),
    avg_cost=("transportation_cost","mean"),
    on_time_rate=("on_time","mean")
)
region_summary["on_time_rate"] *= 100
print(region_summary)

plt.hist(df["delivery_time_hours"].dropna(), bins=25)
plt.title("Distribution of Delivery Time")
plt.show()

for vehicle in df["vehicle_type"].unique():
    values = df.loc[df["vehicle_type"]==vehicle,"delivery_time_hours"].dropna()
    plt.boxplot(values, positions=[list(df["vehicle_type"].unique()).tolist().index(vehicle)+1])
plt.show()

plt.bar(region_summary.index.astype(str), region_summary["on_time_rate"])
plt.title("On-Time Delivery Rate by Region")
plt.xticks(rotation=15)
plt.show()

for level in sorted(df["traffic_level"].dropna().unique()):
    s=df[df["traffic_level"]==level]
    plt.scatter(s["distance_km"],s["delivery_time_hours"],label=level,alpha=.65)
plt.legend()
plt.title("Distance vs Delivery Time")
plt.show()

corr=df[[
    "distance_km","shipment_volume","promised_time_hours",
    "delivery_time_hours","transportation_cost"
]].corr()
plt.imshow(corr.values, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)),corr.columns,rotation=35,ha="right")
plt.yticks(range(len(corr.index)),corr.index)
plt.show()

monthly=df.set_index("delivery_date").resample("ME").agg(
    avg_delivery_time=("delivery_time_hours","mean"),
    avg_cost=("transportation_cost","mean"),
    shipments=("shipment_id","count")
).reset_index()
plt.plot(monthly["delivery_date"],monthly["avg_delivery_time"],marker="o")
plt.title("Monthly Average Delivery Time")
plt.show()
