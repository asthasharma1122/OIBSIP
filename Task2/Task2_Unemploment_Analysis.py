import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# Path to ZIP file
zip_path = r"D:\OIBSIP\Task2\unemployment_dataset.zip"

# List files in the ZIP (optional)
with zipfile.ZipFile(zip_path, 'r') as z:
    print("Files in ZIP:", z.namelist())

# Read CSV from ZIP
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open("Unemployment in India.csv") as f:  # exact name from ZIP
        df = pd.read_csv(f)

# Clean column names
df.columns = df.columns.str.strip()

# Rename columns for convenience
df.rename(columns={
    'Region': 'State',
    'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation'
}, inplace=True)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# Basic checks
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.info())
print(df.head())
print("Missing values:\n", df.isnull().sum())

# ----------------------------
# Plot 1: Overall unemployment trend in India
plt.figure(figsize=(12,6))
df.groupby('Date')['Unemployment_Rate'].mean().plot()
plt.title("Overall Unemployment Rate in India Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ----------------------------
# Plot 2: State-wise unemployment trends (sample 6 states)
sample_states = df['State'].dropna().unique()[:6]
plt.figure(figsize=(14,7))
for state in sample_states:
    state_data = df[df['State'] == state]
    plt.plot(state_data['Date'], state_data['Unemployment_Rate'], label=state)

plt.title("Unemployment Rate Trends (Sample States)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()

# ----------------------------
# Plot 3: Urban vs Rural comparison
plt.figure(figsize=(8,5))
sns.boxplot(x='Area', y='Unemployment_Rate', data=df)
plt.title("Urban vs Rural Unemployment Rate Distribution")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ----------------------------
# Plot 4: Average unemployment rate by state
state_unemp = df.groupby('State')['Unemployment_Rate'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,8))
state_unemp.plot(kind='bar', color='teal')
plt.title("Average Unemployment Rate by State")
plt.xlabel("State")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.show()

# ----------------------------
# Plot 5: Average unemployment by area (Rural vs Urban)
area_unemp = df.groupby('Area')['Unemployment_Rate'].mean()
plt.figure(figsize=(6,6))
area_unemp.plot(kind='bar', color=['orange', 'purple'])
plt.title("Unemployment Rate: Rural vs Urban")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ----------------------------
# Plot 6: Labour Participation vs Unemployment
plt.figure(figsize=(10,6))
plt.scatter(df['Labour_Participation'], df['Unemployment_Rate'], alpha=0.5, color='red')
plt.title("Labour Participation vs Unemployment Rate")
plt.xlabel("Labour Participation Rate (%)")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# ----------------------------
# National average unemployment over time
ts = df.groupby('Date')['Unemployment_Rate'].mean()
plt.figure(figsize=(12,6))
ts.plot()
plt.title("India - National Average Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.show()
