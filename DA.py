import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Define your extracted folder path
extract_folder = r"C:\Users\vasantha kumar\Downloads\97ba4770c04a9ff00cdc3221ae9c8731bdb445c3"

# Prepare to collect all monthly CSV data
all_dataframes = []

# Loop through each monthly subfolder
for folder in sorted(os.listdir(extract_folder)):
    folder_path = os.path.join(extract_folder, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(file_path)
                    df['source_month'] = folder  # Track which month it came from
                    all_dataframes.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Combine all dataframes into one
combined_df = pd.concat(all_dataframes, ignore_index=True)
combined_df

# Standardize column names
combined_df.columns = combined_df.columns.str.strip().str.lower().str.replace(' ', '_')
# Drop 'context' column if it exists
if 'context' in combined_df.columns:
    combined_df.drop(columns=['context'], inplace=True)

# Drop rows missing critical values
clean_df = combined_df.dropna(subset=['latitude', 'longitude', 'crime_type'])

# Remove duplicates and reset index
clean_df = clean_df.drop_duplicates().reset_index(drop=True)
clean_df
# Save to CSV
clean_df.to_csv("cleaned_crime_data.csv", index=False)
print(" Cleaned dataset saved as cleaned_crime_data.csv")
# Show basic info before detailed cleaning
combined_df.info(), combined_df.head()
# Drop duplicate rows
clean_df.drop_duplicates(inplace=True)
# Reset index after cleaning
clean_df.reset_index(drop=True, inplace=True)
# Show summary after cleaning
clean_summary = {
    "Original Rows": combined_df.shape[0],
    "Cleaned Rows": clean_df.shape[0],
    "Dropped Rows": combined_df.shape[0] - clean_df.shape[0],
    "Remaining Columns": clean_df.columns.tolist(),
    "Missing Values (Top 5 Columns)": clean_df.isnull().sum().sort_values(ascending=False).head(5).to_dict()
}
#tools.display_dataframe_to_user(name="Cleaned Crime Data", dataframe=clean_df)
clean_summary
# Set visual style
sns.set(style="whitegrid")

# Top 10 crime types
plt.figure(figsize=(10,5))
sns.countplot(data=clean_df, y='crime_type', order=clean_df['crime_type'].value_counts().head(10).index)
plt.title("Top 10 Most Common Crime Types")
plt.xlabel("Number of Incidents")
plt.ylabel("Crime Type")
plt.tight_layout()
plt.show()

# Crimes by month
plt.figure(figsize=(12,5))
clean_df['month'] = pd.to_datetime(clean_df['month'])
monthly_counts = clean_df.groupby(clean_df['month'].dt.to_period("M")).size()
monthly_counts.plot(kind='line', marker='o')
plt.title("Crime Volume Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.tight_layout()
plt.show()

# Top 10 LSOA crime hotspots
plt.figure(figsize=(10,5))
sns.countplot(data=clean_df, y='lsoa_name', order=clean_df['lsoa_name'].value_counts().head(10).index)
plt.title("Top 10 Crime Hotspots by LSOA")
plt.xlabel("Number of Crimes")
plt.ylabel("LSOA Name")
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder

# Extract year and month number
clean_df['year'] = clean_df['month'].dt.year
clean_df['month_num'] = clean_df['month'].dt.month

# Encode LSOA and reported_by
le_lsoa = LabelEncoder()
le_force = LabelEncoder()
clean_df['lsoa_encoded'] = le_lsoa.fit_transform(clean_df['lsoa_name'].astype(str))
clean_df['force_encoded'] = le_force.fit_transform(clean_df['reported_by'].astype(str))

# Drop original string columns for modelling
model_df = clean_df[['crime_type', 'longitude', 'latitude', 'year', 'month_num', 'lsoa_encoded', 'force_encoded']]
model_df = model_df.dropna()
model_df
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Encode target
le_crime = LabelEncoder()
model_df['crime_type_encoded'] = le_crime.fit_transform(model_df['crime_type'])

# Define features and target
X = model_df.drop(columns=['crime_type', 'crime_type_encoded'])
y = model_df['crime_type_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_crime.classes_))
