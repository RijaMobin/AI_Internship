# Enhanced Feature Engineering with step-by-step change log
# Save as Feature_engineering.py and run locally (no ChatGPT-only modules)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import CountVectorizer
import pprint

pp = pprint.PrettyPrinter(indent=2)

# ----------------------------
# Utility: log changes between two DataFrames
# ----------------------------
change_log = []

def log_change(step_name, before_df, after_df, rows_removed=None, reason=None):
    """
    Compare before_df and after_df and print/add to change_log:
      - added columns
      - removed columns
      - modified columns (values changed)
      - rows removed (by Name if available)
    """
    before_cols = set(before_df.columns)
    after_cols = set(after_df.columns)

    added_cols = sorted(list(after_cols - before_cols))
    removed_cols = sorted(list(before_cols - after_cols))

    # detect modified columns (present in both but values changed)
    common_cols = list(before_cols & after_cols)
    modified_cols = []
    for col in common_cols:
        try:
            # use 'Name' as key if available to align rows
            if 'Name' in before_df.columns and 'Name' in after_df.columns:
                b = before_df.set_index('Name')[col].fillna('__NA__').astype(str)
                a = after_df.set_index('Name')[col].fillna('__NA__').astype(str)
                idx = b.index.intersection(a.index)
                if not idx.empty and (b.loc[idx] != a.loc[idx]).any():
                    modified_cols.append(col)
            else:
                b = before_df[col].fillna('__NA__').astype(str)
                a = after_df[col].fillna('__NA__').astype(str)
                if (b != a).any():
                    modified_cols.append(col)
        except Exception:
            # If any error in comparing (different types), consider modified
            modified_cols.append(col)

    # detect rows removed by Name (if Name exists)
    removed_rows_auto = []
    if 'Name' in before_df.columns and 'Name' in after_df.columns:
        removed_rows_auto = sorted(list(set(before_df['Name'].tolist()) - set(after_df['Name'].tolist())))

    # combine automatic row removal detection with explicit rows_removed param
    rows_removed_final = rows_removed if rows_removed is not None else removed_rows_auto

    entry = {
        'step': step_name,
        'added_columns': added_cols,
        'removed_columns': removed_cols,
        'modified_columns': modified_cols,
        'rows_removed': rows_removed_final,
        'reason': reason
    }
    change_log.append(entry)

    # Print a concise summary for this step
    print("\n--- Step:", step_name, "---")
    if added_cols:
        print("✅ Added columns:", added_cols)
    if removed_cols:
        print("❌ Removed columns:", removed_cols)
    if modified_cols:
        print("✏️ Modified columns (values changed):", modified_cols)
    if rows_removed_final:
        print("🗑 Rows removed:", rows_removed_final, "| Reason:", reason or "unspecified")
    if not (added_cols or removed_cols or modified_cols or rows_removed_final):
        print("No structural change in columns or rows for this step.")
    print("-" * 40)


# ----------------------------
# 0. BEFORE dataset (keep copy)
# ----------------------------
before_data = pd.DataFrame({
    'Name': ['Ali', 'Sara', 'John', 'Anna', 'Mike'],
    'Age': [25, np.nan, 35, 29, np.nan],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'City': ['Lahore', 'Karachi', 'Islamabad', 'Lahore', 'Karachi'],
    'Fare': [5000, 7000, 20000, 15000, 100000],
    'SibSp': [1, 0, 2, 1, 0],
    'Parch': [0, 1, 1, 0, 0],
    'Travel_Date': pd.to_datetime(['2024-01-05', '2024-06-12', '2024-03-19', '2024-07-25', '2024-08-15']),
    'Notes': ['Loves traveling', 'Prefers luxury', 'Budget traveler', 'Family trip', 'Solo backpacker'],
    'Survived': [1, 0, 1, 1, 0]
})
print("\n📌 BEFORE Feature Engineering (original):")
print(before_data)

# working copy
df = before_data.copy()

# ----------------------------
# 1. Handling Missing Data
# ----------------------------
before_step = df.copy()
# Fill missing 'Age' values with median (imputation)
df['Age'] = df['Age'].fillna(df['Age'].median())
log_change("1. Missing data - impute Age with median", before_step, df,
           rows_removed=None, reason="Imputed missing Age values with median")

# ----------------------------
# 2. Encoding Categorical Variables
# ----------------------------
before_step = df.copy()
# Label encode Gender (in-place replacement) and One-Hot encode City
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # modifies Gender values

# One-hot encode City and drop original City column
df = pd.get_dummies(df, columns=['City'], drop_first=True)
log_change("2. Encoding categorical variables (Label + One-Hot)", before_step, df,
           rows_removed=None, reason="Gender encoded; City -> one-hot columns")

# ----------------------------
# 3. Scaling Numerical Features
# ----------------------------
before_step = df.copy()
scaler = MinMaxScaler()
df[['Age_scaled', 'Fare_scaled']] = scaler.fit_transform(df[['Age', 'Fare']])
log_change("3. Scaling numerical features (Min-Max)", before_step, df,
           rows_removed=None, reason="Added scaled numeric columns")

# ----------------------------
# 4. Feature Creation
# ----------------------------
before_step = df.copy()
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
log_change("4. Feature creation (Family_Size)", before_step, df,
           rows_removed=None, reason="Created Family_Size from SibSp and Parch")

# ----------------------------
# 5. Feature Transformation
# ----------------------------
before_step = df.copy()
df['Fare_Log'] = np.log1p(df['Fare'])
log_change("5. Feature transformation (log Fare)", before_step, df,
           rows_removed=None, reason="Added Fare_Log to reduce skew")

# ----------------------------
# 6. Feature Selection (just compute selected features)
# ----------------------------
before_step = df.copy()
X = df.select_dtypes(include=[np.number]).drop(columns=['Survived'])
y = df['Survived']
selector = SelectKBest(score_func=f_classif, k=min(3, X.shape[1]))
selector.fit(X, y)
selected_features = list(X.columns[selector.get_support()])
# Feature selection step does not modify dataset columns, but we log result
log_change("6. Feature selection (ANOVA F-test)", before_step, df,
           rows_removed=None, reason=f"Selected top features: {selected_features}")

# ----------------------------
# 7. Outlier Handling (IQR) - we'll REMOVE outliers here
#    (Option: you can change to 'cap' if you want to keep rows)
# ----------------------------
before_step = df.copy()

Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# identify rows that will be removed as outliers
outlier_mask = (df['Fare'] < lower_limit) | (df['Fare'] > upper_limit)
rows_to_remove = df.loc[outlier_mask, 'Name'].tolist()

# Remove outlier rows (this removes 'Mike' in this dataset)
df = df.loc[~outlier_mask].reset_index(drop=True)

log_change("7. Outlier handling (IQR removal)", before_step, df,
           rows_removed=rows_to_remove, reason=f"Fare outliers outside [{lower_limit}, {upper_limit}] removed")

# ----------------------------
# 8. Date Feature Extraction
# ----------------------------
before_step = df.copy()
df['Year'] = df['Travel_Date'].dt.year
df['Month'] = df['Travel_Date'].dt.month
df['DayOfWeek'] = df['Travel_Date'].dt.dayofweek
log_change("8. Date feature extraction (Year/Month/DayOfWeek)", before_step, df,
           rows_removed=None, reason="Extracted year, month, weekday from Travel_Date")

# ----------------------------
# 9. Text Feature Engineering (Bag of Words)
# ----------------------------
before_step = df.copy()
cv = CountVectorizer()
text_features = cv.fit_transform(df['Notes']).toarray()
text_cols = list(cv.get_feature_names_out())
text_df = pd.DataFrame(text_features, columns=text_cols, index=df.index)
df = pd.concat([df.reset_index(drop=True), text_df.reset_index(drop=True)], axis=1)
log_change("9. Text -> Bag of Words", before_step, df,
           rows_removed=None, reason="Converted Notes into word-count columns: " + ", ".join(text_cols))

# ----------------------------
# 10. Interaction Features
# ----------------------------
before_step = df.copy()
df['Age_Fare'] = df['Age'] * df['Fare']
log_change("10. Interaction features (Age * Fare)", before_step, df,
           rows_removed=None, reason="Created Age_Fare interactive feature")

# ----------------------------
# Final prints & summary
# ----------------------------
print("\n✅ AFTER Feature Engineering (final):")
print(df)

print("\n🏆 Top Selected Features (ANOVA F-Test):")
print(selected_features)

print("\n\n===== CHANGE LOG (all steps) =====")
pp.pprint(change_log)

# Optional: Save before and after for inspection
df.to_csv("feature_engineered_after.csv", index=False)
before_data.to_csv("feature_engineered_before.csv", index=False)
print("\nSaved 'feature_engineered_before.csv' and 'feature_engineered_after.csv' in current folder.")
