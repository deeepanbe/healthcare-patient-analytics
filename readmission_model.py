#!/usr/bin/env python3
"""
readmission_model.py  
Hospital Readmission Prediction Model
Author: Deepanraj A
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

def load_patient_data(filepath='data/patient_records.csv'):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} patient records")
    return df

def feature_engineering(df):
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
    df['high_risk'] = ((df['previous_admissions'] > 2) | (df['comorbidities'] > 3)).astype(int)
    df['long_stay'] = (df['length_of_stay'] > 7).astype(int)
    return df

def train_readmission_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    df = load_patient_data()
    df = feature_engineering(df)
    
    features = ['age', 'previous_admissions', 'comorbidities', 'length_of_stay', 'high_risk', 'diagnosis_code']
    X = df[features]
    y = df['readmitted_30days']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_readmission_model(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Readmission Prediction Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, 'models/readmission_model.pkl')
    print("Model saved")

if __name__ == '__main__':
    main()
