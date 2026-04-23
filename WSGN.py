#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# ==========================================
# 1. Data Ingestion & Exploration
# ==========================================

# Load the synthetic dataset
try:
    df = pd.read_csv("WSGN_Project_Data\WSGN_Jamming_Spoofing_Dataset.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure CSV is in the directory.")
    exit()

# Sanity check
print(f"Data Shape: {df.shape}")
print("-" * 30)

# Check class distribution
class_counts = df['Label'].value_counts()
print("Traffic Distribution:")
print(class_counts)

# Visualizing the Physics: SNR vs PDR
# This proves the data isn't random noise - it follows network physics
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='SNR_dB', y='PDR_Percentage', hue='Label', alpha=0.6, palette='viridis')
plt.title("Physical Layer Analysis: SNR vs Packet Delivery Ratio")
plt.xlabel("Signal-to-Noise Ratio (dB)")
plt.ylabel("Packet Delivery Ratio (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('phy_layer_analysis.png') # Saving for report
print("\n[INFO] Physical Layer Analysis plot saved.")

# ==========================================
# 2. Preprocessing & Feature Engineering
# ==========================================

# Dropping identifiers that cause overfitting (Timestamp, IDs)
# We want the model to learn signal characteristics, not specific device IDs
X = df.drop(['Timestamp', 'Source_Node_ID', 'Label', 'Class_ID'], axis=1)
y = df['Class_ID'] # Target: 0=Normal, 1=Jamming, 2=Spoofing

# Train/Test Split (80/20 standard split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
# Essential because RSSI is in dBm (negative) and Throughput is in Kbps (positive high)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining Set: {X_train.shape}")
print(f"Testing Set: {X_test.shape}")

# ==========================================
# 3. Model Training (Cognitive Engine)
# ==========================================

print("\n[INFO] Initializing Cognitive Radio Learning Engine...")
# Using Random Forest: Robust to noise and high dimensional signal data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 3.1 Cross-Validation (Methodology Requirement)
print("[INFO] Performing 5-Fold Cross-Validation to verify robustness...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

start_train = time.time()
rf_model.fit(X_train_scaled, y_train)
end_train = time.time()

print(f"Model trained in {end_train - start_train:.4f} seconds.")

# ==========================================
# 4. Performance Evaluation
# ==========================================

# Inference
y_pred = rf_model.predict(X_test_scaled)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nSystem Accuracy: {acc*100:.2f}%")
print("-" * 30)
print("Classification Report (Precision, Recall, F1-Score):")
target_names = ['Normal', 'Jamming', 'Spoofing']
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix: Attack Detection")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()

# ==========================================
# 5. Autonomous Fall-Back Mechanism Simulation
# ==========================================

def autonomous_fallback_protocol(signal_vector, prediction):
    """
    Simulates the decision logic of the Smart Grid Node based on ML output.
    """
    prediction_label = target_names[prediction]
    
    print(f"\n[Incoming Signal Vector]: {signal_vector[:4]}...") # Print first few features
    print(f"[Analysis]: Detected State -> {prediction_label.upper()}")
    
    if prediction_label == 'Normal':
        print(">> ACTION: Maintain current channel. Link Stable.")
        return "STABLE"
        
    elif prediction_label == 'Jamming':
        print(">> CRTICAL ALERT: Spectrum Denial Detected.")
        print(">> FALLBACK ACTION: Initiating Frequency Hopping Spread Spectrum (FHSS).")
        print(">> Switching to Backup Channel 11 (2.45 GHz)...")
        return "FHSS_TRIGGERED"
        
    elif prediction_label == 'Spoofing':
        print(">> SECURITY ALERT: Anomalous Jitter/RSSI profile.")
        print(">> FALLBACK ACTION: Revoking Neighbor Trust.")
        print(">> Requesting Multi-Factor Authentication (MFA) from Source Node.")
        return "AUTH_CHALLENGE"

print("\n" + "="*40)
print("SIMULATION: LIVE TRAFFIC INTERCEPTION")
print("="*40)

# Simulate 5 random live events from the test set
indices = np.random.choice(len(X_test), 60)
for i in indices:
    sample_data = X_test.iloc[i].values
    sample_scaled = X_test_scaled[i].reshape(1, -1)
    
    # Predict
    pred_class = rf_model.predict(sample_scaled)[0]
    
    # Execute Logic
    autonomous_fallback_protocol(sample_data, pred_class)
    time.sleep(0.5) # Pause for effect


# In[ ]:




