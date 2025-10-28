import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Attention, Permute, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==================== OUTPUT DIRECTORY ====================
import os
output_dir = 'review/severity_experiment'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}\n")

# ==================== DATA LOADING ====================
print("Loading dataset...")
df = pd.read_csv('/home/m8m/Projects/PPMI_Research_on_Parkinsons-master/review/finalDatasetWithUPDRSScore.csv')

# Define feature categories
non_feature_columns = ['Patient ID', 'Visit Date', 'UPDRS_SCORE', 'Visit', 'Visit_int',
                       'NHY', 'DATSCAN_PUTAMEN_R', 'DATSCAN_CAUDATE_R', 'DATSCAN_CAUDATE_L',
                       'DATSCAN_PUTAMEN_L_ANT', 'DATSCAN_PUTAMEN_R_ANT', 'DATSCAN_PUTAMEN_L',
                       'Disease_Severity']

numerical_features = [
    'Area', 'Circularity', 'ConvexArea', 'EquivDiameter', 'Extent',
    'FilledArea', 'Kurtosis', 'Major axis length', 'Mean',
    'Minor axis length', 'PA_ratio', 'Shannon_Entropy', 'Skewness',
    'Solidity', 'Standard Deviation', 'brightness', 'contrast',
    'correlation', 'dissimilarity', 'energy', 'gabor_energy',
    'gabor_entropy', 'homogeneity', 'lbp_energy', 'lbp_entropy'
]

categorical_features = [
    'NP1ANXS', 'NP1APAT', 'NP1COG', 'NP1DDS', 'NP1DPRS',
    'NP1HALL', 'NP1CNST', 'NP1FATG', 'NP1LTHD', 'NP1PAIN', 'NP1SLPD',
    'NP1SLPN', 'NP1URIN', 'NP2DRES', 'NP2EAT', 'NP2FREZ', 'NP2HOBB',
    'NP2HWRT', 'NP2HYGN', 'NP2RISE', 'NP2SALV', 'NP2SPCH', 'NP2SWAL',
    'NP2TRMR', 'NP2TURN', 'NP2WALK', 'NP3BRADY', 'NP3FACXP', 'NP3FRZGT',
    'NP3FTAPL', 'NP3FTAPR', 'NP3GAIT', 'NP3HMOVL', 'NP3HMOVR', 'NP3KTRML',
    'NP3KTRMR', 'NP3LGAGL', 'NP3LGAGR', 'NP3POSTR', 'NP3PRSPL', 'NP3PRSPR',
    'NP3PSTBL', 'NP3PTRML', 'NP3PTRMR', 'NP3RIGLL', 'NP3RIGLU', 'NP3RIGN',
    'NP3RIGRL', 'NP3RIGRU', 'NP3RISNG', 'NP3RTALJ', 'NP3RTALL', 'NP3RTALU',
    'NP3RTARL', 'NP3RTARU', 'NP3RTCON', 'NP3SPCH', 'NP3TTAPL', 'NP3TTAPR'
]

# Create severity categories from UPDRS scores
# Use existing Disease_Severity column (no need to recalculate)
print(f"Using existing Disease_Severity column")
print(f"Severity distribution: {df['Disease_Severity'].value_counts().to_dict()}")

# Preprocessing - MinMaxScaler for numerical features
print("Preprocessing data...")
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Combine features
all_features = numerical_features + categorical_features
X = df[all_features].values
y_severity = df['Disease_Severity'].values
y_visit = df['Visit'].values
patient_ids = df['Patient ID'].values

# Encode labels
severity_encoder = LabelEncoder()
visit_encoder = LabelEncoder()
y_severity_encoded = severity_encoder.fit_transform(y_severity)
y_visit_encoded = visit_encoder.fit_transform(y_visit)

# Reshape for LSTM (samples, timesteps, features)
X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])

# Split data
X_train, X_test, y_sev_train, y_sev_test, y_vis_train, y_vis_test, pid_train, pid_test = train_test_split(
    X_reshaped, y_severity_encoded, y_visit_encoded, patient_ids, 
    test_size=0.2, random_state=42, stratify=y_severity_encoded
)

# Convert to categorical
y_sev_train_cat = to_categorical(y_sev_train)
y_sev_test_cat = to_categorical(y_sev_test)
y_vis_train_cat = to_categorical(y_vis_train)
y_vis_test_cat = to_categorical(y_vis_test)

# Binarize for multi-class AUC calculation
y_sev_test_binarized = label_binarize(y_sev_test, classes=np.unique(y_severity_encoded))
y_vis_test_binarized = label_binarize(y_vis_test, classes=np.unique(y_visit_encoded))

n_severity_classes = len(np.unique(y_severity_encoded))
n_visit_classes = len(np.unique(y_visit_encoded))

print(f"\nDataset Info:")
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]} ({len(numerical_features)} numerical + {len(categorical_features)} categorical)")
print(f"Severity classes: {n_severity_classes} - {severity_encoder.classes_}")
print(f"Visit classes: {n_visit_classes} - {visit_encoder.classes_}")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ==================== SEVERITY MODEL ====================
print("\n" + "="*60)
print("TRAINING SEVERITY PREDICTION MODEL")
print("="*60)

severity_model = Sequential([
    # First Bidirectional LSTM layer
    Bidirectional(LSTM(96, return_sequences=True, 
                       recurrent_dropout=0.2,
                       kernel_regularizer=regularizers.l2(0.001)),
                  input_shape=(1, X.shape[1])),
    Dropout(0.4),
    BatchNormalization(),
    
    # Second LSTM layer
    LSTM(64, return_sequences=False,
         recurrent_dropout=0.2,
         kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.4),
    BatchNormalization(),
    
    # Dense layers
    Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(n_severity_classes, activation='softmax')
])

severity_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

severity_callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_lr=1e-6, verbose=1, mode='max')
]

severity_history = severity_model.fit(
    X_train, y_sev_train_cat,
    validation_data=(X_test, y_sev_test_cat),
    epochs=200,
    batch_size=32,
    callbacks=severity_callbacks,
    verbose=1
)

# Severity predictions
y_sev_pred_proba = severity_model.predict(X_test)
y_sev_pred = np.argmax(y_sev_pred_proba, axis=1)

# Severity metrics
sev_accuracy = accuracy_score(y_sev_test, y_sev_pred)
sev_auc = roc_auc_score(y_sev_test_cat, y_sev_pred_proba, multi_class='ovr', average='weighted')

print(f"\n{'='*60}")
print(f"SEVERITY MODEL RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {sev_accuracy*100:.2f}%")
print(f"AUC: {sev_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_sev_test, y_sev_pred, target_names=severity_encoder.classes_))

# ==================== VISIT MODEL ====================
print("\n" + "="*60)
print("TRAINING VISIT PREDICTION MODEL")
print("="*60)

visit_model = Sequential([
    # First Bidirectional LSTM layer
    Bidirectional(LSTM(96, return_sequences=True,
                       recurrent_dropout=0.2,
                       kernel_regularizer=regularizers.l2(0.001)),
                  input_shape=(1, X.shape[1])),
    Dropout(0.4),
    BatchNormalization(),
    
    # Second LSTM layer
    LSTM(64, return_sequences=False,
         recurrent_dropout=0.2,
         kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.4),
    BatchNormalization(),
    
    # Dense layers
    Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(n_visit_classes, activation='softmax')
])

visit_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

visit_callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, min_lr=1e-6, verbose=1, mode='max')
]

visit_history = visit_model.fit(
    X_train, y_vis_train_cat,
    validation_data=(X_test, y_vis_test_cat),
    epochs=200,
    batch_size=32,
    callbacks=visit_callbacks,
    verbose=1
)

# Visit predictions
y_vis_pred_proba = visit_model.predict(X_test)
y_vis_pred = np.argmax(y_vis_pred_proba, axis=1)

# Visit metrics
vis_accuracy = accuracy_score(y_vis_test, y_vis_pred)
vis_auc = roc_auc_score(y_vis_test_cat, y_vis_pred_proba, multi_class='ovr', average='weighted')

print(f"\n{'='*60}")
print(f"VISIT MODEL RESULTS")
print(f"{'='*60}")
print(f"Accuracy: {vis_accuracy*100:.2f}%")
print(f"AUC: {vis_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_vis_test, y_vis_pred, target_names=visit_encoder.classes_))

# ==================== LONGITUDINAL ANALYSIS ====================
print("\n" + "="*60)
print("LONGITUDINAL ANALYSIS")
print("="*60)

# Get predictions for all data
all_sev_pred_proba = severity_model.predict(X_reshaped)
all_sev_pred = np.argmax(all_sev_pred_proba, axis=1)
all_vis_pred_proba = visit_model.predict(X_reshaped)
all_vis_pred = np.argmax(all_vis_pred_proba, axis=1)

# Create results dataframe
results_df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Visit_Date': df['Visit Date'].values,
    'UPDRS_SCORE': df['UPDRS_SCORE'].values,
    'Visit': visit_encoder.inverse_transform(y_visit_encoded),
    'Actual_Severity': severity_encoder.inverse_transform(y_severity_encoded),
    'Predicted_Severity': severity_encoder.inverse_transform(all_sev_pred),
    'Actual_Visit': visit_encoder.inverse_transform(y_visit_encoded),
    'Predicted_Visit': visit_encoder.inverse_transform(all_vis_pred)
})

# Convert Visit_Date to datetime for temporal analysis
results_df['Visit_Date'] = pd.to_datetime(results_df['Visit_Date'], format='%Y-%m', errors='coerce')

# ==================== LONGITUDINAL ANALYSIS - SEVERITY OVER TIME ====================
print("\n" + "="*80)
print("LONGITUDINAL ANALYSIS - SEVERITY PROGRESSION OVER TIME")
print("="*80)

# Calculate months from first visit for each patient
results_df = results_df.sort_values(['Patient_ID', 'Visit_Date'])
results_df['Months_From_Baseline'] = results_df.groupby('Patient_ID')['Visit_Date'].transform(
    lambda x: ((x - x.min()).dt.days / 30.44).round(1)  # Average days per month
)

# Map severity to numeric values for plotting
severity_numeric_map = {sev: idx for idx, sev in enumerate(sorted(results_df['Actual_Severity'].unique()))}
results_df['Actual_Severity_Numeric'] = results_df['Actual_Severity'].map(severity_numeric_map)
results_df['Predicted_Severity_Numeric'] = results_df['Predicted_Severity'].map(severity_numeric_map)

print(f"Total patients: {results_df['Patient_ID'].nunique()}")
print(f"Severity mapping: {severity_numeric_map}")

# ==================== PLOTTING ====================
print("\nGenerating plots...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define colors for all severity levels
severity_colors = {
    'Minimal': '#27ae60',         # Light green
    'Mild': '#2ecc71',            # Green
    'Moderate': '#f39c12',        # Orange
    'Moderate to Severe': '#e67e22',  # Dark orange
    'Severe': '#e74c3c',          # Red
    'Very Severe': '#c0392b'      # Dark red
}

# 1. INDIVIDUAL PATIENT LONGITUDINAL PLOTS - Each patient in separate file
print("Creating individual patient longitudinal plots...")
all_patients = results_df['Patient_ID'].unique()

# Create a subdirectory for individual patient plots
patient_plots_dir = os.path.join(output_dir, 'individual_patients')
os.makedirs(patient_plots_dir, exist_ok=True)

for idx, patient_id in enumerate(all_patients):
    patient_data = results_df[results_df['Patient_ID'] == patient_id].sort_values('Months_From_Baseline')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual severity
    ax.plot(patient_data['Months_From_Baseline'], 
            patient_data['Actual_Severity_Numeric'], 
            'o-', linewidth=3, markersize=12, color='#3498db', 
            label='Actual Severity', alpha=0.9)
    
    # Plot predicted severity
    ax.plot(patient_data['Months_From_Baseline'], 
            patient_data['Predicted_Severity_Numeric'], 
            's--', linewidth=3, markersize=10, color='#e74c3c', 
            label='Predicted Severity', alpha=0.9)
    
    ax.set_title(f'Patient {int(patient_id)} - Severity Progression Over Time', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Months from Baseline', fontsize=13, fontweight='bold')
    ax.set_ylabel('Severity Level', fontsize=13, fontweight='bold')
    ax.set_yticks(range(len(severity_numeric_map)))
    ax.set_yticklabels([sev for sev in sorted(severity_numeric_map.keys())], fontsize=11, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Set x-axis limits
    max_months = patient_data['Months_From_Baseline'].max()
    ax.set_xlim([-2, max_months + 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(patient_plots_dir, f'patient_{int(patient_id)}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(all_patients)} patients...")

print(f"✓ Saved: {len(all_patients)} individual patient plots in '{patient_plots_dir}/'")

# 2. COMBINED SUMMARY PLOT - ONLY mean lines (no individual patients)
print("Creating combined summary plot with mean trends only...")
fig, ax = plt.subplots(figsize=(14, 8))

# Calculate and plot mean severity trend by time bins
time_bins = np.arange(0, results_df['Months_From_Baseline'].max() + 6, 6)  # 6-month bins
results_df['Time_Bin'] = pd.cut(results_df['Months_From_Baseline'], bins=time_bins, include_lowest=True)

# Calculate mean and std for actual and predicted
mean_actual = results_df.groupby('Time_Bin')['Actual_Severity_Numeric'].agg(['mean', 'std'])
mean_predicted = results_df.groupby('Time_Bin')['Predicted_Severity_Numeric'].agg(['mean', 'std'])
bin_centers = [(interval.left + interval.right) / 2 for interval in mean_actual.index]

# Plot ONLY the mean lines with confidence bands
ax.plot(bin_centers, mean_actual['mean'], 'o-', linewidth=4, markersize=14, 
        color='#3498db', label='Actual Severity (Mean)', zorder=10)
ax.fill_between(bin_centers, 
                mean_actual['mean'] - mean_actual['std'], 
                mean_actual['mean'] + mean_actual['std'], 
                alpha=0.2, color='#3498db', zorder=5, label='Actual ±1 SD')

ax.plot(bin_centers, mean_predicted['mean'], 's-', linewidth=4, markersize=12, 
        color='#e74c3c', label='Predicted Severity (Mean)', zorder=10)
ax.fill_between(bin_centers, 
                mean_predicted['mean'] - mean_predicted['std'], 
                mean_predicted['mean'] + mean_predicted['std'], 
                alpha=0.2, color='#e74c3c', zorder=5, label='Predicted ±1 SD')

ax.set_xlabel('Months from Baseline', fontsize=14, fontweight='bold')
ax.set_ylabel('Severity Level', fontsize=14, fontweight='bold')
ax.set_title(f'Mean Actual vs Predicted Severity Over Time\n(n={len(all_patients)} patients)', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_yticks(range(len(severity_numeric_map)))
ax.set_yticklabels([sev for sev in sorted(severity_numeric_map.keys())], fontsize=12, rotation=45, ha='right')
ax.legend(fontsize=12, loc='best', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_combined_mean_severity_over_time.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 1_combined_mean_severity_over_time.png")

# 3. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Severity confusion matrix
cm_sev = confusion_matrix(y_sev_test, y_sev_pred)
sns.heatmap(cm_sev, annot=True, fmt='d', cmap='Blues', 
            xticklabels=severity_encoder.classes_, 
            yticklabels=severity_encoder.classes_,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title(f'Severity Confusion Matrix\nAccuracy: {sev_accuracy*100:.2f}% | AUC: {sev_auc:.4f}', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')

# Visit confusion matrix
cm_vis = confusion_matrix(y_vis_test, y_vis_pred)
sns.heatmap(cm_vis, annot=True, fmt='d', cmap='Greens', 
            xticklabels=visit_encoder.classes_, 
            yticklabels=visit_encoder.classes_,
            ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title(f'Visit Confusion Matrix\nAccuracy: {vis_accuracy*100:.2f}% | AUC: {vis_auc:.4f}', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 3_confusion_matrices.png")

# 4. Severity Model Metrics Summary (ONLY Severity)
fig, ax = plt.subplots(figsize=(8, 6))

metrics_data = {
    'Metric': ['Accuracy (%)', 'AUC (×100)'],
    'Score': [sev_accuracy * 100, sev_auc * 100]
}

bars = ax.bar(metrics_data['Metric'], metrics_data['Score'], color=['steelblue', 'darkorange'], 
              edgecolor='black', linewidth=1.5)

ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Severity Model Performance Metrics', fontsize=15, fontweight='bold', pad=15)
ax.set_ylim([0, 105])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_severity_metrics_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 3_severity_metrics_summary.png")

# 5. Severity Training History (ONLY Severity)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Severity accuracy
axes[0].plot(severity_history.history['accuracy'], label='Train', linewidth=2.5, color='#3498db')
axes[0].plot(severity_history.history['val_accuracy'], label='Validation', linewidth=2.5, color='#e74c3c')
axes[0].set_title('Severity Model - Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Severity loss
axes[1].plot(severity_history.history['loss'], label='Train', linewidth=2.5, color='#3498db')
axes[1].plot(severity_history.history['val_loss'], label='Validation', linewidth=2.5, color='#e74c3c')
axes[1].set_title('Severity Model - Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Severity Model Training History', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_severity_training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 4_severity_training_history.png")

# ==================== SAVE RESULTS ====================
# Save predictions CSV
results_df.to_csv(os.path.join(output_dir, 'longitudinal_results.csv'), index=False)
print("✓ Saved: longitudinal_results.csv")

# Save summary report
with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
    f.write("="*60 + "\n")
    f.write("PPMI PARKINSON'S DISEASE PREDICTION RESULTS\n")
    f.write("="*60 + "\n\n")
    
    f.write("SEVERITY MODEL\n")
    f.write("-"*60 + "\n")
    f.write(f"Accuracy: {sev_accuracy*100:.2f}%\n")
    f.write(f"AUC: {sev_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_sev_test, y_sev_pred, target_names=severity_encoder.classes_))
    
    f.write("\n" + "="*60 + "\n\n")
    f.write("VISIT MODEL\n")
    f.write("-"*60 + "\n")
    f.write(f"Accuracy: {vis_accuracy*100:.2f}%\n")
    f.write(f"AUC: {vis_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_vis_test, y_vis_pred, target_names=visit_encoder.classes_))
    
    f.write("\n" + "="*60 + "\n")
    f.write("GENERATED FILES\n")
    f.write("-"*60 + "\n")
    f.write("1. 1_individual_patient_longitudinal.png - 30 individual patient severity over time\n")
    f.write("2. 2_combined_longitudinal_summary.png - All patients combined with trend lines\n")
    f.write("3. 3_confusion_matrices.png - Model performance matrices\n")
    f.write("1. individual_patients/ folder - Separate PNG for each patient\n")
    f.write("2. 1_combined_mean_severity_over_time.png - Mean actual vs predicted trends\n")
    f.write("3. 2_confusion_matrices.png - Severity & Visit confusion matrices\n")
    f.write("4. 3_severity_metrics_summary.png - Severity model performance metrics\n")
    f.write("5. 4_severity_training_history.png - Severity training curves\n")
    f.write("6. longitudinal_results.csv - Complete predictions with visit dates and months\n")
    f.write("7. results_summary.txt (this file)\n")

print("✓ Saved: results_summary.txt")

print("\n" + "="*80)
print("✅ EXPERIMENT COMPLETE!")
print("="*80)
print(f"All files saved in: {output_dir}")
print(f"\nSeverity Model Performance: {sev_accuracy*100:.2f}% accuracy")
print(f"Visit Model Performance: {vis_accuracy*100:.2f}% accuracy")
print(f"\nLongitudinal Analysis:")
print(f"  • {len(all_patients)} patients analyzed")
print(f"  • Individual plots: {len(all_patients)} separate PNG files in 'individual_patients/' folder")
print(f"  • Combined plot: Mean actual vs predicted severity only (no individual lines)")
print(f"  • Time scale: Months from baseline visit")
print(f"\nGenerated Files:")
print(f"  1. individual_patients/ - {len(all_patients)} patient trajectory plots")
print(f"  2. 1_combined_mean_severity_over_time.png")
print(f"  3. 2_confusion_matrices.png")
print(f"  4. 3_severity_metrics_summary.png")
print(f"  5. 4_severity_training_history.png")
print(f"  6. longitudinal_results.csv")
print(f"  7. results_summary.txt")
print("="*80)
