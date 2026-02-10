#!/usr/bin/env python3
"""Train classifier on validated dataset"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle

class SignalFeatureExtractor:
    def extract_features(self, samples):
        features = []
        power = np.abs(samples) ** 2
        features.append(np.mean(power))
        features.append(np.std(power))
        features.append(np.max(power))
        features.append(np.min(power))
        
        fft_vals = np.fft.fft(samples)
        fft_power = np.abs(fft_vals) ** 2
        features.append(np.mean(fft_power))
        features.append(np.std(fft_power))
        features.append(np.max(fft_power))
        
        peak_freq_idx = np.argmax(fft_power)
        features.append(peak_freq_idx / len(fft_power))
        
        i_samples = np.real(samples)
        q_samples = np.imag(samples)
        features.append(np.mean(i_samples))
        features.append(np.std(i_samples))
        features.append(np.mean(q_samples))
        features.append(np.std(q_samples))
        
        phase = np.angle(samples)
        features.append(np.mean(phase))
        features.append(np.std(phase))
        
        phase_diff = np.diff(phase)
        features.append(np.mean(phase_diff))
        features.append(np.std(phase_diff))
        
        bandwidth = np.sum(fft_power > np.max(fft_power) * 0.1)
        features.append(bandwidth / len(fft_power))
        
        return np.array(features)

def load_dataset(data_dir='datasets_validated'):
    X = []
    y = []
    
    labels = sorted(os.listdir(data_dir))
    print(f"Loading {len(labels)} classes: {labels}")
    
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        print(f"  {label}: {len(files)} samples")
        
        extractor = SignalFeatureExtractor()
        for filename in files:
            filepath = os.path.join(label_dir, filename)
            data = np.load(filepath, allow_pickle=True).item()
            samples = data['samples']
            features = extractor.extract_features(samples)
            X.append(features)
            y.append(label)
    
    return np.array(X), np.array(y)

def main():
    print("="*70)
    print("TRAINING REDDIT-PROOF CLASSIFIER")
    print("="*70)
    
    X, y = load_dataset()
    print(f"\nDataset: {len(X)} samples, {len(np.unique(y))} classes")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name} ({best_score:.1%})")
    print(f"{'='*70}")
    
    y_pred = best_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(np.unique(y))
    print(f"\n{' '*15}", end='')
    for cls in classes:
        print(f"{cls[:8]:>8}", end=' ')
    print()
    for i, cls in enumerate(classes):
        print(f"{cls[:15]:>15}", end=' ')
        for j in range(len(classes)):
            print(f"{cm[i,j]:>8}", end=' ')
        print()
    
    with open('rtl_classifier_validated.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler, 'model_name': best_name}, f)
    
    print(f"\n✅ Model saved: rtl_classifier_validated.pkl")
    print(f"✅ Accuracy: {best_score:.1%}")
    print(f"✅ Ready for Reddit!")

if __name__ == '__main__':
    main()
