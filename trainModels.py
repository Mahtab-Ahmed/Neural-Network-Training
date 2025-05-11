import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_models(data_dir='processed_training_data', model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # Aggregate all data from the directory
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
                print(f"[INFO] Loaded: {filename} with shape {df.shape}")
            except Exception as e:
                print(f"[WARN] Failed to load {filename}: {e}")

    if not all_data:
        print("[ERROR] No valid CSV files found for training.")
        return

    data = pd.concat(all_data, ignore_index=True)

    target_columns = ['steer', 'accel', 'gear']
    feature_columns = [col for col in data.columns if col not in target_columns]

    # Save the feature columns used in training
    features_file_path = os.path.join(model_dir, "model_features.txt")
    with open(features_file_path, "w") as f:
        for col in feature_columns:
            f.write(col + "\n")
    print(f"[✓] Saved model feature names to {features_file_path}")

    models = {}
    for target in target_columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(data[feature_columns], data[target])
        models[target] = model
        model_path = os.path.join(model_dir, f"{target}_model.pkl")
        joblib.dump(model, model_path)
        print(f"[✓] Saved model for {target} to {model_path}")

if __name__ == "__main__":
    train_models()
