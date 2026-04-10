import os
import time
import warnings
import numpy as np
import pandas as pd
import tsfast
import tsfel
import pickle
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
#DATA_DIR = "testdata/Intrinsic data"
DATA_DIR = "prev-data/Dataset/Intrinsic data"
CATEGORIES = ["N", "NS", "OT", "UT"]
SIGNAL_COL = 0
CACHE_DIR = "cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def load_data():
    X = []
    y = []
    print("Loading FULL dataset...")
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, cat)
        if not os.path.exists(cat_dir):
            continue
        files = [f for f in os.listdir(cat_dir) if f.endswith(".csv")]
        print(f"  {cat}: loading {len(files)} files")
        for f in files:
            df = pd.read_csv(os.path.join(cat_dir, f))
            signal = df.iloc[:, 0].values.astype(np.float64)
            X.append(signal)
            y.append(cat)
    return X, np.array(y)

def benchmark_tsfast(X):
    rust_features = ["mean", "std", "variance", "min", "max", "energy", "rms", "autocorr_lag1", "slope", "intercept"]
    #rust_features = ["paa-10-0", "paa-10-1", "paa-10-2", "paa-10-3", "paa-10-4", "paa-10-5", "paa-10-6", "paa-10-7", "paa-10-8", "paa-10-9"]
    start = time.time()
    extracted = []
    for x in X:
        feat_vals = tsfast.extract(x, rust_features)
        extracted.append(feat_vals)
    end = time.time()
    return np.array(extracted), end - start

def get_tsfel_features(X):
    cache_path = os.path.join(CACHE_DIR, "tsfel_features.pkl")
    if os.path.exists(cache_path):
        print("  Loading TSFEL features from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print("  Extracting TSFEL features (first time)...")
    cfg = tsfel.get_features_by_domain("statistical")
    start = time.time()
    extracted = []
    for x in X:
        feat_vals = tsfel.time_series_features_extractor(cfg, x, fs=100, verbose=0)
        extracted.append(feat_vals.values[0])
    res = (np.array(extracted), time.time() - start)
    with open(cache_path, "wb") as f:
        pickle.dump(res, f)
    return res

def get_tsfresh_features(X):
    cache_path = os.path.join(CACHE_DIR, "tsfresh_features.pkl")
    if os.path.exists(cache_path):
        print("  Loading tsfresh features from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    print("  Extracting tsfresh features (first time)... This may take a while.")
    data_list = []
    for i, x in enumerate(X):
        temp_df = pd.DataFrame({"id": i, "time": np.arange(len(x)), "value": x})
        data_list.append(temp_df)
    full_df = pd.concat(data_list)
    
    start = time.time()
    extracted = extract_features(full_df, column_id="id", column_sort="time", 
                                 default_fc_parameters=EfficientFCParameters(),
                                 disable_progressbar=False,
                                 n_jobs=4) # Use more jobs for full dataset
    res = (extracted.values, time.time() - start)
    with open(cache_path, "wb") as f:
        pickle.dump(res, f)
    return res

def run_ml(X_feats, y, label=""):
    X_feats = np.nan_to_num(X_feats, nan=0.0, posinf=0.0, neginf=0.0)
    std_devs = np.std(X_feats, axis=0)
    constant_features = np.sum(std_devs == 0)
    print(f"  [{label}] Removing {constant_features} constant features")
    X_feats = X_feats[:, std_devs > 0]

    
    X_train, X_test, y_train, y_test = train_test_split(X_feats, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"  [{label}] Training on {X_train.shape}")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return acc, f1

if __name__ == "__main__":
    X_raw, y = load_data()
    counts = np.unique(y, return_counts=True)
    print(f"Total samples: {len(X_raw)}, Distribution: {dict(zip(counts[0], counts[1]))}")

    print("\nBenchmarking tsfast...")
    feats_tsfast, time_tsfast = benchmark_tsfast(X_raw)
    acc_tsfast, f1_tsfast = run_ml(feats_tsfast, y, "tsfast")
    
    print("\nBenchmarking TSFEL...")
    #feats_tsfel, time_tsfel = get_tsfel_features(X_raw)
    #acc_tsfel, f1_tsfel = run_ml(feats_tsfel, y, "TSFEL")
    
    print("\nBenchmarking tsfresh...")
    #feats_tsfresh, time_tsfresh = get_tsfresh_features(X_raw)
    #acc_tsfresh, f1_tsfresh = run_ml(feats_tsfresh, y, "tsfresh")
    
    print("\n--- Results ---")
    print(f"tsfast:  Time={time_tsfast:.4f}s, Accuracy={acc_tsfast:.4f}, Features={feats_tsfast.shape[1]}")
    #print(f"TSFEL:   Time={time_tsfel:.4f}s, Accuracy={acc_tsfel:.4f}, Features={feats_tsfel.shape[1]}")
    #print(f"tsfresh: Time={time_tsfresh:.4f}s, Accuracy={acc_tsfresh:.4f}, Features={feats_tsfresh.shape[1]}")
    
    with open("benchmark_results_full.txt", "w") as f:
        f.write(f"tsfast|{time_tsfast:.4f}|{acc_tsfast:.4f}|{f1_tsfast:.4f}|{feats_tsfast.shape[1]}\n")
        #f.write(f"TSFEL|{time_tsfel:.4f}|{acc_tsfel:.4f}|{f1_tsfel:.4f}|{feats_tsfel.shape[1]}\n")
        #f.write(f"tsfresh|{time_tsfresh:.4f}|{acc_tsfresh:.4f}|{f1_tsfresh:.4f}|{feats_tsfresh.shape[1]}\n")
