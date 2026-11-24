import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection

def load_and_preprocess_data(csv_path="creditcard.csv"):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Scale Amount and Time features separately
    amount_scaler = sklearn.preprocessing.StandardScaler()
    time_scaler = sklearn.preprocessing.StandardScaler()
    
    df['scaled_amount'] = amount_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = time_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Drop original unscaled columns
    df = df.drop(['Amount', 'Time'], axis=1)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return {
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'amount_scaler': amount_scaler,
        'time_scaler': time_scaler
    }