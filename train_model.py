from phishing_detector import PhishingDetector
import pandas as pd
import os

# Create a sample dataset if one doesn't exist
if not os.path.exists('phishing_dataset.csv'):
    print("Creating a sample dataset for training...")
    data = {
        'text': [
            "Dear customer, your account has been locked. Please update your information at secure-bank.com-login.xyz",
            "Hi Sarah, here are the meeting notes from yesterday. Let me know if you need anything else.",
            "URGENT: Your account will be suspended. Click here to verify: secure-login.bank-verification.com",
            "Your Amazon package has been shipped. Track it here: amazon.com/orders",
            "You've won $1,000,000! Claim now at lottery-winner.xyz",
            "Dear user, your PayPal account needs verification. Login at paypal-secure-login.com",
            "Hello, your invoice for this month's subscription is ready. View at billing.company.com",
            "Security alert! Your bank account was accessed from an unknown device. Verify at bank-secure.com",
            "Hi John, thanks for your order! Track your package at legit-store.com/tracking",
            "Your account will be deactivated unless you confirm your details at account-verification.xyz"
        ],
        'url': [
            "http://secure-bank.com-login.xyz/update",
            "",
            "http://secure-login.bank-verification.com",
            "https://amazon.com/orders",
            "http://lottery-winner.xyz",
            "http://paypal-secure-login.com",
            "https://billing.company.com",
            "http://bank-secure.com",
            "https://legit-store.com/tracking",
            "http://account-verification.xyz"
        ],
        'is_phishing': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv('phishing_dataset.csv', index=False)
    print("Sample dataset created!")

# Initialize the detector
detector = PhishingDetector()

# Load dataset
print("Loading dataset...")
df = detector.load_data('phishing_dataset.csv')

if df is not None:
    print("Dataset loaded successfully!")
    # Prepare features
    text_col = 'text'
    url_col = 'url'
    label_col = 'is_phishing'
    
    # First prepare features (this will create the processed_text column)
    X, y = detector.prepare_features(df, text_col, url_col, label_col)
    
    # Then analyze data
    detector.analyze_data(df, text_col, label_col)
    
    # Check if dataset is too small for splitting
    if len(df) < 10:  # Arbitrary threshold for minimum dataset size
        print("Dataset is too small for train-test split. Training on full dataset...")
        detector.train_model(X, y)  # Train on full dataset
    else:
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        print("Training model...")
        detector.train_model(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        detector.evaluate_model(X_test, y_test)
    
    # Save model
    print("Saving model...")
    detector.save_model()
    
    # Verify model was saved
    if os.path.exists('phishing_detector_model.pkl'):
        print("Model saved successfully at phishing_detector_model.pkl")
    else:
        print("ERROR: Model file not found after saving!")
else:
    print("Failed to load dataset!")