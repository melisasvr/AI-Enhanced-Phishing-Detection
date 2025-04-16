import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

class PhishingDetector:
    def __init__(self):
        self.pipeline = None
        self.vectorizer = None
        self.model = None
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, filepath):
        """Load and prepare the dataset"""
        try:
            # For CSV files
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            # For other formats, add appropriate loaders here
            else:
                raise ValueError("Unsupported file format")
                
            print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
            print("Column names:", df.columns.tolist())
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and remove stopwords
        tokens = nltk.word_tokenize(text)
        tokens = [self.ps.stem(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_url_features(self, url):
        """Extract features from URLs"""
        features = {}
        
        if not isinstance(url, str):
            return {}
        
        # Length of URL
        features['url_length'] = len(url)
        
        # Count of special characters
        features['special_char_count'] = sum(c in string.punctuation for c in url)
        
        # Number of subdomains
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            features['subdomain_count'] = domain.count('.')
        except:
            features['subdomain_count'] = 0
        
        # Presence of suspicious terms
        suspicious_terms = ['login', 'secure', 'account', 'update', 'verify', 'bank', 'paypal']
        features['suspicious_term_count'] = sum(term in url.lower() for term in suspicious_terms)
        
        return features
    
    def prepare_features(self, df, text_column, url_column=None, label_column=None):
        """Prepare features for model training"""
        # Preprocess text
        print("Preprocessing text data...")
        df['processed_text'] = df[text_column].apply(self.preprocess_text)
        
        # Extract URL features if URL column is provided
        if url_column and url_column in df.columns:
            print("Extracting URL features...")
            url_features = df[url_column].apply(self.extract_url_features)
            url_features_df = pd.DataFrame(url_features.tolist())
            
            if not url_features_df.empty:
                # Combine with processed text
                X = pd.concat([df['processed_text'], url_features_df], axis=1)
            else:
                X = df['processed_text']
        else:
            X = df['processed_text']
        
        # Prepare target variable if provided
        y = None
        if label_column and label_column in df.columns:
            y = df[label_column]
        
        return X, y
    
    def analyze_data(self, df, text_column, label_column):
        """Perform exploratory data analysis"""
        print("\n--- Data Analysis ---")
        
        # Basic stats
        print(f"Dataset shape: {df.shape}")
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Text length distribution
        df['text_length'] = df[text_column].apply(lambda x: len(str(x)))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='text_length', hue=label_column, kde=True)
        plt.title('Distribution of Text Length by Class')
        plt.xlabel('Text Length')
        plt.ylabel('Count')
        plt.savefig('text_length_distribution.png')
        
        # Class distribution
        if label_column in df.columns:
            plt.figure(figsize=(8, 6))
            df[label_column].value_counts().plot(kind='bar')
            plt.title('Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.savefig('class_distribution.png')
            
            print("\nClass distribution:")
            print(df[label_column].value_counts(normalize=True))
        
        # Most common words in phishing vs. legitimate
        if label_column in df.columns:
            from collections import Counter
            
            # Words in phishing
            phishing_texts = ' '.join(df[df[label_column] == 1]['processed_text'].tolist())
            phishing_words = Counter(phishing_texts.split())
            print("\nTop 10 words in phishing texts:")
            print(dict(phishing_words.most_common(10)))
            
            # Words in legitimate
            legit_texts = ' '.join(df[df[label_column] == 0]['processed_text'].tolist())
            legit_words = Counter(legit_texts.split())
            print("\nTop 10 words in legitimate texts:")
            print(dict(legit_words.most_common(10)))
    
    def train_model(self, X_train, y_train):
        """Train the phishing detection model"""
        print("\n--- Training Model ---")
        
        if isinstance(X_train, pd.DataFrame) and 'processed_text' in X_train.columns:
            text_features = X_train['processed_text']
            
            # Separate numerical features if they exist
            numerical_features = X_train.drop('processed_text', axis=1) if X_train.shape[1] > 1 else None
            
            # TF-IDF Vectorization for text
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_text_features = self.vectorizer.fit_transform(text_features)
            
            # Combine with numerical features if they exist
            if numerical_features is not None and not numerical_features.empty:
                X_numerical = numerical_features.values
                X_combined = np.hstack((X_text_features.toarray(), X_numerical))
            else:
                X_combined = X_text_features
            
            # Train RandomForest model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_combined, y_train)
            
        else:
            # If only text features
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            self.pipeline.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        print("\n--- Model Evaluation ---")
        
        if self.pipeline:
            y_pred = self.pipeline.predict(X_test)
            proba = self.pipeline.predict_proba(X_test)[:, 1]
        else:
            # Process the same way as in training
            if isinstance(X_test, pd.DataFrame) and 'processed_text' in X_test.columns:
                text_features = X_test['processed_text']
                numerical_features = X_test.drop('processed_text', axis=1) if X_test.shape[1] > 1 else None
                
                X_text_features = self.vectorizer.transform(text_features)
                
                if numerical_features is not None and not numerical_features.empty:
                    X_numerical = numerical_features.values
                    X_combined = np.hstack((X_text_features.toarray(), X_numerical))
                else:
                    X_combined = X_text_features
                
                y_pred = self.model.predict(X_combined)
                proba = self.model.predict_proba(X_combined)[:, 1]
            else:
                # Should not get here normally
                print("Error: Test data format doesn't match training data")
                return
        
        # Print metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')
        
        # Feature importance if using RandomForest directly
        if self.model:
            if hasattr(self.model, 'feature_importances_'):
                if self.vectorizer:
                    # Get feature names from vectorizer
                    feature_names = self.vectorizer.get_feature_names_out()
                    
                    # Only look at text features importance
                    n_text_features = len(feature_names)
                    importances = self.model.feature_importances_[:n_text_features]
                    
                    # Get top features
                    indices = np.argsort(importances)[-20:]  # Top 20 features
                    
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(len(indices)), importances[indices])
                    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                    plt.xlabel('Feature Importance')
                    plt.title('Top 20 Text Features')
                    plt.tight_layout()
                    plt.savefig('feature_importance.png')
    
    def save_model(self, filename='phishing_detector_model.pkl'):
        """Save the trained model"""
        if self.pipeline:
            joblib.dump(self.pipeline, filename)
        else:
            # Save both vectorizer and model
            model_data = {
                'vectorizer': self.vectorizer,
                'model': self.model
            }
            joblib.dump(model_data, filename)
        print(f"\nModel saved as {filename}")
    
    def load_model(self, filename='phishing_detector_model.pkl'):
        """Load a trained model"""
        try:
            loaded = joblib.load(filename)
            if isinstance(loaded, dict):
                self.vectorizer = loaded['vectorizer']
                self.model = loaded['model']
                self.pipeline = None
            else:
                self.pipeline = loaded
                self.vectorizer = None
                self.model = None
            print(f"Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text, url=None):
        """Make prediction on new data"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if self.pipeline:
            # If using pipeline
            prediction = self.pipeline.predict([processed_text])[0]
            probability = self.pipeline.predict_proba([processed_text])[0][1]
        else:
            # If using separate vectorizer and model
            # Extract URL features if URL is provided
            url_features = {}
            if url:
                url_features = self.extract_url_features(url)
                
            # Vectorize text
            X_text = self.vectorizer.transform([processed_text])
            
            if url_features:
                # Convert URL features to array
                X_url = np.array([[v for v in url_features.values()]])
                
                # Combine features
                X_combined = np.hstack((X_text.toarray(), X_url))
                
                # Make prediction
                prediction = self.model.predict(X_combined)[0]
                probability = self.model.predict_proba(X_combined)[0][1]
            else:
                # Text only prediction
                prediction = self.model.predict(X_text)[0]
                probability = self.model.predict_proba(X_text)[0][1]
        
        return {
            'is_phishing': bool(prediction),
            'phishing_probability': float(probability),
            'confidence': 'High' if abs(probability - 0.5) > 0.4 else 'Medium' if abs(probability - 0.5) > 0.2 else 'Low'
        }


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = PhishingDetector()
    
    # Example with your own dataset
    # Replace with your dataset path
    dataset_path = "phishing_dataset.csv"
    
    # Load data - adjust column names to match your dataset
    df = detector.load_data(dataset_path)
    
    if df is not None:
        # Assuming columns: 'text', 'url', 'is_phishing' (0/1)
        text_col = 'text'
        url_col = 'url'
        label_col = 'is_phishing'
        
        # Analyze data
        detector.analyze_data(df, text_col, label_col)
        
        # Prepare features
        X, y = detector.prepare_features(df, text_col, url_col, label_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        detector.train_model(X_train, y_train)
        
        # Evaluate model
        detector.evaluate_model(X_test, y_test)
        
        # Save model
        detector.save_model()
        
        # Example prediction with new data
        sample_text = "Dear customer, your account has been locked. Please update your information at secure-bank.com-login.xyz"
        sample_url = "http://secure-bank.com-login.xyz/update"
        
        result = detector.predict(sample_text, sample_url)
        print("\nSample prediction:")
        print(f"Text: {sample_text}")
        print(f"URL: {sample_url}")
        print(f"Result: {result}")