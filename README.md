# AI-Enhanced Phishing Detection

This project implements an NLP-based phishing detection system that can analyze email content and URLs to identify potential phishing attempts. The system uses machine learning techniques to classify messages as either legitimate or phishing.

## Features
- Text preprocessing and feature extraction
- URL feature analysis
- Machine learning model for phishing classification
- Model evaluation with metrics and visualizations
- Simple web interface for interactive testing
- Easy-to-use Python API for integration

## Getting Started
### Prerequisites
Install the required packages:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib gradio
```

Download NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Project Structure
- `phishing_detector.py`: Core implementation of the phishing detection model
- `app.py`: Simple web interface using Gradio
- `phishing_dataset.csv`: Your phishing dataset (not included)

### Dataset Requirements
The model expects a CSV file with at least the following columns:
- `text`: The message content
- `is_phishing`: Binary label (0 for legitimate, 1 for phishing)

Optional column:
- `url`: URLs contained in the message

### Public Datasets
Here are some public datasets you can use for this project:

1. **UCI Phishing Websites Dataset**  
   https://archive.ics.uci.edu/ml/datasets/phishing+websites
   
2. **PhishingCorpus**  
   https://monkey.org/~jose/phishing/

3. **Mendeley Phishing Dataset**  
   https://data.mendeley.com/datasets/h3cgnj8hft/1

4. **PhishTank**  
   https://phishtank.org/developer_info.php

5. **OpenPhish**  
   https://openphish.com/

## Usage

### Training a Model

```python
from phishing_detector import PhishingDetector

# Initialize detector
detector = PhishingDetector()

# Load dataset
df = detector.load_data("phishing_dataset.csv")

# Prepare features
X, y = detector.prepare_features(df, text_column='text', url_column='url', label_column='is_phishing')

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
detector.train_model(X_train, y_train)

# Evaluate model
detector.evaluate_model(X_test, y_test)

# Save model
detector.save_model("phishing_model.pkl")
```

### Making Predictions

```python
from phishing_detector import PhishingDetector

# Load pre-trained model
detector = PhishingDetector()
detector.load_model("phishing_model.pkl")

# Make a prediction
text = "Dear customer, your account has been locked. Please verify at secure-bank.com-login.xyz"
url = "http://secure-bank.com-login.xyz/verify"

result = detector.predict(text, url)
print(result)
```

### Running the Web App

```bash
python app.py
```

This will start a local web server with a user interface for testing the phishing detection system.

## Model Improvements

Here are some ways to improve the model:

1. **Feature Engineering**:
   - Add domain age as a feature
   - Check for SSL certificate information
   - Analyze HTML and JavaScript content
   - Extract email header information

2. **Advanced Models**:
   - Try deep learning approaches (LSTM, BERT)
   - Implement ensemble methods
   - Use more sophisticated URL analysis

3. **Continuous Learning**:
   - Add feedback loop for false positives/negatives
   - Implement active learning for uncertain classifications

## Research Papers
Some research papers on phishing detection that may be helpful:
1. Basnet, R., Mukkamala, S., & Sung, A. H. (2008). Detection of phishing attacks: A machine learning approach. Soft Computing Applications in Industry, 373-383.
2. Zhuang, W., Jiang, Q., & Xiong, T. (2012, August). An intelligent anti-phishing strategy model for phishing website detection. In 2012 32nd International Conference on Distributed Computing Systems Workshops (pp. 51-56). IEEE.
3. Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection from URLs. Expert Systems with Applications, 117, 345-357.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
