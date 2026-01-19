import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def main():
    # Load AG News dataset from HuggingFace
    print("Loading AG News dataset...")
    dataset = datasets.load_dataset('ag_news')
    
    # Use the training split
    train_data = dataset['train']
    
    # Prepare features and labels
    X = train_data['text']
    y = train_data['label']
    
    print(f"Training on {len(X)} samples...")
    
    # Create TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Train the model
    print("Training baseline model...")
    pipeline.fit(X, y)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the pipeline (includes both vectorizer and model)
    model_path = 'models/baseline_model.pkl'
    joblib.dump(pipeline, model_path)
    
    print(f"Baseline model saved to {model_path}")

if __name__ == "__main__":
    main()