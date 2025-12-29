import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class PredictionRecord:
    """Data structure for training the SJF predictor"""
    prompt: str
    prompt_len: int
    actual_response_len: Optional[int] = None


class SJFLengthPredictor:
    """
    Lightweight predictor optimized for SJF scheduling decisions.
    Uses simple but effective features to predict response length.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_names = [
            'prompt_len', 'word_count', 'sentence_count', 'question_count',
            'list_indicators', 'code_indicators', 'explanation_indicators',
            'complexity_score'
        ]
    
    def _extract_features(self, records: List[PredictionRecord]) -> np.ndarray:
        """Extract features from prompt text that correlate with response length"""
        features = []
        
        for record in records:
            prompt = record.prompt.lower()
            
            # Basic length features
            prompt_len = record.prompt_len
            word_count = len(prompt.split())
            sentence_count = len(re.findall(r'[.!?]+', prompt))
            
            # Question indicators
            question_count = prompt.count('?')
            
            # List/enumeration indicators (often lead to longer responses)
            list_patterns = [
                r'\blist\b', r'\benumerate\b', r'\bsteps?\b', r'\bpoints?\b',
                r'\breasons?\b', r'\bexamples?\b', r'\bways?\b', r'\bmethods?\b'
            ]
            list_indicators = sum(1 for pattern in list_patterns if re.search(pattern, prompt))
            
            # Code/technical indicators (often lead to detailed responses)
            code_patterns = [
                r'\bcode\b', r'\bfunction\b', r'\balgorithm\b', r'\bimplement\b',
                r'\bprogram\b', r'\bscript\b', r'```', r'\bapi\b'
            ]
            code_indicators = sum(1 for pattern in code_patterns if re.search(pattern, prompt))
            
            # Explanation indicators (usually longer responses)
            explanation_patterns = [
                r'\bexplain\b', r'\bdescribe\b', r'\bdetail\b', r'\bhow\s+does\b',
                r'\bwhy\b', r'\bwhat\s+is\b', r'\btell\s+me\s+about\b'
            ]
            explanation_indicators = sum(1 for pattern in explanation_patterns if re.search(pattern, prompt))
            
            # Complexity score (combination of above factors)
            complexity_score = (
                list_indicators * 2 + 
                code_indicators * 1.5 + 
                explanation_indicators * 1.8 +
                min(question_count, 1) * 0.5
            )
            
            features.append([
                prompt_len, word_count, sentence_count, question_count,
                list_indicators, code_indicators, explanation_indicators,
                complexity_score
            ])
        
        return np.array(features, dtype=np.float32)
    
    def train(self, records: List[PredictionRecord]) -> Dict:
        """Train the predictor on historical data"""
        # Filter records with actual response lengths
        training_records = [r for r in records if r.actual_response_len is not None]
        
        if len(training_records) < 10:
            raise ValueError("Need at least 10 training samples with actual response lengths")
        
        X = self._extract_features(training_records)
        y = np.array([r.actual_response_len for r in training_records], dtype=np.float32)
        
        # Train/validation split with indices to track records
        indices = np.arange(len(training_records))
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X, y, indices, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_val = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)
        
        # Store test data for later use
        val_records = [training_records[i] for i in idx_val]
        self.test_data = {
            'X_val': X_val,
            'y_val': y_val,
            'y_pred_val': y_pred_val,
            'val_records': val_records
        }
        
        return {
            'mae': float(mae),
            'r2_score': float(r2),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'mean_actual_length': float(np.mean(y_train)),
            'test_data': self.test_data
        }
    
    def predict(self, records: List[PredictionRecord]) -> np.ndarray:
        """Predict response lengths for new prompts"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self._extract_features(records)
        predictions = self.model.predict(X)
        
        # Ensure predictions are positive
        predictions = np.maximum(predictions, 1.0)
        
        return predictions
    
    def predict_single(self, prompt: str, prompt_len: int) -> float:
        """Predict response length for a single prompt (convenience method)"""
        record = PredictionRecord(prompt=prompt, prompt_len=prompt_len)
        prediction = self.predict([record])[0]
        return float(prediction)


def load_and_train_predictor(data_path: str, max_samples: int = 5000) -> SJFLengthPredictor:
    """
    Load data and train the SJF predictor
    
    Args:
        data_path: Path to the CSV dataset
        max_samples: Maximum number of samples to use for training
    
    Returns:
        Trained SJFLengthPredictor
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    # Create training records
    records = []
    for _, row in df.iterrows():
        prompt = str(row['Prompt'])
        response = str(row['Response'])
        
        prompt_len = len(prompt.split())
        response_len = len(response.split())
        
        records.append(PredictionRecord(
            prompt=prompt,
            prompt_len=prompt_len,
            actual_response_len=response_len
        ))
    
    # Train the predictor
    predictor = SJFLengthPredictor()
    training_stats = predictor.train(records)
    
    print(f"SJF Predictor Training Results:")
    print(f"  - MAE: {training_stats['mae']:.2f} tokens")
    print(f"  - R² Score: {training_stats['r2_score']:.3f}")
    print(f"  - Training samples: {training_stats['n_train']}")
    print(f"  - Validation samples: {training_stats['n_val']}")
    print(f"  - Mean actual length: {training_stats['mean_actual_length']:.1f} tokens")
    
    return predictor


if __name__ == "__main__":
    # Example usage and testing
    try:
        predictor = load_and_train_predictor('./data/prompt_engineering_dataset.csv')
        
        # Show test results using actual validation data
        if hasattr(predictor, 'test_data'):
            print("\n" + "="*60)
            print("TEST DATA EVALUATION RESULTS")
            print("="*60)
            
            test_data = predictor.test_data
            y_actual = test_data['y_val']
            y_predicted = test_data['y_pred_val']
            val_records = test_data['val_records']
            
            # Calculate additional metrics
            mean_actual = np.mean(y_actual)
            mean_predicted = np.mean(y_predicted)
            median_actual = np.median(y_actual)
            median_predicted = np.median(y_predicted)
            
            print(f"Validation Set Statistics:")
            print(f"  - Number of test samples: {len(y_actual)}")
            print(f"  - Mean actual length: {mean_actual:.1f} tokens")
            print(f"  - Mean predicted length: {mean_predicted:.1f} tokens")
            print(f"  - Median actual length: {median_actual:.1f} tokens")
            print(f"  - Median predicted length: {median_predicted:.1f} tokens")
            
            # Show some example predictions
            print(f"\nSample Predictions (first 10 test cases):")
            print(f"{'Actual':<8} {'Predicted':<10} {'Error':<8} {'Prompt Preview'}")
            print("-" * 70)
            
            for i in range(min(10, len(val_records))):
                actual = y_actual[i]
                predicted = y_predicted[i]
                error = abs(actual - predicted)
                prompt_preview = val_records[i].prompt[:50]
                if len(val_records[i].prompt) > 50:
                    prompt_preview += "..."
                
                print(f"{actual:<8.1f} {predicted:<10.1f} {error:<8.1f} {prompt_preview}")
            
            # Error analysis
            errors = np.abs(y_actual - y_predicted)
            print(f"\nError Analysis:")
            print(f"  - Mean Absolute Error: {np.mean(errors):.2f} tokens")
            print(f"  - Median Absolute Error: {np.median(errors):.2f} tokens")
            print(f"  - Max Error: {np.max(errors):.2f} tokens")
            print(f"  - Min Error: {np.min(errors):.2f} tokens")
            print(f"  - 95th percentile error: {np.percentile(errors, 95):.2f} tokens")
            
            # # Accuracy within different thresholds
            # within_1 = np.mean(errors <= 1.0) * 100
            # within_3 = np.mean(errors <= 3.0) * 100
            # within_5 = np.mean(errors <= 5.0) * 100
            
            # print(f"\nAccuracy Analysis:")
            # print(f"  - Predictions within 1 token: {within_1:.1f}%")
            # print(f"  - Predictions within 3 tokens: {within_3:.1f}%")
            # print(f"  - Predictions within 5 tokens: {within_5:.1f}%")
    
    except FileNotFoundError:
        print("Dataset file not found. Please ensure './data/prompt_engineering_dataset.csv' exists.")
    except Exception as e:
        print(f"Error: {e}")
