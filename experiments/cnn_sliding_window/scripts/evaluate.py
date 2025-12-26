#!/usr/bin/env python3
"""Model evaluation script with comprehensive metrics and visualizations."""
import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import create_datasets_from_config


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return cm


def plot_roc_curve(y_true, y_scores, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, save_path):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return avg_precision


def evaluate_model(
    model_path: str,
    config_path: str = None,
    dataset_path: str = './dataset',
    output_dir: str = './evaluation_results',
    batch_size: int = 16,
    cpu_only: bool = False
):
    """Evaluate model on test set with comprehensive metrics.

    Args:
        model_path: Path to saved model (.keras file)
        config_path: Path to config file (optional, will infer from model_path)
        dataset_path: Path to dataset directory
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        cpu_only: Force CPU-only evaluation
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Force CPU if requested
    if cpu_only:
        tf.config.set_visible_devices([], 'GPU')
        print("✓ GPU disabled - using CPU only")

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded: {model.name}")
    print(f"  Parameters: {model.count_params():,}")

    # Load config
    if config_path is None:
        # Infer config from model path
        if 'baseline' in model_path.lower():
            config_path = 'configs/baseline.yaml'
        elif 'mobilenet' in model_path.lower():
            config_path = 'configs/mobilenet.yaml'
        else:
            config_path = 'configs/baseline.yaml'

    print(f"\nLoading config from: {config_path}")
    config = Config.from_yaml(config_path)
    config.data.dataset_path = dataset_path
    config.training.batch_size = batch_size

    # Create datasets
    print("\nCreating test dataset...")
    datasets = create_datasets_from_config(
        config=config.data,
        augmentation_config=config.augmentation,
        batch_size=batch_size,
        cache=False
    )

    test_dataset = datasets['test']
    test_size = datasets['sizes']['test']

    print(f"✓ Test samples: {test_size}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(output_dir, f"eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)

    # Evaluate model
    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)

    print("\nComputing metrics on test set...")
    test_results = model.evaluate(test_dataset, verbose=1)

    # Get predictions
    print("\nGenerating predictions...")
    y_true_onehot = []
    y_pred_probs = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_true_onehot.extend(labels.numpy())
        y_pred_probs.extend(predictions)

    y_true_onehot = np.array(y_true_onehot)
    y_pred_probs = np.array(y_pred_probs)

    # Convert one-hot to class indices
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Get positive class probabilities for ROC/PR curves
    y_scores = y_pred_probs[:, 1]  # Probability of positive class

    # Print test results
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)

    metrics_dict = {}
    for metric_name, value in zip(model.metrics_names, test_results):
        metrics_dict[metric_name] = float(value)
        print(f"{metric_name:15s}: {value:.4f}")

    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)

    class_names = ['Negative (Non-brood)', 'Positive (Brood)']
    cm = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        os.path.join(eval_dir, 'confusion_matrix.png')
    )

    print(f"\n{'':<20s} {'Predicted Negative':<20s} {'Predicted Positive':<20s}")
    print(f"{'Actual Negative':<20s} {cm[0,0]:<20d} {cm[0,1]:<20d}")
    print(f"{'Actual Positive':<20s} {cm[1,0]:<20d} {cm[1,1]:<20d}")

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    detailed_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f} (of predicted positives, how many are correct)")
    print(f"Recall:      {recall:.4f} (of actual positives, how many we caught)")
    print(f"F1-Score:    {f1_score:.4f} (harmonic mean of precision and recall)")
    print(f"Specificity: {specificity:.4f} (of actual negatives, how many we identified)")

    print(f"\nTrue Positives:  {tp} (correctly identified brood cells)")
    print(f"True Negatives:  {tn} (correctly identified non-brood cells)")
    print(f"False Positives: {fp} (non-brood cells misclassified as brood)")
    print(f"False Negatives: {fn} (brood cells missed)")

    # Classification report
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)

    # ROC Curve
    print("\n" + "="*80)
    print("ROC CURVE")
    print("="*80)

    roc_auc = plot_roc_curve(
        y_true,
        y_scores,
        os.path.join(eval_dir, 'roc_curve.png')
    )
    print(f"AUC-ROC: {roc_auc:.4f}")
    detailed_metrics['roc_auc'] = roc_auc

    # Precision-Recall Curve
    print("\n" + "="*80)
    print("PRECISION-RECALL CURVE")
    print("="*80)

    avg_precision = plot_precision_recall_curve(
        y_true,
        y_scores,
        os.path.join(eval_dir, 'precision_recall_curve.png')
    )
    print(f"Average Precision: {avg_precision:.4f}")
    detailed_metrics['average_precision'] = avg_precision

    # Save results to JSON
    results = {
        'model_path': model_path,
        'config_path': config_path,
        'test_samples': test_size,
        'timestamp': timestamp,
        'model_metrics': metrics_dict,
        'detailed_metrics': detailed_metrics,
        'classification_report': classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True
        )
    }

    results_file = os.path.join(eval_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n✓ Results saved to: {eval_dir}")
    print(f"  - evaluation_results.json")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - precision_recall_curve.png")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test Accuracy:  {accuracy:.2%}")
    print(f"Test Precision: {precision:.2%}")
    print(f"Test Recall:    {recall:.2%}")
    print(f"Test F1-Score:  {f1_score:.2%}")
    print(f"Test AUC-ROC:   {roc_auc:.2%}")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best model from hyperparameter tuning
  python scripts/evaluate.py --model-path ./hyperparameter_tuning/baseline_tuning/trial_0008/best_model.keras

  # Evaluate specific model
  python scripts/evaluate.py --model-path ./saved_models/my_model.keras --config configs/baseline.yaml

  # CPU-only evaluation
  python scripts/evaluate.py --model-path ./model.keras --cpu-only
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved model (.keras file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional, will infer from model path)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU-only evaluation'
    )

    args = parser.parse_args()

    try:
        results = evaluate_model(
            model_path=args.model_path,
            config_path=args.config,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            cpu_only=args.cpu_only
        )

        return 0

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
