"""
Main script for PathMNIST classification (Part 1).
Students should implement the TODO sections to achieve >99% accuracy.
"""

import sys, json
import os

from classification_dataset import create_pathmnist_dataloaders
from classification_models import get_model, count_parameters
from classification_train import train_model

def main(args):
    print("=== CPH 100A Project 2 - Part 1: PathMNIST Classification ===")
    
    # Load data
    print("Loading PathMNIST dataset...")
    train_loader, val_loader, num_classes = create_pathmnist_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
        data_root='./data'
    )
    
    print(f"\nCreating {args.model_name} model...")
    model = get_model(args.model_name, num_classes=num_classes)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    print(f"\nTraining {args.model_name} model...")
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps_per_epoch=args.max_steps_per_epoch #TODO: Change for your full runs
        )
        
        print("Training completed successfully!")
        
        # Show final results
        if history['val_acc']:
            best_val_acc = max(history['val_acc'])
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            
            results = {"best_val_accuracy": best_val_acc}

            if hasattr(args, "results_path"):
                os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
                with open(args.results_path, "w") as f:
                    json.dump(results, f)
                print(f"Saved validation accuracy to {args.results_path}")
            else:
                print("No results_path provided; skipping JSON save.")
        
    except NotImplementedError as e:
        print(f"❌ Training failed: {e}")
        return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CPH 100A Project 2 - PathMNIST Classification')
    parser.add_argument('--model_name', type=str, default='cnn',
                       choices=['mlp', 'cnn'], #TODO: add your models names here
                       help='Model to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for regularization')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
    parser.add_argument('--results_path', type=str, default=None,
                    help='Path to save experiment results (used by dispatcher)')
    parser.add_argument('--max_steps_per_epoch', type=int, default=100,
                    help='Maximum number of steps (batches) per epoch for faster exploration')
    args = parser.parse_args()
    
    main(args) 