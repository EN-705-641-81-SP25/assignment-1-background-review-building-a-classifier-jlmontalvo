#!/usr/bin/env python3
"""
Run single_run function to generate the training loss plot
"""

from easydict import EasyDict
from model import load_data, run, visualize_epochs
from typing import List, Tuple, Dict, Union

EMBEDDING_TYPES = ["glove-twitter-50", "glove-twitter-100", "glove-twitter-200", "word2vec-google-news-300"]

def single_run(dev_d: Dict[str, List[Union[str, int]]],
               train_d: Dict[str, List[Union[str, int]]],
               test_d: Dict[str, List[Union[str, int]]]):
    train_config = EasyDict({
        'batch_size': 64,
        'lr': 0.025,
        'num_epochs': 20,
        'save_path': 'model.pth',
        'embeddings': EMBEDDING_TYPES[0],  # glove-twitter-50
        'num_classes': 2,
    })

    print("Starting single run with glove-twitter-50 embeddings...")
    epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run(train_config, dev_d, train_d, test_d)
    visualize_epochs(epoch_train_losses, epoch_dev_loss, "single_run_loss.png")
    print("Training completed! Plot saved as 'single_run_loss.png'")

if __name__ == '__main__':
    print("Loading IMDB dataset...")
    dev_data, train_data, test_data = load_data()
    
    print("Running single training run...")
    single_run(dev_data, train_data, test_data)
    
    print("Done!")
