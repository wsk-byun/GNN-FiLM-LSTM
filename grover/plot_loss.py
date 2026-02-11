#!/usr/bin/env python3

import argparse
from itertools import dropwhile
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs, train_losses, val_losses = [], [], []
    with open(log_path, 'r') as f:
        lines = dropwhile(lambda l: not l.startswith('Epoch:'), f)
        for line in lines:
            if line.startswith('EP:'):
                break
            if not line.startswith('Epoch:'):
                continue
            parts = line.split()
            epoch = int(parts[1])
            loss_train = float(parts[3])
            loss_val   = float(parts[5])

            epochs.append(epoch)
            train_losses.append(loss_train)
            val_losses.append(loss_val)

    return epochs, train_losses, val_losses

def plot_losses(epochs, train_losses, val_losses, out_path):
    plt.figure()
    plt.plot(epochs, train_losses,
             marker='o', linestyle='-', markersize=4,
             label='Train loss')
    plt.plot(epochs, val_losses,
             marker='s', linestyle='--', markersize=4,
             label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved plot to {out_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help='Path to training log file')
    parser.add_argument('output', default='loss_plot.png', help='Output png filename')
    args = parser.parse_args()
    
    epochs, train_losses, val_losses = parse_log(args.logfile)
    if not epochs:
        print('No epochs parsed.')
        return

    plot_losses(epochs, train_losses, val_losses, args.output)

if __name__ == '__main__':
    main()







    
