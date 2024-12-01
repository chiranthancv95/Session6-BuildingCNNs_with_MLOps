# MNIST Classification with PyTorch

A highly optimized CNN model for MNIST digit classification targeting >99.4% accuracy with less than 20K parameters.

## Project Highlights

- **High Accuracy Target**: Aims for >99.4% accuracy on the test set
- **Lightweight**: Uses less than 20K trainable parameters
- **Modern Architecture**: Implements skip connections, batch normalization, and GAP
- **Automated Testing**: Includes GitHub Actions for continuous integration

## Model Architecture

The model uses several modern deep learning techniques:
- Skip connections for better gradient flow
- Batch Normalization after convolutions
- Strategic dropout (p=0.08) for regularization
- Global Average Pooling (GAP)
- Efficient channel progression (1→12→16)

### Architecture Details