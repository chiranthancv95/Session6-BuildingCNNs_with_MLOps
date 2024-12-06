# # MNIST Neural Network with MLOps Pipeline using Pytorch

A highly optimized CNN model for MNIST digit classification targeting >99.4% accuracy with less than 20K parameters.

## Project Highlights

- **High Accuracy Target**: Aims for >99.4% accuracy on the test set
- **Lightweight**: Uses less than 20K trainable parameters
- **Modern Architecture**: Implements max pooling, batch normalization, and GAP
- **Automated Testing**: Includes GitHub Actions for continuous integration

## Model Architecture

The model uses several modern deep learning techniques:
- Batch Normalization after convolutions
- Strategic dropout for regularization
- Global Average Pooling (GAP)
- Efficient channel progression (1→16→32→16→10)

### Architecture Details
The network architecture consists of the following layers:

1. Input Block
   - Conv2d(1, 16, 3) → Output: 26x26x16
   - ReLU
   - BatchNorm2d(16)

2. Convolution Block 2
   - Conv2d(16, 32, 3) → Output: 24x24x32
   - ReLU
   - BatchNorm2d(32)

3. MaxPool Block
   - MaxPool2d(2, 2) → Output: 12x12x32

4. Convolution Block 3
   - Conv2d(32, 16, 3) → Output: 10x10x16
   - ReLU
   - BatchNorm2d(16)

5. Convolution Block 4
   - Conv2d(16, 10, 3) → Output: 8x8x10
   - ReLU
   - BatchNorm2d(10)

6. Convolution Block 5
   - Conv2d(10, 10, 3) → Output: 6x6x10
   - ReLU
   - BatchNorm2d(10)

7. Global Average Pooling
   - AvgPool2d(6, 6) → Output: 1x1x10

8. Output Block
   - Conv2d(10, 10, 1) → Output: 1x1x10
   - Reshape to (Batch_Size, 10)
   - LogSoftmax

Total Parameters: 11,498
