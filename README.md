# MNIST Neural Network with MLOps Pipeline using Pytorch

[![ML Pipeline](https://github.com/chiranthancv95/Session6-BuildingCNNs_with_MLOps/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/chiranthancv95/Session6-BuildingCNNs_with_MLOps/actions/workflows/ml-pipeline.yml)

A highly optimized CNN model for MNIST digit classification targeting >99.4% accuracy with less than 20K parameters under 20 epochs.

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

1. Convolution Block 1
   - Conv2d(1, 32, 3) → Output: 26x26x32
   - ReLU
   - BatchNorm2d(32)
   - Dropout(0.1)

2. Convolution Block 2
   - Conv2d(32, 32, 3) → Output: 24x24x32
   - ReLU
   - BatchNorm2d(32)
   - Dropout(0.1)

3. MaxPool Block
   - MaxPool2d(2, 2) → Output: 12x12x32

4. Convolution Block 3
   - Conv2d(32, 16, 3) → Output: 10x10x16
   - ReLU
   - BatchNorm2d(16)
   - Dropout(0.1)

5. Convolution Block 4
   - Conv2d(16, 16, 3) → Output: 8x8x16
   - ReLU
   - BatchNorm2d(16)
   - Dropout(0.1)

6. Convolution Block 5
   - Conv2d(16, 10, 3) → Output: 6x6x10
   - ReLU
   - BatchNorm2d(10)
   - Dropout(0.1)

7. Global Average Pooling
   - AvgPool2d(6, 6) → Output: 1x1x10

8. Output Block
   - Conv2d(10, 10, 1) → Output: 1x1x10
   - Reshape to (Batch_Size, 10)
   - LogSoftmax

Total Parameters: 12,062


## Setup and Installation

### Local Development

#### 1. Clone the repository:
bash
git clone https://github.com/your-username/your-repo-name.git
cd Session5

#### 2. Create and activate virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


#### 3. Install dependencies:
pip install torch torchvision tqdm pytest pytest-cov


#### 4. Run tests:
pytest tests/test_model.py -v

#### 5. Train the model:
python train.py

### GitHub Actions

The CI/CD pipeline automatically runs when:
- Pushing to main/master branch
- Creating a pull request
- Manually triggering the workflow

## Results

- Test Accuracy: >99.4% in 20 epochs
- Parameter Count: 12,062
- Training Time: ~3mins on GPU per epoch
