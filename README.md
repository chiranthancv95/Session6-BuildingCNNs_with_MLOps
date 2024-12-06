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


## Test Logs

epoch:  1
loss=0.024014929309487343 batch_id=2:   0%|          | 1/469 [00:00<00:48,  9.61it/s]Train Epoch: 1 [0/60000 (0%)]	Loss: 0.010228
loss=0.02772822417318821 batch_id=104:  22%|██▏       | 103/469 [00:05<00:17, 21.00it/s] Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.024542
loss=0.05305032059550285 batch_id=204:  44%|████▎     | 205/469 [00:10<00:12, 20.48it/s]Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.011755
loss=0.017784442752599716 batch_id=303:  65%|██████▍   | 303/469 [00:15<00:10, 15.38it/s]Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.018023
loss=0.028166037052869797 batch_id=404:  86%|████████▌ | 404/469 [00:20<00:03, 20.89it/s]Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.029899
loss=0.014716924168169498 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.65it/s]Training Accuracy on Epoch 1: 59464/60000 (99.11%)

Test set: Average loss: 0.0204, Accuracy: 9929/10000 (99.29%)


Test set: Average loss: 0.0204, Accuracy: 9929/10000 (99.290000%)

epoch:  2
loss=0.030370373278856277 batch_id=2:   0%|          | 2/469 [00:00<00:32, 14.36it/s]Train Epoch: 2 [0/60000 (0%)]	Loss: 0.029677
loss=0.02657270058989525 batch_id=104:  22%|██▏       | 103/469 [00:05<00:16, 21.63it/s] Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.041567
loss=0.028205113485455513 batch_id=204:  44%|████▎     | 205/469 [00:10<00:12, 20.89it/s]Train Epoch: 2 [25600/60000 (43%)]	Loss: 0.051009
loss=0.011306379921734333 batch_id=304:  65%|██████▍   | 303/469 [00:16<00:08, 19.22it/s]Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.005887
loss=0.046336885541677475 batch_id=404:  86%|████████▌ | 404/469 [00:20<00:03, 21.56it/s]Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.010697
loss=0.10978298634290695 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.61it/s]Training Accuracy on Epoch 2: 59471/60000 (99.12%)

Test set: Average loss: 0.0211, Accuracy: 9933/10000 (99.33%)


Test set: Average loss: 0.0211, Accuracy: 9933/10000 (99.330000%)

epoch:  3
loss=0.02020110934972763 batch_id=1:   0%|          | 1/469 [00:00<01:18,  6.00it/s] Train Epoch: 3 [0/60000 (0%)]	Loss: 0.020131
loss=0.0120161771774292 batch_id=104:  22%|██▏       | 103/469 [00:05<00:16, 22.13it/s] Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.009605
loss=0.04097449779510498 batch_id=204:  43%|████▎     | 204/469 [00:10<00:12, 21.17it/s]  Train Epoch: 3 [25600/60000 (43%)]	Loss: 0.020418
loss=0.009573469869792461 batch_id=303:  65%|██████▍   | 304/469 [00:16<00:08, 19.41it/s]Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.052187
loss=0.01235639862716198 batch_id=404:  86%|████████▌ | 403/469 [00:20<00:03, 21.92it/s] Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.009688
loss=0.0067975823767483234 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.80it/s]Training Accuracy on Epoch 3: 59505/60000 (99.17%)

Test set: Average loss: 0.0211, Accuracy: 9935/10000 (99.35%)


Test set: Average loss: 0.0211, Accuracy: 9935/10000 (99.350000%)

epoch:  4
loss=0.003936056047677994 batch_id=0:   0%|          | 1/469 [00:00<01:09,  6.77it/s]Train Epoch: 4 [0/60000 (0%)]	Loss: 0.003936
loss=0.01662367768585682 batch_id=104:  22%|██▏       | 105/469 [00:05<00:16, 21.56it/s]Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.027208
loss=0.013915895484387875 batch_id=204:  43%|████▎     | 204/469 [00:09<00:12, 21.66it/s]Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.030557
loss=0.008075988851487637 batch_id=303:  65%|██████▍   | 303/469 [00:15<00:08, 19.50it/s]Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.026047
loss=0.008608952164649963 batch_id=404:  86%|████████▋ | 405/469 [00:20<00:02, 21.64it/s]Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.019043
loss=0.01619519479572773 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.03it/s]Training Accuracy on Epoch 4: 59485/60000 (99.14%)

Test set: Average loss: 0.0192, Accuracy: 9944/10000 (99.44%)


Test set: Average loss: 0.0192, Accuracy: 9944/10000 (99.440000%)

epoch:  5
loss=0.010085005313158035 batch_id=2:   0%|          | 2/469 [00:00<00:32, 14.47it/s]Train Epoch: 5 [0/60000 (0%)]	Loss: 0.015011
loss=0.0274409968405962 batch_id=104:  22%|██▏       | 103/469 [00:04<00:17, 21.40it/s] Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.127384
loss=0.04963379353284836 batch_id=203:  43%|████▎     | 204/469 [00:09<00:15, 16.91it/s]Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.022073
loss=0.07224424183368683 batch_id=303:  64%|██████▍   | 302/469 [00:15<00:07, 21.45it/s] Train Epoch: 5 [38400/60000 (64%)]	Loss: 0.019171
loss=0.01867007464170456 batch_id=404:  86%|████████▌ | 404/469 [00:19<00:03, 21.43it/s]Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.023499
loss=0.06507457047700882 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.22it/s]Training Accuracy on Epoch 5: 59486/60000 (99.14%)

Test set: Average loss: 0.0214, Accuracy: 9932/10000 (99.32%)


Test set: Average loss: 0.0214, Accuracy: 9932/10000 (99.320000%)

epoch:  6
loss=0.027975408360362053 batch_id=2:   0%|          | 2/469 [00:00<00:32, 14.36it/s]Train Epoch: 6 [0/60000 (0%)]	Loss: 0.012378
loss=0.007701841648668051 batch_id=104:  22%|██▏       | 104/469 [00:04<00:17, 21.03it/s]Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.028479
loss=0.010286466218531132 batch_id=202:  43%|████▎     | 202/469 [00:10<00:17, 14.84it/s]Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.017915
loss=0.02136816456913948 batch_id=303:  65%|██████▍   | 303/469 [00:15<00:07, 21.75it/s]Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.030659
loss=0.023122591897845268 batch_id=404:  86%|████████▋ | 405/469 [00:20<00:03, 21.30it/s]Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.022509
loss=0.033433929085731506 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.66it/s]Training Accuracy on Epoch 6: 59544/60000 (99.24%)

Test set: Average loss: 0.0197, Accuracy: 9935/10000 (99.35%)


Test set: Average loss: 0.0197, Accuracy: 9935/10000 (99.350000%)

epoch:  7
loss=0.017363643273711205 batch_id=2:   0%|          | 2/469 [00:00<00:31, 14.76it/s]Train Epoch: 7 [0/60000 (0%)]	Loss: 0.009204
loss=0.005220447201281786 batch_id=102:  22%|██▏       | 102/469 [00:05<00:23, 15.78it/s]Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.044096
loss=0.14407365024089813 batch_id=204:  44%|████▎     | 205/469 [00:11<00:12, 21.36it/s]Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.050563
loss=0.02461986243724823 batch_id=303:  65%|██████▍   | 304/469 [00:16<00:08, 20.19it/s]Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.014437
loss=0.026274647563695908 batch_id=403:  86%|████████▌ | 403/469 [00:21<00:03, 17.32it/s]Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.019106
loss=0.02488061785697937 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 18.96it/s]Training Accuracy on Epoch 7: 59534/60000 (99.22%)

Test set: Average loss: 0.0204, Accuracy: 9938/10000 (99.38%)


Test set: Average loss: 0.0204, Accuracy: 9938/10000 (99.380000%)

epoch:  8
loss=0.023396793752908707 batch_id=1:   0%|          | 1/469 [00:00<00:49,  9.48it/s]Train Epoch: 8 [0/60000 (0%)]	Loss: 0.020138
loss=0.006153527181595564 batch_id=104:  22%|██▏       | 105/469 [00:04<00:16, 21.72it/s]Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.031490
loss=0.02423950284719467 batch_id=204:  44%|████▎     | 205/469 [00:10<00:12, 21.19it/s]Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.011029
loss=0.004300405271351337 batch_id=303:  65%|██████▍   | 304/469 [00:15<00:08, 20.55it/s]Train Epoch: 8 [38400/60000 (64%)]	Loss: 0.012454
loss=0.0679585337638855 batch_id=402:  86%|████████▌ | 403/469 [00:20<00:04, 14.75it/s]Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.021117
loss=0.05275838077068329 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.66it/s]Training Accuracy on Epoch 8: 59550/60000 (99.25%)

Test set: Average loss: 0.0189, Accuracy: 9945/10000 (99.45%)


Test set: Average loss: 0.0189, Accuracy: 9945/10000 (99.450000%)

epoch:  9
loss=0.022362085059285164 batch_id=2:   0%|          | 2/469 [00:00<00:33, 13.94it/s]Train Epoch: 9 [0/60000 (0%)]	Loss: 0.045888
loss=0.015183770097792149 batch_id=104:  22%|██▏       | 104/469 [00:04<00:16, 22.16it/s]Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.005878
loss=0.029631586745381355 batch_id=204:  43%|████▎     | 203/469 [00:10<00:12, 21.00it/s]Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.013017
loss=0.0059848977252841 batch_id=304:  65%|██████▌   | 305/469 [00:15<00:07, 21.04it/s]Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.015970
loss=0.018013497814536095 batch_id=403:  86%|████████▌ | 403/469 [00:21<00:03, 18.12it/s]Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.033395
loss=0.0063308957032859325 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.38it/s]Training Accuracy on Epoch 9: 59528/60000 (99.21%)

Test set: Average loss: 0.0173, Accuracy: 9943/10000 (99.43%)


Test set: Average loss: 0.0173, Accuracy: 9943/10000 (99.430000%)

epoch:  10
loss=0.0118715800344944 batch_id=2:   0%|          | 2/469 [00:00<00:30, 15.51it/s]  Train Epoch: 10 [0/60000 (0%)]	Loss: 0.025668
loss=0.05255088582634926 batch_id=103:  22%|██▏       | 104/469 [00:05<00:23, 15.66it/s]Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.007208
loss=0.015357122756540775 batch_id=204:  43%|████▎     | 204/469 [00:10<00:12, 21.45it/s]Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.024121
loss=0.026160655543208122 batch_id=304:  65%|██████▍   | 303/469 [00:15<00:07, 21.44it/s]Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.009505
loss=0.0234760120511055 batch_id=404:  86%|████████▌ | 403/469 [00:20<00:03, 19.65it/s]  Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.018248
loss=0.002634796081110835 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.52it/s]Training Accuracy on Epoch 10: 59545/60000 (99.24%)

Test set: Average loss: 0.0192, Accuracy: 9940/10000 (99.40%)


Test set: Average loss: 0.0192, Accuracy: 9940/10000 (99.400000%)

epoch:  11
loss=0.013538988307118416 batch_id=2:   0%|          | 1/469 [00:00<00:47,  9.94it/s]Train Epoch: 11 [0/60000 (0%)]	Loss: 0.034079
loss=0.02400517836213112 batch_id=102:  22%|██▏       | 102/469 [00:05<00:26, 13.60it/s]Train Epoch: 11 [12800/60000 (21%)]	Loss: 0.022877
loss=0.023765839636325836 batch_id=204:  43%|████▎     | 204/469 [00:10<00:13, 20.10it/s]Train Epoch: 11 [25600/60000 (43%)]	Loss: 0.012909
loss=0.0140724191442132 batch_id=304:  65%|██████▍   | 303/469 [00:15<00:08, 20.08it/s]  Train Epoch: 11 [38400/60000 (64%)]	Loss: 0.047134
loss=0.02788660116493702 batch_id=404:  86%|████████▋ | 405/469 [00:21<00:03, 20.87it/s]Train Epoch: 11 [51200/60000 (85%)]	Loss: 0.015043
loss=0.00750223733484745 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.29it/s]Training Accuracy on Epoch 11: 59573/60000 (99.29%)

Test set: Average loss: 0.0192, Accuracy: 9941/10000 (99.41%)


Test set: Average loss: 0.0192, Accuracy: 9941/10000 (99.410000%)

epoch:  12
loss=0.007340218871831894 batch_id=1:   0%|          | 1/469 [00:00<00:50,  9.34it/s]Train Epoch: 12 [0/60000 (0%)]	Loss: 0.012537
loss=0.010545887053012848 batch_id=103:  22%|██▏       | 102/469 [00:05<00:18, 20.16it/s]Train Epoch: 12 [12800/60000 (21%)]	Loss: 0.020926
loss=0.041968345642089844 batch_id=204:  43%|████▎     | 204/469 [00:10<00:12, 21.78it/s]Train Epoch: 12 [25600/60000 (43%)]	Loss: 0.049168
loss=0.010107995010912418 batch_id=303:  65%|██████▍   | 304/469 [00:15<00:10, 16.00it/s]Train Epoch: 12 [38400/60000 (64%)]	Loss: 0.011966
loss=0.027761168777942657 batch_id=404:  86%|████████▌ | 403/469 [00:21<00:03, 21.71it/s]Train Epoch: 12 [51200/60000 (85%)]	Loss: 0.022972
loss=0.05379658564925194 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.35it/s]Training Accuracy on Epoch 12: 59555/60000 (99.26%)

Test set: Average loss: 0.0189, Accuracy: 9938/10000 (99.38%)


Test set: Average loss: 0.0189, Accuracy: 9938/10000 (99.380000%)

epoch:  13
loss=0.02138565294444561 batch_id=2:   0%|          | 2/469 [00:00<00:32, 14.48it/s]Train Epoch: 13 [0/60000 (0%)]	Loss: 0.049692
loss=0.027828121557831764 batch_id=104:  22%|██▏       | 104/469 [00:05<00:17, 21.46it/s]Train Epoch: 13 [12800/60000 (21%)]	Loss: 0.007873
loss=0.03155240789055824 batch_id=204:  43%|████▎     | 203/469 [00:10<00:12, 20.76it/s] Train Epoch: 13 [25600/60000 (43%)]	Loss: 0.050490
loss=0.02635227143764496 batch_id=302:  65%|██████▍   | 303/469 [00:15<00:11, 15.02it/s]Train Epoch: 13 [38400/60000 (64%)]	Loss: 0.024444
loss=0.0750853419303894 batch_id=404:  86%|████████▌ | 404/469 [00:20<00:03, 21.30it/s]  Train Epoch: 13 [51200/60000 (85%)]	Loss: 0.045784
loss=0.01255252119153738 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.67it/s]Training Accuracy on Epoch 13: 59562/60000 (99.27%)

Test set: Average loss: 0.0202, Accuracy: 9942/10000 (99.42%)


Test set: Average loss: 0.0202, Accuracy: 9942/10000 (99.420000%)

epoch:  14
loss=0.01076086051762104 batch_id=2:   0%|          | 2/469 [00:00<00:32, 14.55it/s] Train Epoch: 14 [0/60000 (0%)]	Loss: 0.011549
loss=0.004547318443655968 batch_id=104:  22%|██▏       | 105/469 [00:05<00:16, 21.55it/s]Train Epoch: 14 [12800/60000 (21%)]	Loss: 0.008348
loss=0.036565739661455154 batch_id=204:  43%|████▎     | 204/469 [00:10<00:12, 20.70it/s]Train Epoch: 14 [25600/60000 (43%)]	Loss: 0.034584
loss=0.008657818660140038 batch_id=304:  65%|██████▍   | 304/469 [00:16<00:08, 19.76it/s]Train Epoch: 14 [38400/60000 (64%)]	Loss: 0.030988
loss=0.009510001167654991 batch_id=404:  86%|████████▋ | 405/469 [00:20<00:02, 21.60it/s]Train Epoch: 14 [51200/60000 (85%)]	Loss: 0.025635
loss=0.053149398416280746 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.60it/s]Training Accuracy on Epoch 14: 59556/60000 (99.26%)

Test set: Average loss: 0.0244, Accuracy: 9929/10000 (99.29%)


Test set: Average loss: 0.0244, Accuracy: 9929/10000 (99.290000%)

epoch:  15
loss=0.007508186157792807 batch_id=0:   0%|          | 1/469 [00:00<01:25,  5.50it/s]Train Epoch: 15 [0/60000 (0%)]	Loss: 0.007508
loss=0.013168876990675926 batch_id=104:  22%|██▏       | 104/469 [00:05<00:17, 21.42it/s]Train Epoch: 15 [12800/60000 (21%)]	Loss: 0.020497
loss=0.01587562821805477 batch_id=204:  43%|████▎     | 203/469 [00:10<00:12, 20.97it/s] Train Epoch: 15 [25600/60000 (43%)]	Loss: 0.007201
loss=0.010535025037825108 batch_id=303:  65%|██████▍   | 303/469 [00:15<00:08, 19.48it/s]Train Epoch: 15 [38400/60000 (64%)]	Loss: 0.032399
loss=0.0768662765622139 batch_id=404:  86%|████████▌ | 403/469 [00:20<00:03, 21.32it/s] Train Epoch: 15 [51200/60000 (85%)]	Loss: 0.022118
loss=0.00859697163105011 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.65it/s]Training Accuracy on Epoch 15: 59570/60000 (99.28%)

Test set: Average loss: 0.0223, Accuracy: 9927/10000 (99.27%)


Test set: Average loss: 0.0223, Accuracy: 9927/10000 (99.270000%)

epoch:  16
loss=0.06447727233171463 batch_id=2:   0%|          | 2/469 [00:00<00:32, 14.51it/s] Train Epoch: 16 [0/60000 (0%)]	Loss: 0.014788
loss=0.019100964069366455 batch_id=103:  22%|██▏       | 104/469 [00:04<00:17, 21.05it/s]Train Epoch: 16 [12800/60000 (21%)]	Loss: 0.005924
loss=0.005157134961336851 batch_id=204:  43%|████▎     | 203/469 [00:09<00:12, 20.59it/s]Train Epoch: 16 [25600/60000 (43%)]	Loss: 0.054251
loss=0.0631936639547348 batch_id=304:  65%|██████▍   | 303/469 [00:15<00:07, 21.08it/s]  Train Epoch: 16 [38400/60000 (64%)]	Loss: 0.060175
loss=0.03350675106048584 batch_id=404:  86%|████████▌ | 402/469 [00:20<00:03, 21.21it/s]Train Epoch: 16 [51200/60000 (85%)]	Loss: 0.021391
loss=0.05511141195893288 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 20.05it/s]Training Accuracy on Epoch 16: 59561/60000 (99.27%)

Test set: Average loss: 0.0207, Accuracy: 9928/10000 (99.28%)


Test set: Average loss: 0.0207, Accuracy: 9928/10000 (99.280000%)

epoch:  17
loss=0.0301892701536417 batch_id=1:   0%|          | 1/469 [00:00<00:47,  9.88it/s]  Train Epoch: 17 [0/60000 (0%)]	Loss: 0.036337
loss=0.006460652686655521 batch_id=104:  22%|██▏       | 104/469 [00:05<00:18, 19.93it/s]Train Epoch: 17 [12800/60000 (21%)]	Loss: 0.009278
loss=0.004071793053299189 batch_id=203:  43%|████▎     | 203/469 [00:10<00:16, 16.38it/s]Train Epoch: 17 [25600/60000 (43%)]	Loss: 0.044044
loss=0.007604601327329874 batch_id=304:  65%|██████▍   | 303/469 [00:16<00:08, 20.33it/s]Train Epoch: 17 [38400/60000 (64%)]	Loss: 0.013502
loss=0.04935906082391739 batch_id=404:  86%|████████▋ | 405/469 [00:21<00:03, 20.74it/s]Train Epoch: 17 [51200/60000 (85%)]	Loss: 0.034624
loss=0.006511567626148462 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 18.84it/s]Training Accuracy on Epoch 17: 59550/60000 (99.25%)

Test set: Average loss: 0.0218, Accuracy: 9938/10000 (99.38%)


Test set: Average loss: 0.0218, Accuracy: 9938/10000 (99.380000%)

epoch:  18
loss=0.02757839299738407 batch_id=2:   0%|          | 1/469 [00:00<00:47,  9.92it/s]Train Epoch: 18 [0/60000 (0%)]	Loss: 0.031652
loss=0.005205744877457619 batch_id=104:  22%|██▏       | 105/469 [00:05<00:17, 20.53it/s]Train Epoch: 18 [12800/60000 (21%)]	Loss: 0.007060
loss=0.016547448933124542 batch_id=202:  43%|████▎     | 202/469 [00:10<00:17, 15.33it/s]Train Epoch: 18 [25600/60000 (43%)]	Loss: 0.019447
loss=0.035997673869132996 batch_id=303:  64%|██████▍   | 302/469 [00:15<00:07, 21.18it/s]Train Epoch: 18 [38400/60000 (64%)]	Loss: 0.012091
loss=0.004821793641895056 batch_id=404:  86%|████████▌ | 404/469 [00:20<00:02, 21.91it/s] Train Epoch: 18 [51200/60000 (85%)]	Loss: 0.012538
loss=0.08121426403522491 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.42it/s]Training Accuracy on Epoch 18: 59545/60000 (99.24%)

Test set: Average loss: 0.0206, Accuracy: 9936/10000 (99.36%)


Test set: Average loss: 0.0206, Accuracy: 9936/10000 (99.360000%)

epoch:  19
loss=0.018327292054891586 batch_id=2:   0%|          | 2/469 [00:00<00:32, 14.33it/s]Train Epoch: 19 [0/60000 (0%)]	Loss: 0.015435
loss=0.027623962610960007 batch_id=104:  22%|██▏       | 104/469 [00:04<00:16, 21.69it/s]Train Epoch: 19 [12800/60000 (21%)]	Loss: 0.033990
loss=0.03376195207238197 batch_id=204:  43%|████▎     | 203/469 [00:10<00:14, 18.34it/s] Train Epoch: 19 [25600/60000 (43%)]	Loss: 0.024435
loss=0.04118270054459572 batch_id=304:  65%|██████▍   | 304/469 [00:15<00:08, 19.63it/s] Train Epoch: 19 [38400/60000 (64%)]	Loss: 0.021415
loss=0.021419083699584007 batch_id=403:  86%|████████▌ | 404/469 [00:20<00:03, 17.51it/s]Train Epoch: 19 [51200/60000 (85%)]	Loss: 0.015473
loss=0.020774537697434425 batch_id=468: 100%|██████████| 469/469 [00:24<00:00, 19.29it/s]Training Accuracy on Epoch 19: 59547/60000 (99.25%)

Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.37%)


Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.370000%)