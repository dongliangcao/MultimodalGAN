# MultimodalGAN
CycleGAN used for multimodal MRI image translation
## Requirements
pytorch>=1.5, torchvision, PIL, SimpleITK
## Usage
### Prepare Data
downloand data from [here](https://drive.google.com/file/d/1oujSbBfMQZyCUDXIBCX-ZkXhMkCAll_C/view?usp=sharing) and unzip it outside the code folder

### Train
python main.py --mode train

optional (visualize training process in TensorBoard): tensorbooard --logdir logs/
### Test
python main.py --mode test
### Optional config
--data_root: data root stores all training dataset and test dataset (e.g. trainT1, trainT2, testT1, testT2...)

--model_root: pre-trained G_A2B (generator modal A -> modal B) model used for test

--epochs: number of epochs to train

--output_root: output root stores all test outputs

--ssim: use ssim as reconstruction loss
