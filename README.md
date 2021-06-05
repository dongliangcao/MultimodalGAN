# MultimodalGAN
CycleGAN used for multimodal translation in medical imaging
## Requirements
pytorch>=1.5, imageio, PIL
## Usage
### Train
python main.py --mode train
### Test
python main.py --mode test
### Optional config
--data_root: data root stores all training dataset and test dataset (e.g. trainS1, trainS2, testS1, testS2...)
--model_root: pre-trained G_A2B (generator modal A -> modal B) model used for test
--epochs: number of epochs to train
--output_root: output root stores all test outputs
