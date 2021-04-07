## Test Segmentation Network on Analogy Question Images and Generate Part-based Comparison Model (PCM) Features
### Requirements
* Software: The code was developed with python 3.6, opencv-python 3.1, pytorch 1.0, torchvision 0.2
* Hardware: GPU with memory 12G or more is recommended. The default training uses 2 GPUs.

### Usage
* Download Analogy Question images from [here](https://cs.jhu.edu/~qliu24/Analogy/analogy_question.zip) and put it in the *data/analogy_question/* folder.
* Download our pre-trained segmentation model from [here](https://cs.jhu.edu/~qliu24/Analogy/deeplab_resnet101_syncars_nocroppedcar_subtype5_epoch50.pth) and put it in the *results/models/* folder.
* Run the code to test the model on the Analogy Question images:
```
python main.py --exp test --num_subtypes 5 --dataset syncars_a --vis_pred --pred_save_dir results/pred_AQ/\
  --resume results/models/deeplab_resnet101_syncars_nocroppedcar_subtype5_epoch50.pth
```
* Generate PCM features:
```
python generate_pcm_features.py
```
