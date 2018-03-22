RPNplus
===============

Code accompanying the paper ["Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters(CVPR2017)"](https://arxiv.org/pdf/1703.06283). As for the generator for synthetic data, please take this [repo](https://github.com/huangshiyu13/generator_synthetic_data) for reference.
![](https://raw.githubusercontent.com/huangshiyu13/RPNplus/master/readme_img/rpn.jpg)
## Requirement
- ubuntu or Mac OS
- tensorflow==1.1+
- pip install image
- pip install sklearn
- pip install scipy
- [image_pylib](https://github.com/huangshiyu13/image_pylib)(This repository should be put under the same folder with RPNplus.)

## Usage
**Run Demo:**

- Download model files([RPN_model ](https://drive.google.com/file/d/0BzU4ETbYHM6fcWU3eXZHNWpZQkU/view?usp=sharing)& [VGG16_model](https://drive.google.com/file/d/0BzU4ETbYHM6fb3EyeHdXbVBSeEE/view?usp=sharing)) first, and put them in the ./models/ folder.
- The number 0 is your GPU index, and you can change to any available GPU index.
- This demo will test the images in the ./images/ folder and output the results to ./results/ folder.
```bash
python demo.py 0
```
![](https://raw.githubusercontent.com/huangshiyu13/RPNplus/master/readme_img/ladygaga.jpg)
![ATOCAR Logo](https://raw.githubusercontent.com/huangshiyu13/RPNplus/master/readme_img/Yoga.jpg)
![](https://raw.githubusercontent.com/huangshiyu13/RPNplus/master/readme_img/acrobatism.jpg)

**Train:**

- The number 0 is your GPU index, and you can change to any available GPU index.
- Open train.py and set `imageLoadDir` and `anoLoadDir` to proper values(`imageLoadDir` means where you store your images and `anoLoadDir` means where you store your annotation files).
```bash
python train.py 0
```

## Dataset Download
* [Precarious Dataset](https://drive.google.com/open?id=0BzU4ETbYHM6faEdhZ0hMNmtqUTA)
* [Synthetic Dataset](https://drive.google.com/open?id=0BzU4ETbYHM6feVM2ZE9qNzVxeHM)


## Related Datasets
* [MIKKI Dataset](https://mikki.momenta.ai/)

## Cite

Please cite our paper if you use this code or our datasets in your own work:

```
@InProceedings{Huang_2017_CVPR,
author = {Huang, Shiyu and Ramanan, Deva},
title = {Expecting the Unexpected: Training Detectors for Unusual Pedestrians With Adversarial Imposters},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

## Acknowledgement
* Our code is based on Yinpeng Dong's code and this repo: https://github.com/machrisaa/tensorflow-vgg

## Author
Shiyu Huang(huangsy13@gmail.com)
