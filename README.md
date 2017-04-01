RPNplus
===============

Code accompanying the paper ["Recognition in-the-Tail:Training Detectors for Unusual Pedestrians with Synthetic Imposters"](http://ml.cs.tsinghua.edu.cn:5000/publications/synunity/). As for the generator for synthetic data, please take this [repo](https://github.com/huangshiyu13/generator_synthetic_data) for reference.

## Requirement
- ubuntu or Mac OS
- tensorflow==0.12.0+(with GPU)
- pip install image
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

## Cite

Please cite our paper if you use this code or our datasets in your own work:

```
@article{huang2017recognition,
  title={Recognition in-the-Tail: Training Detectors for Unusual Pedestrians with Synthetic Imposters},
  author={Huang, Shiyu and Ramanan, Deva},
  journal={arXiv preprint arXiv:1703.06283},
  year={2017}
}
```

## Acknowledgement
* Our code is based on Yinpeng Dong's code and this repo: https://github.com/machrisaa/tensorflow-vgg

## Author
Shiyu Huang(huangsy13@gmail.com)
