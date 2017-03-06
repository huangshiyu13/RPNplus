RPNplus
===============

Code accompanying the paper ["Recognition in-the-Tail:Training Detectors for Unusual Pedestrians with Synthetic Imposters"](http://ml.cs.tsinghua.edu.cn:5000/publications/synunity/){:target="_blank"}

##Requirement
- ubuntu or Mac OS
- tensorflow==0.12.0+(with GPU)
- pip install image
- [image_pylib](https://github.com/huangshiyu13/image_pylib)(This repository should be put under the same folder with RPNplus.)

##Usage
**Run Demo:**

- Download model files([RPN_model ](https://drive.google.com/file/d/0BzU4ETbYHM6fcWU3eXZHNWpZQkU/view?usp=sharing)& [VGG16_model](https://drive.google.com/file/d/0BzU4ETbYHM6fb3EyeHdXbVBSeEE/view?usp=sharing)) first, and put them in the ./models/ folder.
- The number 0 is your GPU index, and you can change to any available GPU index.
- This demo will test the images in the ./images/ folder and output the results to ./results/ folder.
```bash
python demo.py 0
```

**Train:**

- The number 0 is your GPU index, and you can change to any available GPU index.
- Open train.py and set `imageLoadDir` and `anoLoadDir` to proper values(`imageLoadDir` means where you store your images and `anoLoadDir` means where you store your annotation files).
```bash
python train.py 0
```
