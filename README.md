This repo contains the trained model pertaining to the paper accepted to ISIC 2018 (workshop of MICCAI 2018):

*Paper Under Review*

Some simple code is provided to perform model inference on a given image.

To get started, download the trained model files from our [latest release](https://github.com/neil454/lyme-1600-model/releases/latest), which contains 5 sets of weights, one for each training fold.

To setup your python environment, run:

    pip install numpy scipy tensorflow-gpu keras

Finally, run predict.py (if no image is given, prediction will be done on EM_example.jpg):

    python predict.py <PATH/TO/IMAGE.JPG>
