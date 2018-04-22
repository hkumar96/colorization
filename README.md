# Colorization

This repository contains code for the EE698K term project that dealt with automatic colorization of images
## Getting Started

To run the application, do the following steps
1. Simply download the repo as .zip file and extract. Or clone the repository using terminal using following command
```bash
$ git clone https://github.com/hkumar96/colorization.git
```
2. In the colorization directory, make the following directory structure 
```bash
$ mkdir Data_zoo/LaMem/lamem/images
$ mkdir logs/image_checkpoints/
$ mkdir logs/image_pred/
```
3. Add the training images into `Data_zoo/LaMem/lamem/images` folder in `.jpg` format.

### Prerequisites

The application uses following libraries:
* numpy
* cv2
* tensorflow
* skimage
* scipy
* matplotlib

### Running the application

After installing the libraries and downloading the source code, using terminal change current directory to downloaded repo.

```bash
$ cd <directory_name>/colorization
```
Run the training instance using
```bash
$ python colorize.py
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* https://github.com/shekkizh/Colorization.tensorflow

