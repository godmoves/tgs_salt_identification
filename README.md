# Code for Kaggle TGS salt identification challenge

## Background

Several areas of Earth with large accumulations of oil and gas also have huge deposits of salt below the surface.

But unfortunately, knowing where large salt deposits are precisely is very difficult. Professional seismic imaging 
still requires expert human interpretation of salt bodies. This leads to very subjective, highly variable renderings. 
More alarmingly, it leads to potentially dangerous situations for oil and gas company drillers.

To create the most accurate seismic images and 3D renderings, TGS (the world’s leading geoscience data company) is 
hoping Kaggle’s machine learning community will be able to build an algorithm that automatically and accurately 
identifies if a subsurface target is salt or not.

## Method

(WIP)

## Usage

### Notebook version

Upload the `.ipynb` file to [Google Colab](https://colab.research.google.com), and change the runtime type to `GPU` or
`TPU`, then run all code step by step.

Notice that you need to upload your kaggle token API to download the data.

### Terminal version

#### Dependencies

- python3
- tensorflow
- numpy
- sklean
- pandas
- tqdm

Use `pip` to install what you need.

#### Command

Download and unzip all data and move them into folder `./data`.
Then just type `python main.py` in the terminal.

