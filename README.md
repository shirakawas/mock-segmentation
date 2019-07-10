# Evolutionary Image Segmentation Based on Multiobjective Clustering

This is a Python implementation of the following paper:

[Shinichi Shirakawa and Tomoharu Nagao, "Evolutionary Image Segmentation Based on Multiobjective Clustering," Proceedings of the 2009 IEEE Congress on Evolutionary Computation (CEC 2009), pp. 2466-2473 (2009).](http://shiralab.ynu.ac.jp/data/paper/cec09_shirakawa.pdf)

If you use this code for your research, please cite our paper:

```
@inproceedings{ShirakawaCEC2009,
    author = {Shinichi Shirakawa and Tomoharu Nagao},
    title = {Evolutionary Image Segmentation Based on Multiobjective Clustering},
    booktitle = {Proceedings of the 2009 IEEE Congress on Evolutionary Computation (CEC 2009)},
    pages = {2466--2473},
    year = {2009}
}
```

## Requirements
We tested the codes on the following environment:

* Python 3.6.0
* Python package version
	* NumPy 1.16.2
	* SciPy 0.19.1
	* Matplotlib  2.0.2
	* cv2 3.4.0
	* Numba 0.35.0
	* DEAP 1.3

## Usage

* Run the python script as `python mock_segmentation.py`
* In the default setting, the program loads `paprika.png` as the input image and uses RGB color space
* After execution, the result (output images and a graph) is saved in `./out/`
* If you want to use another image file, please add `-i` option as `python mock_segmentation.py -i your_image.png`
* If you want to use L*a*b* color space, please add `-c` option as `python mock_segmentation.py -c Lab`
* If you want to run the code with a different setting, please directly modify the script (parameters are set in the beginning of `mock_segmentation.py`)
