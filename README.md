# MNNMDA

# Dependencies
- pandas

- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- openpyxl

## Install MATLAB Engine API for Python

https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html 

```bash
cd "matlabroot\extern\engines\python"
python setup.py install
```

# Usage

```python
python main.py --model "MNNMDA" --alpha 1.0 --beta 100.0 --mode "CVS1" --n_splits 5 --seed 666 --comment "demo" 
```

