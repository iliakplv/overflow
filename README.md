# overflow
Data Science with Stack Overflow 2018 Developer Survey data

## Get the data

1. [Download](https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey)

2. Create `data` directory and put `so_survey_results_public.csv` in it

## Data Analysis

**Python 3.6** for [pandas](https://pandas.pydata.org/) experiments

1. Install dependencies: `pip install -r requirements/analysis.txt`

2. Run Jupyter notebook: `jupyter notebook`

3. Play with `analysis.ipynb`

**Python 2.7** for [TFDV](https://github.com/tensorflow/data-validation) experiments

1. Install dependencies: `pip install -r requirements/tfdv.txt`

2. Run Jupyter notebook: `jupyter notebook`

3. Play with `tfdv.ipynb`

## Machine Learning

**Python 3.6**

1. Install dependencies: `pip install -r requirements/model.txt`

2. Generate schema and preprocess the data by running `python data.py`

3. Train and evaluate the model by running `python model.py`

## To-do

- Reframe the classification problem (current set of features doesn't seem to predict current target)

- Inspect current data split (train/eval/test) with TFDV

- Make a training/evaluation demo (ipynb)
