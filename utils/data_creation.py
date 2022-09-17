import random
import pandas as pd

from sklearn.datasets import make_regression

def make_synthetic_tabular_data(rows:int, columns:int, random_scale:bool=True, return_dataframe=True, seed=0, scale_range=(1, 15, 1)) -> dict:
    """Makes a synthetic tabular dataset for the shape of given row and columns at varying scales. 

    Args:
        rows (int): The number of rows for the dataset.
        columns (int): The number of columns with different scales for the dataset.
        random_scales (bool): Whether to vary each columns scale randomly.

    Returns:
        dict: Dataset with keys as column names and values of each key as all rows.
    """
    random.seed(seed)
    dataset = {}
    scales = {}

    x, y = make_regression(
    n_samples=rows,
    n_features = columns,
    noise=random.random()+1,
    random_state=seed
    )

    scale_range = random.sample(range(scale_range[0], scale_range[1], scale_range[2]), columns)
    for idx in range(x.shape[1]):
        # set scale
        if random_scale:
            scale = scale_range[idx]
        else:
            scale = 1
        scales[idx] = scale
        x[:,idx] = x[:,idx]+scale*(random.random()*10)
    
    if return_dataframe:
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
    
    return x, y, scales

