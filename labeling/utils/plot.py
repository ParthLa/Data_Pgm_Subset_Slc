import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

def plot_df_bar(df: DataFrame, mode:str):
    """Plot a dataframe

    Args:
        df (DataFrame): a dataframe with numerical valued columns
        mode (str): Way to plot the bargraph
    """    
    if mode == "seperate":
        df.plot.bar(width=0.75,subplots=True)
    elif mode == "aggregate":
        df.plot.bar(width=0.75,suplots=False)
    plt.show()

