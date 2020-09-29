import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.stats as stats
import utils
import geopandas
import cartopy.crs as ccrs




def clean(year = 2016,boro_sel = ['Manhattan','Staten Island','Queens','Bronx','Brooklyn'])


