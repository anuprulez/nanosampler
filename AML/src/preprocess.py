import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self, p_data, p_mapper):
        self.p_data = p_data
        self.p_mapper = p_mapper

    def merge_data(self):
        # Implement data cleaning operations here
        df_GSE175758_GEO_processed = pd.read_csv(self.p_data, sep="\t")
        print(df_GSE175758_GEO_processed.head())

        df_GSE175758_GEO_processed_mapper = pd.read_csv(self.p_mapper, sep="\t")
        print(df_GSE175758_GEO_processed_mapper.head())