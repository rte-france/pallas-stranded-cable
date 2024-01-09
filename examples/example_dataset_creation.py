import os
import pandas as pd
from strdcable.overheadlines import TablesOHLconductors
from strdcable.config import cfg, __PATH_STRDCABLE__

path = __PATH_STRDCABLE__


if __name__ == "__main__":
    """
    This example reads part of a csv dataset, isolating conductors. It then creates a table object which contains
    two dataframes: one for global cable properties and one for all layer properties. 
    """

    # path to csv containing data for cables
    cables_table_path = os.path.join(path, cfg['example_files']['cables_dataset'])
    # dataframe obtained from the csv dataset
    df_cabledata = pd.read_csv(cables_table_path)

    # creating a table object with properties for all selected cables
    cables_table = TablesOHLconductors(df_cabledata)
    # creating the dataset in the table object
    cables_table.create_dataset()
    # obtaining cable data
    cables_dataframe = cables_table.dfconductors
    # obtaining layer data
    layers_dataframe = cables_table.dflayers

    # printing selected columns in the dataframe of global cable properties
    print(cables_dataframe[['cable', 'bimaterial', 'lineic_mass', 'rated_strength', 'rugosity']])

    # printing selected columns in the dataframe of layer properties, sorted by cable then by layer number
    print(layers_dataframe[['cable', 'layer', 'nature', 'dwires', 'ultimate_stress']].sort_values(by=['cable', 'layer']))
