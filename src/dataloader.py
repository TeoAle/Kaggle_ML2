import pandas as pd
import os


def reverse_ohe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    cols = [col for col in df.columns if col.startswith(key)]
    df[key] = df[cols].idxmax(axis=1).str.replace(key, '').astype(int)
    df = df.drop(columns=cols)
    return df


def load_raw_df(PATH: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    df = df.drop(['Id'], axis=1)
    return df


def load_elu_data(PATH: str) -> pd.DataFrame:
    """
    # TODO: add docstring
    """
    elu_data_raw = pd.read_csv(
        '..//data//num_to_elu.txt', sep=":", header=None)
    elu_data_raw['ELU'] = elu_data_raw[1].str[5:9]

    # compute climatic and geologic zone from ELU code
    elu_data_raw['Climatic_Zone'] = elu_data_raw['ELU'
                                                 ].str[0].astype('category')
    elu_data_raw['Geologic_Zone'] = elu_data_raw['ELU'
                                                 ].str[1].astype('category')

    elu_data_raw.drop([1, 'ELU'], axis=1, inplace=True)
    elu_data_raw.columns = ['Soil_Type', 'Climatic_Zone', 'Geologic_Zone']

    return elu_data_raw


def load_train_df(
        PATH: str,
        decode_dummies: bool = True,
        add_geo_features: bool = True  # ignored if decode_dummies is False
) -> pd.DataFrame:

    df = load_raw_df(PATH)

    if decode_dummies:
        df = reverse_ohe(df, 'Soil_Type')
        df = reverse_ohe(df, 'Wilderness_Area')

        if add_geo_features:
            elu_data = load_elu_data(PATH)
            df = df.merge(elu_data, on='Soil_Type', how='left')

        df.Soil_Type = df.Soil_Type.astype('category')
        df.Wilderness_Area = df.Wilderness_Area.astype('category')

    df.Cover_Type = df.Cover_Type.astype('category')
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].astype('float64')

    return df
