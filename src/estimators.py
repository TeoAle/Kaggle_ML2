import numpy as np
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from prep_utils import encode_sin_cos
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def get_estimator(est):

    # column selector
    cols_drop = [
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Hillshade_9am',
        'Hillshade_Noon'
    ]

    column_selector = FunctionTransformer(
        lambda x: x.drop(columns=cols_drop),
        feature_names_out=lambda x: x.columns.drop(cols_drop)
    ).set_output(transform='pandas')

    # ordinal encoder
    transformer_ordinal = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ).set_output(transform='pandas')

    transformer_box_cox = FunctionTransformer(
        np.log1p,
    ).set_output(transform='pandas')

    transformer_cyclical = FunctionTransformer(
        encode_sin_cos
    ).set_output(transform='pandas')

    cols_cyclical = ['Aspect']

    cols_categorical = [
        'Wilderness_Area',
        'Soil_Type',
        'Climatic_Zone',
        'Geologic_Zone'
    ]

    cols_skew = [
        'Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
    ]

    cols_scale = [
        'Elevation',
        'Slope',
        'Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
        'Hillshade_3pm',

    ]

    encoders = ColumnTransformer([
        ('categorical', transformer_ordinal, cols_categorical),
        ('cyclical', transformer_cyclical, cols_cyclical),
        ('skewed', transformer_box_cox, cols_skew)
    ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    scaler = ColumnTransformer([
        ('numerical', StandardScaler(), cols_scale)
    ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    prep = Pipeline([
        ('dropper', column_selector),
        ('feat_eng', encoders),
        ('scaler', scaler)
    ])

    model = Pipeline([
        ('prep', prep),
        ('est', est)
    ])

    return model


def objective(trial, experiment_id, X_train, y_train):
    with mlflow.start_run(experiment_id=experiment_id, nested=True):

        model = get_estimator('passthrough')

        # Categorical Encoder
        categorical_encoder_type = trial.suggest_categorical(
            'categorical_encoder', ['onehot', 'ordinal'])

        if categorical_encoder_type == 'onehot':
            categorical_encoder = OneHotEncoder(
                drop='first', handle_unknown='ignore', sparse_output=False)
        else:
            categorical_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1)

        # Skewed Encoder
        skewed_encoder_type = trial.suggest_categorical(
            'skewed_encoder', ['box-cox', 'log' 'passthrough']),

        if skewed_encoder_type == 'box-cox':
            box_cox = PowerTransformer(
                method='box-cox'
            ).set_output(transform='pandas')
            skewed_encoder = FunctionTransformer(
                lambda x: box_cox.fit_transform(x + 1),
            ).set_output(transform='pandas')
        elif skewed_encoder_type == 'log':
            skewed_encoder = FunctionTransformer(
                np.log1p
            ).set_output(transform='pandas')
        else:
            skewed_encoder = 'passthrough'

        # Cyclical Encoder
        cyclical_encoder_type = trial.suggest_categorical(
            'cyclical_encoder', ['sin_cos', 'passthrough'])

        if cyclical_encoder_type == 'sin_cos':
            cyclical_encoder = FunctionTransformer(
                encode_sin_cos).set_output(transform='pandas')
        else:
            cyclical_encoder = 'passthrough'

        # Scaler
        scaler_type = trial.suggest_categorical(
            'scaler', ['standard', 'minmax', 'passthrough'])

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = 'passthrough'

        estimator = RandomForestClassifier(verbose=False)

        param_dict = {
            'prep__feat_eng__categorical': categorical_encoder,
            'prep__feat_eng__cyclical': cyclical_encoder,
            'prep__scaler': scaler,
            'prep__feat_eng__skewed': skewed_encoder,
            'est': estimator,
            'est__n_estimators': trial.suggest_int(
                'n_estimators', 100, 500, 100),
            'est__max_depth': trial.suggest_int('max_depth', 3, 15),
            }

        # Set parameters
        model.set_params(**param_dict)

        acc = cross_val_score(model, X_train, y_train,
                              cv=3, scoring='accuracy').mean()
        f1 = cross_val_score(model, X_train, y_train,
                             cv=3, scoring='f1_macro').mean()

        metrics = {
            "accuracy": acc,
            "f1_macro": f1,
        }

        mlflow.log_params(param_dict)
        mlflow.log_metrics(metrics)
        # NOTE model is not logged to mlflow: eval if worth and then save it

        return f1
