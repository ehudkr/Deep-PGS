import os
import pickle
from pandas import read_csv as pd_read_csv

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout
from keras.models import Model

from deepgs.parser import load_plink_text
from deepgs.transform import transform
import deepgs.model as models

# if executed from within the test directory, go one level up the tree.
if os.path.split(os.getcwd())[1] == "tests":
    os.chdir("..")

file_path = os.path.join("data", "sample_100-10000")
if os.path.exists(file_path + ".pkl"):
    G = pickle.load(open(file_path + ".pkl", "rb"))
    map_table = pd_read_csv(file_path + ".map", delim_whitespace=True,
                            header=None, names=["chr", "snp", "cm", "bp"])
else:
    g_df, map_table = load_plink_text(file_path)
    G = transform(g_df)

model = models.create_architecture_small(map_table)
model = models.compile_model(model)

GG = models.format_as_model_input(G, map_table)
model.predict(GG)


def create_dummy_architecture(map_table, output_dim=1, output_activation=None):
    """
    A small dummy architecture to be tested on the small 15-by-10 dataset.

    Parameters
    ----------
    map_table: pd.DataFrame
        PLINK's map file loaded as DataFrame.
    output_dim: int
        Output dimension (that would depend on the choice of your loss function)
    output_activation: str or keras.activations
        activation type to apply on the output layer

    Returns
    -------
        Model: keras model.
    """
    inputs = {chr_num: Input(shape=(1, 4, df.shape[0]),
                             name="input_chr{:02}".format(chr_num))
              for chr_num, df in map_table.groupby("chr")}
    conv_1 = {chr_num: Conv2D(filters=2, kernel_size=(4, 1), use_bias=True,
                              strides=1, data_format="channels_first", activation="relu",
                              name="conv_1_chr{:02}".format(chr_num))(input_layer)
              for chr_num, input_layer in inputs.items()}
    pool_1 = {chr_num: MaxPooling2D(pool_size=(1, 1),
                                    data_format="channels_first",
                                    name="maxpool_1_chr{:02}".format(chr_num))(conv_layer)
              for chr_num, conv_layer in conv_1.items()}
    # flatten = {chr_num: Flatten(data_format="channels_first")(conv_layer)
    #            for chr_num, conv_layer in pool_2.items()}
    flatten = {chr_num: Flatten(name="flatten_chr{:02}".format(chr_num))(conv_layer)
               for chr_num, conv_layer in pool_1.items()}
    dense_1 = {chr_num: Dense(units=4, activation="relu",
                              name="dense_1_chr{:02}".format(chr_num))(flat_layer)
               for chr_num, flat_layer in flatten.items()}

    merge = concatenate(list(dense_1.values()), name="concat")
    dense_2 = Dense(units=10, activation="relu",
                    name="dense_merged_1")(merge)
    dense_3 = Dense(units=10, activation="relu",
                    name="dense_merged_2")(dense_2)
    dropout_1 = Dropout(0.4, name="droupout_merge_1")(dense_3)
    output = Dense(units=output_dim, activation=output_activation,
                   name="output")(dropout_1)

    model = Model(list(inputs.values()), output)
    return model
