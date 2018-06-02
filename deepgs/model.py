from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv1D, MaxPooling1D, Flatten, concatenate, Dropout, Reshape
import keras.metrics as k_metrics


LATENT_DIM = 4


def create_architecture(map_table, output_dim=1, output_activation=None):
    """
    (conv -> pool) * 2 -> FC -> concat -> FC -> FC -> output.

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
    inputs = {chr_num: Input(shape=(1, LATENT_DIM, df.shape[0]),
                             name="input_chr{:02}".format(chr_num))
              for chr_num, df in map_table.groupby("chr")}

    # First convolution swipes the 4-dimensional embedding of the data in order to output a vector rather than a matrix:
    conv_1 = {chr_num: Conv2D(filters=32, kernel_size=(LATENT_DIM, 2048), use_bias=True,
                              strides=1, data_format="channels_first", activation="relu",
                              name="conv_1_chr{:02}".format(chr_num))(input_layer)
              for chr_num, input_layer in inputs.items()}

    # Discard redundant dimensions (since kernel_size[0] equal to data dimension the output dim is explicit 1,
    # On the way of doing that (technical reasons) transpose the result so that the channels will be the last dimension:
    reshape = {chr_num: Reshape((conv_layer.shape[3].value, conv_layer.shape[1].value))(conv_layer)
               for chr_num, conv_layer in conv_1.items()}

    pool_1 = {chr_num: MaxPooling1D(pool_size=2,
                                    name="maxpool_1_chr{:02}".format(chr_num))(conv_layer)
              for chr_num, conv_layer in reshape.items()}
    conv_2 = {chr_num: Conv1D(filters=64, kernel_size=1064, use_bias=True,
                              strides=1, activation="relu",
                              name="conv_2_chr{:02}".format(chr_num))(pool_layer)
              for chr_num, pool_layer in pool_1.items()}
    pool_2 = {chr_num: MaxPooling1D(pool_size=2,
                                    name="maxpool_2_chr{:02}".format(chr_num))(conv_layer)
              for chr_num, conv_layer in conv_2.items()}
    flatten = {chr_num: Flatten(name="flatten_chr{:02}".format(chr_num))(pool_layer)
               for chr_num, pool_layer in pool_2.items()}
    dense_1 = {chr_num: Dense(units=256, activation="relu",
                              name="dense_1_chr{:02}".format(chr_num))(flat_layer)
               for chr_num, flat_layer in flatten.items()}

    merge = concatenate(list(dense_1.values()), name="concat")
    dense_2 = Dense(units=4196, activation="relu",
                    name="dense_merged_1")(merge)
    dense_3 = Dense(units=1024, activation="relu",
                    name="dense_merged_2")(dense_2)
    dropout_1 = Dropout(0.4, name="droupout_merge_1")(dense_3)

    output = Dense(units=output_dim, activation=output_activation,
                   name="output")(dropout_1)

    # Verify that the chromosome input is defined in an ordered fashion:
    model = Model(inputs=[inputs[chr_num] for chr_num in sorted(list(inputs.keys()))],
                  outputs=output)
    return model


def create_architecture_small(map_table, output_dim=1, output_activation=None):
    """
    (conv -> pool) * 2 -> FC -> concat -> FC -> FC -> output

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
    inputs = {chr_num: Input(shape=(1, LATENT_DIM, df.shape[0]),
                             name="input_chr{:02}".format(chr_num))
              for chr_num, df in map_table.groupby("chr")}

    # First convolution swipes the 4-dimensional embedding of the data in order to output a vector rather than a matrix:
    conv_1 = {chr_num: Conv2D(filters=32, kernel_size=(LATENT_DIM, 128), use_bias=True,
                              strides=1, data_format="channels_first", activation="relu",
                              name="conv_1_chr{:02}".format(chr_num))(input_layer)
              for chr_num, input_layer in inputs.items()}

    # Discard redundant dimensions (since kernel_size[0] equal to data dimension the output dim is explicit 1,
    # On the way of doing that (technical reasons) transpose the result so that the channels will be the last dimension:
    reshape = {chr_num: Reshape((conv_layer.shape[3].value, conv_layer.shape[1].value))(conv_layer)
               for chr_num, conv_layer in conv_1.items()}

    pool_1 = {chr_num: MaxPooling1D(pool_size=2,
                                    name="maxpool_1_chr{:02}".format(chr_num))(conv_layer)
              for chr_num, conv_layer in reshape.items()}
    conv_2 = {chr_num: Conv1D(filters=64, kernel_size=64, use_bias=True,
                              strides=1, activation="relu",
                              name="conv_2_chr{:02}".format(chr_num))(pool_layer)
              for chr_num, pool_layer in pool_1.items()}
    pool_2 = {chr_num: MaxPooling1D(pool_size=2,
                                    name="maxpool_2_chr{:02}".format(chr_num))(conv_layer)
              for chr_num, conv_layer in conv_2.items()}
    flatten = {chr_num: Flatten(name="flatten_chr{:02}".format(chr_num))(pool_layer)
               for chr_num, pool_layer in pool_2.items()}
    dense_1 = {chr_num: Dense(units=256, activation="relu",
                              name="dense_1_chr{:02}".format(chr_num))(flat_layer)
               for chr_num, flat_layer in flatten.items()}

    merge = concatenate(list(dense_1.values()), name="concat")
    dense_2 = Dense(units=1024, activation="relu",
                    name="dense_merged_1")(merge)
    dense_3 = Dense(units=1024, activation="relu",
                    name="dense_merged_2")(dense_2)
    dropout_1 = Dropout(0.4, name="droupout_merge_1")(dense_3)

    output = Dense(units=output_dim, activation=output_activation,
                   name="output")(dropout_1)

    # Verify that the chromosome input is defined in an ordered fashion:
    model = Model([inputs[chr_num] for chr_num in sorted(list(inputs.keys()))], output)
    return model


def compile_model(model, optimizer="sgd", loss="mean_squared_error", metrics=None):
    metrics = metrics or [k_metrics.mean_absolute_error,
                          k_metrics.mean_absolute_percentage_error,
                          k_metrics.mean_squared_error,
                          k_metrics.mean_squared_logarithmic_error]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model


def format_as_model_input(G, map_table):
    """
    Splits the genotype dataset by chromosome into a list of genotypes to conform with the keras model input.

    Parameters
    ----------
    G: xr.DataArray
        DataArray of transformed genotypes (output of transform module).
    map_table: pd.DataFrame
        DataFrame of the corresponding PLINK map file.

    Returns
    -------
        list[np.array]: List the size of number of chromosomes in the data. Each entry being the genotypes of a specific
                        chromosome.
    """
    # Split the data by chromosomes, convert it to numpy array and reshape it to have an explicit channel depth 1:
    split_data = [G[:, :, df.index].values.reshape(G.shape[0], 1, G.shape[1], df.shape[0])
                  for chr_num, df in map_table.groupby("chr")]
    return split_data
