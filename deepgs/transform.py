# transform well-defined rich input into numpy tensor
import numpy as np
import pandas as pd
import xarray as xr


MAP = {"A": np.array([1, 0, 0, 0]),
       "C": np.array([0, 1, 0, 0]),
       "G": np.array([0, 0, 1, 0]),
       "T": np.array([0, 0, 0, 1]),

       # Source: http://www.sbcs.qmul.ac.uk/iubmb/misc/naseq.html#300
       "R": np.array([1, 0, 1, 0]) / 2,     # Purine (adenine or guanine): R
       "Y": np.array([0, 1, 0, 1]) / 2,     # Pyrimidine (thymine or cytosine): Y
       "W": np.array([1, 0, 0, 1]) / 2,     # Adenine or thymine: W
       "S": np.array([0, 1, 1, 0]) / 2,     # Guanine or cytosine: S
       "M": np.array([1, 1, 0, 0]) / 2,     # Adenine or cytosine: M
       "K": np.array([0, 0, 1, 1]) / 2,     # Guanine or thymine: K
       "H": np.array([1, 1, 1, 0]) / 3,     # Adenine or thymine or cytosine: H
       "B": np.array([0, 1, 1, 1]) / 3,     # Guanine or cytosine or thymine: B
       "V": np.array([1, 1, 1, 0]) / 3,     # Guanine or adenine or cytosine: V
       "D": np.array([1, 0, 1, 1]) / 3,     # Guanine or adenine or thymine: D
       "N": np.array([1, 1, 1, 1]) / 4,     # Guanine or adenine or thymine or cytosine: N

       "0": np.array([0, 0, 0, 0])          # Missing
       }


def transform(G):
    """
    Transform a GWAS matrix into a 3D tensor data-set.

    Parameters
    ----------
    G: pd.DataFrame
        The output from parser.py load_plink_text().
        an (n_individual x [n_variants * 2]) DataFrame containing GWAS data-set alleles in rich-index format:
        Index are subjects' FID and IID. Columns are variants SNP and HAP (variant names and 2 alleles [haplotypes]).

    Returns
    -------
    xr.DataArray:
        A transformation of the input into a 3D tensor.
        Dimensions are annotated: individuals x embed_space x variants.
        Dim 0 coordinates - both individuals ids concatenated with '_': "{FID}_{IID}".
        Dim 2 coordinates - variant names.
    """
    G = G.applymap(lambda x: MAP[x])   # convert each entry (textual allele) to it's corresponding vector representation
    G = G.groupby(level="SNP", axis="columns").sum()    # sum the vectors of the two alleles in each snp
    # The result is a DataFrame with each entry being a vector
    # convert the resulted (2D) DataFrame of vectors to a 3D tensor:
    G_tensor = G.values.flatten()   # Flat the matrix to a vector (size n_snp * n_individuals) of vectors (size 4).
    G_tensor = np.stack(G_tensor, 0)    # Transform the vector of vectors to a 2D array (4 x [n_snp * n_individuals])
    # We want a tensor with dims (n_individual x 4 x n_snp), but that requires a detour:
    # Transform 2D matrix to a 3D tensor (n_individual x n_snp x 4):
    G_tensor = G_tensor.reshape(G.shape[0], G.shape[1], MAP["A"].size)
    G_tensor = G_tensor.transpose((0, 2, 1))    # Transform into desired (n_individual x 4 x n_snp)
    # G_tensor = np.stack(G.values.flatten(), 0).reshape(G.shape[0], G.shape[1], MAP["A"].size).transpose((0, 2, 1))

    # g_tensor = np.empty((g_annot.shape[0], MAP["A"].size, g_annot.shape[1]))
    # for i in range(g_annot.shape[0]):
    #     for j in range(g_annot.shape[1]):
    #         g_tensor[i, :, j] = g_annot.values[i, j]

    # Create a richly indexed 3D tensor:
    ids = G.reset_index()[["FID", "IID"]].astype(str)
    ids = ids["FID"].str.cat(ids["IID"], sep="_")   # squeeze the two columns id into a single column id
    G_tensor = xr.DataArray(G_tensor, coords={"subjects": ids.values,
                                              "variants": G.columns.get_level_values(0).values},
                            dims=("subjects", "embed", "variants"))
    return G_tensor

