# import pandas_plink
import dask.dataframe as dd
import pandas as pd


def load_plink_text(file_path):
    """
    Loads a PLINK in textual format (ped + map files), and returns a rich-format (indexed) DataFrame.

    SNP         snp_0    snp_1
    HAP           0  1     0  1
    FID  IID
    2105 2105     T  T     C  C
    5375 5375     T  T     C  C
    3217 3217     T  T     C  C
    5088 5088     C  T     T  C

    Parameters
    ----------
    file_path: str
        Path to the location of the PLINK data-set files.
        As in PLINK, the filename should be provided without the extensions. The two corresponding map and ped files
        should have the same name (but with different extensions).
        For example: if the files are "/study/gwas.ped" and "/study/gwas.map" then "/study/gwas" should be provided.

    Returns
    -------
    ped: pd.DataFrame
        A (num_individuals x num_variants) DataFrame.
        The index: MultiIndex of FID and IID of each individual. variants names.
        The columns: MultiIndex of variants and haplotypes, i.e. level 0 is variant names and under it two columns -
        one for each allele.
    map_file: pd.DataFrame
        PLINK map file.

    """
    # Load the map file:
    map_file = pd.read_csv(file_path + ".map", delim_whitespace=True,
                           header=None, names=["chr", "snp", "cm", "bp"])

    # Load the individual id values (the first two columns of ped file)
    ids = pd.read_csv(file_path + ".ped", delim_whitespace=True,
                      header=None, usecols=[0, 1], names=["FID", "IID"])
    # # Squeeze the MultiIndex id into a single index:
    # ids = ids["FID"].str.cat(ids["IID"], sep="+")
    # ids.name = "ID"

    # Sniff for the number of columns and load only the genotype values (i.e. skipping the first 6 in ped file):
    n_cols = pd.read_csv(file_path + ".ped", delim_whitespace=True, header=None, nrows=1).shape[1]
    first_allele = 6    # The number of the first column containing genotype (skipping the ids and phenotype, sex, etc.)
    ped = pd.read_csv(file_path + ".ped",
                      delim_whitespace=True, header=None, usecols=range(first_allele, n_cols), dtype="category")
    # Re-index the genotype file:
    ped.columns = pd.MultiIndex.from_product([map_file["snp"], [0, 1]], names=["SNP", "HAP"])
    ped.index = ids.set_index(ids.columns.tolist()).index

    return ped, map_file


# def load_plink_text(file_path, in_memory=True):
#     map_file = pd.read_csv(file_path + ".map", delim_whitespace=True,
#                            header=None, names=["chr", "snp", "cm", "bp"])
#
#     ids = pd.read_csv(file_path + ".ped", delim_whitespace=True, header=None, usecols=[0, 1], names=["FID", "IID"])
#     # ids = ids["FID"].str.cat(ids["IID"], sep="+")
#     # ids.name = "ID"
#
#     n_cols = pd.read_csv(file_path + ".ped", delim_whitespace=True, header=None, nrows=1).shape[1]
#     first_allele = 6
#
#     ped = pd.read_csv(file_path + ".ped",
#                       delim_whitespace=True, header=None, usecols=range(first_allele, n_cols), dtype="category")
#     ped.columns = pd.MultiIndex.from_product([map_file["snp"], [0, 1]])
#     ped.index = ids.set_index(ids.columns.tolist()).index
#
#     if in_memory:
#         return ped, map_file
#     else:
#         genotype = dd.read_csv(file_path + ".ped",
#                                delim_whitespace=True, header=None, usecols=range(first_allele, n_cols),
#                                dtype="category")
#
#         return genotype, map_file, ids


# def get_subject_ids(fam, sep="_"):
#     fam.columns = fam.columns.str.upper()
#     subject_ids = fam["FID"].str.cat(fam["IID"], sep=sep)
#     subject_ids.name = "ID"
#     return subject_ids
#
#
# def get_annotated_genotype(bim, fam, bed):
#     subject_ids = get_subject_ids(fam)
#     bed = dd.from_dask_array(bed, columns=subject_ids)
#     # bed = bed.rename(index=dict(zip(bed.index, bim["snp"])))
#     # bed = bed.rename(index=dict(zip(bed.index, bim[["chrom", "snp"]])))
#     # bed = bed.set_index(bim["snp"])
#     bed = bed.assign(snp=bim["snp"]).set_index("snp")
#     return bed
#
#
# def load_plink_binary(file_path):
#     """
#     Loads a set of plink files in binary format (bim, fam, map files).
#
#     Parameters
#     ----------
#     file_path: str
#         Path to binary plink-formatted set of files.
#         Do not include files' extension (such as (bim, fam).
#
#     Returns
#     -------
#     bim: DataFrame
#         Alleles.
#     fam: DataFrame
#         Samples.
#     bed: dask_array
#         Genotypes.
#     """
#     bim, fam, bed = pandas_plink.read_plink(file_path)
#     return bim, fam, bed
