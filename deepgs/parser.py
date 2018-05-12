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

    # Sniff for the number of columns and load only the genotype values (i.e. skipping the first 6 in ped file):
    n_cols = pd.read_csv(file_path + ".ped", delim_whitespace=True, header=None, nrows=1).shape[1]
    first_allele = 6    # The number of the first column containing genotype (skipping the ids and phenotype, sex, etc.)
    ped = pd.read_csv(file_path + ".ped",
                      delim_whitespace=True, header=None, usecols=range(first_allele, n_cols), dtype="category")
    # Re-index the genotype file:
    ped.columns = pd.MultiIndex.from_product([map_file["snp"], [0, 1]], names=["SNP", "HAP"])
    ped.index = ids.set_index(ids.columns.tolist()).index

    return ped, map_file
