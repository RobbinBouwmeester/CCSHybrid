"""
Main code used to energy minimize structures with obabel and calculate features

For the library versions see the .yml file
"""


__author__ = ["Robbin Bouwmeester", "Hans Vissers"]
__license__ = "Apache License, Version 2.0"
__maintainer__ = ["Robbin Bouwmeester", "Hans Vissers"]
__email__ = ["Robbin Bouwmeester", "Hans Vissers"]
__credits__ = [
    "Robbin Bouwmeester",
    "Hans Vissers"
]


import subprocess
import os
import re
import argparse

from multiprocessing import freeze_support
from multiprocessing import Pool

from argparse import Namespace

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from rdkit.Chem.rdmolfiles import MolToMolFile
from rdkit.Chem.rdmolfiles import MolFromPDBFile

def lreplace(pattern, sub, string):
    return re.sub('^%s' % pattern, sub, string)


def generate_3d_mol(inchi_file, outfile_name):
    """
    Function that allows to go from inchi to a mol file.

    Parameters
    ----------
    inchi_file : str
        path to file inchi
    outfile_name : str
        path where to write mol file
    """
    smiles = open(inchi_file).readlines()[0].strip()
    mol = Chem.MolFromSmiles(smiles)
    MolToMolFile(mol,outfile_name)


def generate_3d_mol_obgen(inchi_file, outfile_name):
    """
    Function that allows to go from inchi to a mol file with obgen.

    Parameters
    ----------
    inchi_file : str
        path to file inchi
    outfile_name : str
        path where to write mol file
    """
    args = ["obgen", inchi_file, " > ", outfile_name]
    p = subprocess.Popen(args)
    p.wait()


def generate_conformers_rdkit_inchi(inchi_str, outfile_name, num_conf=2):
    """
    Function that allows to generate multiple conformers from an inchi.

    Parameters
    ----------
    inchi_str : str
        string that contains a smiles
    outfile_name : str
        path where to write mol files of conformers
    num_conf : int
        number of conformers to generate
    """
    m = Chem.MolFromSmiles(inchi_str)

    try:
        m = Chem.AddHs(m)
    except BaseException:
        pass
    
    try:
        ids = AllChem.EmbedMultipleConfs(
            m, numConfs=num_conf, params=AllChem.ETKDG())
    except:
        return False

    writer = Chem.SDWriter(outfile_name)

    for i, conf in enumerate(m.GetConformers()):
        tm = Chem.Mol(m, False, conf.GetId())
        prop = AllChem.MMFFGetMoleculeProperties(tm, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(tm, prop)
        writer.write(tm)

    writer.close()
    
    return True

def generate_conformers_rdkit(mol_file, outfile_name, num_conf=2):
    """
    Function that allows to generate multiple conformers from a mol file.

    Parameters
    ----------
    mol_file : str
        path to mol file
    outfile_name : str
        path where to write mol files of conformers
    num_conf : int
        number of conformers to generate
    """
    m = Chem.MolFromMolFile(mol_file)
    print("Read M")
    try:
        m = Chem.AddHs(m)
    except BaseException:
        pass
    print("Added H")
    ids = AllChem.EmbedMultipleConfs(
        m, numConfs=num_conf, params=AllChem.ETKDG())
    print("Got multiple confs")
    writer = Chem.SDWriter(outfile_name)

    for i, conf in enumerate(m.GetConformers()):
        tm = Chem.Mol(m, False, conf.GetId())
        prop = AllChem.MMFFGetMoleculeProperties(tm, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(tm, prop)
        writer.write(tm)

    writer.close()


def generate_conformers(mol_file, outfile_name, num_conf=5, score="energy"):
    """
    Function that allows to generate multiple conformers from a mol file.

    Parameters
    ----------
    mol_file : str
        path to mol file
    outfile_name : str
        path where to write mol files of conformers
    num_conf : int
        number of conformers to generate
    score : str
        type of scoring to use for minimization
    """
    args = [
        "obabel",
        mol_file,
        "-o",
        "sdf",
        "-O",
        outfile_name,
        "--conformer",
        "--nconf",
        num_conf,
        "--score",
        score,
        "--writefoncormers"]
    args = list(map(str, (args)))
    p = subprocess.Popen(args)
    p.wait()


def split_out_sdf(sdf_file, outfile_names="structure_rmsd_*.mol"):
    """
    Function that allows to write an sdf to multiple mol files

    Parameters
    ----------
    sdf_file : str
        path to sdf file
    outfile_name : str
        path where to write mol files
    """
    args = ["obabel", sdf_file, "-o", "mol", "-O", outfile_names]
    p = subprocess.Popen(args)
    p.wait()


def energy_min_step(
        mol_file,
        outfile_name,
        force_field="gaff",
        min_change_val=1e-04,
        num_steps=50):
    """
    Function that allows to energy minimize structures

    Parameters
    ----------
    mol_file : str
        path to mol file
    outfile_name : str
        path where to write mol files
    force_field : str
        force fields to use
    min_change_val : float
        minimal delta energy to stop optimizing
    num_steps : int
        number of steps for optimization
    """
    args = [
        "obminimize",
        "-sd",
        "-c",
        min_change_val,
        "-o",
        "pdb",
        "-n",
        num_steps,
        "-ff",
        force_field,
        mol_file]
    args = list(map(str, (args)))
    f = open(outfile_name, "w")

    print(" ".join(args))
    
    p = subprocess.call(args,stdout=f)
    f.close()


def mol_to_pdb(mol_file, outfile_name):
    """
    Function that allows to convert a mol file to pdb

    Parameters
    ----------
    mol_file : str
        path to mol file
    outfile_name : str
        path where to write mol files
    """
    args = ["obabel", mol_file, "-o", "pdb", "-O", outfile_name]
    p = subprocess.Popen(args)
    p.wait()


def driftscope(
        pdb_file,
        outfile,
        tolerance=0.0001,
        gas_val=1,
        driftscope_loc=""):
    """
    Function that allows to run driftscope

    Parameters
    ----------
    pdb_file : str
        path to mol file
    outfile : str
        path where to write mol files
    tolerance : float
        delta to stop optimization
    gas_val : int
        correction factor for the gas
    driftscope_loc : str
        location of the driftscope executable
    """
    atom_types_loc = os.path.join(driftscope_loc, "docs/atom_types/types.txt")
    args = [
        os.path.join(
            driftscope_loc,
            "lib/project.exe"),
        "-pdb",
        pdb_file,
        "-types",
        atom_types_loc,
        "-tol",
        tolerance,
        "-gas",
        gas_val,
        "-out",
        outfile]
    args = list(map(str, (args)))
    print(" ".join(args))
    p = subprocess.Popen(args)
    p.wait()


def replace_hetatm_with_atm(infile):
    """
    Make sure there are no HETATM in the pdb file

    Parameters
    ----------
    infile : str
        path to a pdb file
    """
    infile_content = open(infile).readlines()
    outfile_content = []
    for line in infile_content:
        outfile_content.append(lreplace("HETATM", "ATOM  ", line))

    outfile = open(infile, "w")
    for line in outfile_content:
        outfile.write(line)

    outfile.close()


def parse_driftscope(infile):
    """
    Function that parses driftscope

    Parameters
    ----------
    infile : str
        path to driftscope out file
    """
    return max([float(line.lstrip("STRUCT_CCS	").rstrip())
                  for line in open(infile) if line.startswith("STRUCT_CCS")])


def run_pipeline_single_structure(
        identifier,
        inchi_file,
        num_conf=2,
        num_steps=5000,
        run_driftscope=True,
        run_obabel_conformers=True,
        rem_temp_files=True,
        feature_prefix="conformer_",
        driftscope_loc="C:/DriftScope/"):
    """
    Function that allows to run driftscope

    Parameters
    ----------
    identifier : str
        path to mol file
    inchi_file : str
        path to inchi file
    num_conf : int
        number of conformers to generate
    num_steps : int
        number of steps for energy minimization
    run_driftscope : boolean
        run driftscope?
    run_obabel_conformers : boolean
        use rdkit or obabel conformers? Set to true for obabel conformers
    rem_temp_files : boolean
        remove temporary intermediate files?
    feature_prefix : str
        string to append to features
    driftscope_loc : str
        location of the driftscope executable
    """
    ret_dict = {}
    conformer_filename = "temp/" + identifier + "_conformer.sdf"
    splitoutsdf_filename = "temp/" + identifier + "_conformer_energy_*.mol"

    if run_obabel_conformers:
        generate_3d_mol(inchi_file, "temp/" + identifier + ".mol")
        generate_conformers(
            "temp/" +
            identifier +
            ".mol",
            conformer_filename,
            num_conf=num_conf)
    else:
        inchi = open(inchi_file).readlines()[0].rstrip()
        ret_code = generate_conformers_rdkit_inchi(
                    inchi, conformer_filename, num_conf=num_conf)
        print(ret_code)
        if not ret_code:
            return ret_dict

    split_out_sdf(conformer_filename, outfile_names=splitoutsdf_filename)
    files_for_energy_min = []
    for i in range(num_conf + 1):
        if os.path.exists(splitoutsdf_filename.replace("*", str(i))):
            files_for_energy_min.append(
                splitoutsdf_filename.replace(
                    "*", str(i)))

    for f in files_for_energy_min:
        struct_optimized_filename = f.rstrip(".mol") + "_opt.pdb"
        projection_app_filename = "temp/" + identifier + "_PA.txt"

        energy_min_step(f, struct_optimized_filename, num_steps=num_steps)
        replace_hetatm_with_atm(struct_optimized_filename)
        conformer_num = f.rstrip(".mol").split("_")[-1]
        if run_driftscope:
            driftscope(
                struct_optimized_filename,
                projection_app_filename,
                driftscope_loc=driftscope_loc)
            ret_dict[feature_prefix+conformer_num] = parse_driftscope(
                "temp/" + identifier + "_PA.txt")
        else:
            ret_dict[conformer_num] = 1

        if rem_temp_files:
            # TODO Due to alternative paths files are not always present... Fix
            # this in a better way
            try:
                os.remove("temp/" + identifier + "_PA.txt")
            except BaseException:
                pass
            try:
                os.remove(f)
            except BaseException:
                pass
            if run_driftscope:
                try:
                    os.remove(f.rstrip(".mol") + "_opt.pdb")
                except BaseException:
                    pass

    if rem_temp_files:
        try:
            os.remove(conformer_filename)
        except BaseException:
            pass
        try:
            os.remove("temp/" + identifier + ".mol")
        except BaseException:
            pass

    for i in range(1, num_conf + 1):
        if feature_prefix+str(i) not in ret_dict.keys():
            ret_dict[feature_prefix+str(i)] = 0.0

    return ret_dict


def run_df(run_tuple, num_conf=200, num_steps=20000):
    """
    Function that allows to run the energy minimzation with a dataframe

    Parameters
    ----------
    run_tuple : tuple
        tuple with the dataframe to run and the arguments for energy minimization
    num_conf : int
        number of conformers to generate
    num_steps : int
        number of steps for energy minimization
    """
    df, argu = run_tuple
    df_dict = {}

    for index_name, row in df.iterrows():
        inchi_f_name = "temp/" + str(row["index"]) + ".smi"
        temp_inchi = open(inchi_f_name, "w")
        temp_inchi.write(row["InChI Code"])
        temp_inchi.close()

        df_dict[str(row["index"])] = run_pipeline_single_structure(str(row["index"]),
                                                                   inchi_f_name,
                                                                   run_obabel_conformers=argu.run_obabel_conformers,
                                                                   num_conf=argu.num_conf,
                                                                   rem_temp_files=argu.rem_temp_files,
                                                                   run_driftscope=argu.run_driftscope,
                                                                   num_steps=argu.num_steps,
                                                                   driftscope_loc=argu.driftscope_loc)
        if argu.rem_temp_files:
            os.remove(inchi_f_name)
    return pd.DataFrame(df_dict).T


def parallelize_dataframe(df, func, cli_arguments, n_cores=256, batch_size=512):
    """
    Parallelize energy minimization on a DF

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe with compounds to energy minimize
    func : <function>
        function to run parallel
    cli_arguments : str
        arguments to run for energy minimization
    n_cores : int
        number of cores to run in parallel
    batch_size : int
        batch size
    """
    if not os.path.isdir("./temp"):
        os.mkdir("./temp")

    try: df_split_first = np.array_split(df, len(df.index)/batch_size)
    except: df_split_first = [df]
    
    for df_index,df_batch in enumerate(df_split_first):
        df_split = np.array_split(df_batch, n_cores)

        run_tuple = list(zip(df_split, [cli_arguments for i in range(n_cores)]))

        pool = Pool(n_cores)
        df = pd.concat(pool.map(func, run_tuple))
        pool.close()
        pool.join()
        df.to_csv("temp/minimized_structs_batch_%s.csv" % (df_index))

    concat_batches = []
    for f in os.listdir("temp/"):
        if not f.startswith("minimized_structs_batch"):
            continue
        concat_batches.append(pd.read_csv(os.path.join("temp/",f)))
    df = pd.concat(concat_batches,axis=0)

    return df


def parse_argument():
    """
    Read arguments from the command line

    Parameters
    ----------

    Returns
    -------

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--driftscope_loc",
        type=str,
        dest="driftscope_loc",
        default="C:/Users/asus/Desktop/DriftScope/",
        help="Indicate the location of DriftScope for the Projection Approximation approach")

    parser.add_argument(
        "--infile_name",
        type=str,
        dest="infile_name",
        default="CCS_consolidated_sample.csv",
        help="Indicate the location of DriftScope for the Projection Approximation approach")

    parser.add_argument(
        "--outfile_name",
        type=str,
        dest="outfile_name",
        default="ModellingResults.csv",
        help="Indicate the location of DriftScope for the Projection Approximation approach")

    parser.add_argument(
        "--num_steps",
        type=int,
        dest="num_steps",
        default=20000,
        help="The maximum number of steps to take for energy minimization")

    parser.add_argument("--num_conf",
                        type=int,
                        dest="num_conf",
                        default=200,
                        help="Number of structural conformers to generate")

    parser.add_argument("--n_jobs",
                        type=int,
                        dest="n_jobs",
                        default=256,
                        help="Number of threads to spawn")

    parser.add_argument(
        "--run_driftscope",
        action="store_true",
        help="Flag to indicate to run DriftScope Projection Approximation approach")

    parser.add_argument(
        "--run_obabel_conformers",
        action="store_true",
        help="Flag that indicates if obabel should be used for generating conformers")

    parser.add_argument(
        "--rem_temp_files",
        action="store_true",
        help="Flag that indicates to remove temporary intermediate files")

    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    results = parser.parse_args()

    return(results)

def main():
    argu = parse_argument()
    df = pd.read_csv(argu.infile_name, encoding="latin1")

    df_results = parallelize_dataframe(
        df, run_df, cli_arguments=argu, n_cores=argu.n_jobs)
    df_results.to_csv(argu.outfile_name)

def get_feat(df,n_jobs=256):
    """
    Calculate features from energy minimized structures

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe with compounds to energy minimize
    """
    freeze_support()
    args = Namespace(driftscope_loc='C:/DriftScope/', infile_name='CCS_consolidated_sample.csv', n_jobs=n_jobs, num_conf=200, num_steps=10000, outfile_name='ModellingResults.csv', rem_temp_files=False, run_driftscope=False, run_obabel_conformers=False)
    df_results = parallelize_dataframe(
        df, run_df, args, n_cores=n_jobs)

    pdb_files = [os.path.join("temp/",f) for f in os.listdir("temp/") if f.endswith(".pdb")]
    mols = [MolFromPDBFile(pdb) for pdb in pdb_files]
    names = [f.lstrip("temp/").rstrip(".pdb") for f,m in zip(pdb_files,mols) if m != None]
    mols = [m for m in mols if m != None]
    mols = pd.Series(mols,index=names)

    n_all = set(Calculator(descriptors, ignore_3D=False).descriptors)
    n_2D = set(Calculator(descriptors, ignore_3D=True).descriptors)
    features_3D = n_all.difference(n_2D)
    print("calculator")
    print(features_3D)

    calc = Calculator(features_3D)
    print("calculator - pandas + ")
    df_ret = calc.pandas(mols,nproc=256)
    df_ret.index = names

    df_ret = df_ret.apply(pd.to_numeric, errors='coerce')
    df_ret.fillna(-1000.0,inplace=True)

    idents = list(set([n.split("_conformer_")[0] for n in df_ret.index]))

    feat_dict = {}
    for i in idents:
        sel = [ident for ident in df_ret.index if ident.startswith(i+"_conformer_")]
        sub_df = df_ret.loc[sel,:]
        sub_df_median = sub_df.median()
        sub_df_mean = sub_df.mean()
        sub_df_min = sub_df.min()
        sub_df_max = sub_df.max()
        sub_df_std = sub_df.std()
        sub_df_10 = sub_df.quantile(q=0.1)
        sub_df_20 = sub_df.quantile(q=0.2)
        sub_df_30 = sub_df.quantile(q=0.3)
        sub_df_40 = sub_df.quantile(q=0.4)
        sub_df_60 = sub_df.quantile(q=0.6)
        sub_df_70 = sub_df.quantile(q=0.7)
        sub_df_80 = sub_df.quantile(q=0.8)
        sub_df_90 = sub_df.quantile(q=0.9)

        sub_df_median.index = [n+"|median" for n in sub_df_median.index]
        sub_df_mean.index = [n+"|mean" for n in sub_df_mean.index]
        sub_df_min.index = [n+"|min" for n in sub_df_min.index]
        sub_df_max.index = [n+"|max" for n in sub_df_max.index]
        sub_df_std.index = [n+"|std" for n in sub_df_std.index]

        sub_df_10.index = [n+"|q10" for n in sub_df_10.index]
        sub_df_20.index = [n+"|q20" for n in sub_df_20.index]
        sub_df_30.index = [n+"|q30" for n in sub_df_30.index]
        sub_df_40.index = [n+"|q40" for n in sub_df_40.index]
        sub_df_60.index = [n+"|q60" for n in sub_df_60.index]
        sub_df_70.index = [n+"|q70" for n in sub_df_70.index]
        sub_df_80.index = [n+"|q80" for n in sub_df_80.index]
        sub_df_90.index = [n+"|q90" for n in sub_df_90.index]

        feat_dict[i] = pd.concat([sub_df_median,sub_df_mean,sub_df_min,sub_df_max,sub_df_std,sub_df_10,sub_df_20,sub_df_30,sub_df_40,sub_df_60,sub_df_70,sub_df_80,sub_df_90],axis=0)
        
    feat_df = pd.DataFrame(feat_dict)

    return feat_df

if __name__ == "__main__":
    main()