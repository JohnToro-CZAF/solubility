#!/usr/bin/env python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
import itertools
from esol import ESOLCalculator

# Turn off TensorFlow logging
import deepchem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# -------- Utility Functions -----------------------------------#


def featurize_data(tasks, featurizer, normalize, dataset_file):
    loader = deepchem.data.CSVLoader(tasks=tasks, smiles_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(dataset_file, shard_size=8192)
    move_mean = True
    if normalize:
        transformers = [deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset, move_mean=move_mean)]
    else:
        transformers = []
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    return dataset, featurizer, transformers


def generate_prediction(input_file_name, model, featurizer, transformers):
    df = pd.read_csv(input_file_name)
    mol_list = [Chem.MolFromSmiles(x) for x in df.SMILES]
    val_feats = featurizer.featurize(mol_list)
    if len(transformers) > 0:
        res = model.predict_on_batch(val_feats, transformers)
    else:
        res = model.predict_on_batch(val_feats)
    # kind of a hack
    # seems like some models return a list of lists and others (e.g. RF) return a list
    # check to see if the first element in the returned array is a list, if so, flatten the list
    if type(res[0]) is list:
        df["pred_vals"] = list(itertools.chain.from_iterable(*res))
    else:
        df["pred_vals"] = res
    return df


# ----------- Model Generator Functions --------------------------#


def generate_graph_conv_model():
    batch_size = 128
    model = deepchem.models.GraphConvModel(1, batch_size=batch_size, mode='regression')
    return model


def generate_weave_model():
    batch_size = 64
    model = deepchem.models.WeaveModel(1, batch_size=batch_size, learning_rate=1e-3, use_queue=False, mode='regression')
    return model


def generate_rf_model():
    model_dir = "."
    sklearn_model = RandomForestRegressor(n_estimators=500)
    return deepchem.models.SklearnModel(sklearn_model, model_dir)


# ---------------- Function to Run Models ----------------------#


def run_model(model_func, task_list, featurizer, normalize, training_file_name, validation_file_name, nb_epoch):
    dataset, featurizer, transformers = featurize_data(task_list, featurizer, normalize, training_file_name)
    model = model_func()
    if nb_epoch > 0:
        model.fit(dataset, nb_epoch)
    else:
        model.fit(dataset)
    pred_df = generate_prediction(validation_file_name, model, featurizer, transformers)
    return pred_df


# ------------------ Function to Calculate ESOL ----------------------*

def calc_esol(input_file_name, smiles_col="SMILES"):
    df = pd.read_csv(input_file_name)
    esol_calculator = ESOLCalculator()
    res = []
    for smi in df[smiles_col].values:
        mol = Chem.MolFromSmiles(smi)
        res.append(esol_calculator.calc_esol(mol))
    df["pred_vals"] = res
    return df


# ----------------- main ---------------------------------------------*
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", type=str, default="dls_100_unique.csv")
    parser.add_argument("--output_file_name", type=str, default="solubility_comparison.csv")
    parser.add_argument("--model_types", type=str, nargs="+", default=["all","esol","rf", "weave", "gc"], choices=["all","esol","rf", "weave", "gc"])
    args = parser.parse_args()
    model_types = args.model_types
    
    if "all" in model_types:
        model_types = ["esol", "rf", "weave", "gc"]
    
    
    training_file_name = "delaney.csv"
    validation_file_name = args.input_file_name
    validation_df = pd.read_csv(validation_file_name)
    output_file_name = args.output_file_name
    task_list = ['measured log(solubility:mol/L)']

    print("=====ESOL=====")
    if "esol" in model_types:
        esol_df = calc_esol(validation_file_name)
    else:
        esol_df = None

    print("=====Random Forest=====")
    if "rf" in model_types:
        featurizer = deepchem.feat.CircularFingerprint(size=1024)
        model_func = generate_rf_model
        rf_df = run_model(model_func, task_list, featurizer, False, training_file_name, validation_file_name, nb_epoch=-1)
    else:
        rf_df = None

    print("=====Weave======")
    if "weave" in model_types:
        featurizer = deepchem.feat.WeaveFeaturizer()
        model_func = generate_weave_model
        weave_df = run_model(model_func, task_list, featurizer, True, training_file_name, validation_file_name, nb_epoch=30)
    else:
        weave_df = None

    print("=====Graph Convolution=====")
    if "gc" in model_types:
        featurizer = deepchem.feat.ConvMolFeaturizer()
        model_func = generate_graph_conv_model
        gc_df = run_model(model_func, task_list, featurizer, True, training_file_name, validation_file_name, nb_epoch=20)
    else:
        gc_df = None

    output_df = pd.DataFrame(validation_df[["SMILES", "Chemical name", "LogS exp (mol/L)"]])
    if esol_df is not None:
        output_df["ESOL"] = esol_df["pred_vals"]
    if rf_df is not None:
        output_df["RF"] = rf_df["pred_vals"]
    if weave_df is not None:
        output_df["Weave"] = weave_df["pred_vals"]
    if gc_df is not None:
        output_df["GC"] = gc_df["pred_vals"]
    output_df.to_csv(output_file_name, index=False, float_format="%0.2f")
    print("wrote results to", output_file_name)


if __name__ == "__main__":
    main()
