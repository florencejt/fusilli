import importlib
import os
import warnings
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import sys

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch_sparse

from fusionlibrary.datamodules import (
    CustomDataModule,
    KFoldDataModule,
    GraphDataModule,
    KFoldGraphDataModule,
)
from fusionlibrary.eval_functions import (
    compare_methods_boxplot,
    eval_one_rep_kfold,
    eval_replications,
)
from fusion_models.base_pl_model import BaseModel
from mnd_data.mnd_preprocessing import get_mnd_data_ready
from train_functions import train_and_test
from utils.arguments import init_parser
from utils.pl_utils import (
    get_final_val_metrics,
    init_trainer,
    set_logger,
    update_repetition_results,
)


warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*MPS available but not used.*")
warnings.filterwarnings(
    "ignore", message="Checkpoint directory.*exists and is not empty."
)


fusion_model_dict = [
    {"name": "Tabular1Unimodal", "path": "fusion_models.tabular1_unimodal"},
    {"name": "Tabular2Unimodal", "path": "fusion_models.tabular2_unimodal"},
    # {"name": "ImgUnimodal", "path": "fusion_models.img_unimodal"},
    {
        "name": "ConcatTabularFeatureMaps",
        "path": "fusion_models.concat_tabular_feature_maps",
    },
    # {
    #     "name": "ConcatImageMapsTabularData",
    #     "path": "fusion_models.concat_img_maps_tabular_data",
    # },
    {"name": "ConcatTabularData", "path": "fusion_models.concat_tabular_data"},
    # {
    #     "name": "ConcatImageMapsTabularMaps",
    #     "path": "fusion_models.concat_img_maps_tabular_maps",
    # },
    {
        "name": "TabularChannelWiseMultiAttention",
        "path": "fusion_models.tabular_channelwise_att",
    },
    # {
    #     "name": "ImageChannelWiseMultiAttention",
    #     "path": "fusion_models.img_tab_channelwise_att",
    # },
    # {"name": "CrossmodalMultiheadAttention", "path": "fusion_models.crossmodal_att"},
    {
        "name": "TabularCrossmodalMultiheadAttention",
        "path": "fusion_models.tab_crossmodal_att",
    },
    {"name": "TabularDecision", "path": "fusion_models.tabular_decision"},
    # {"name": "ImageDecision", "path": "fusion_models.img_tab_decision"},
    {"name": "MCVAE_tab", "path": "fusion_models.mcvae_tab"},
    # {
    #     "name": "ConcatImgLatentTabDoubleTrain",
    #     "path": "fusion_models.concat_img_latent_tab_doubletrain",
    # },
    # {
    #     "name": "ConcatImgLatentTabDoubleLoss",
    #     "path": "fusion_models.concat_img_latent_tab_doubleloss",
    # },
    {"name": "EdgeCorrGNN", "path": "fusion_models.edge_corr_gnn"},
]

fusion_model_names = [fusion_model["name"] for fusion_model in fusion_model_dict]

fusion_models = []
for model in fusion_model_dict:
    module_name = model["name"]
    module_path = model["path"]

    module = importlib.import_module(module_path)
    module_class = getattr(module, module_name)

    fusion_models.append(module_class)

print("Methods tested: ", fusion_model_names)

# Parse the arguments
parser = init_parser()
args = parser.parse_args()

params = {
    "cluster": args.cluster,
    "pred_type": args.pred_type,
    "multiclass_dims": args.num_classes,
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "log": args.log,
    "kfold_flag": args.kfold,
    "num_replications": args.num_reps,
    "dataloadermodule": CustomDataModule,
    "num_k": args.num_folds,
    "subspace_latdims": 5,  # TODO make this an argument
    "test_size": 0.3,
    "data_source": "ADNI",  # TODO this is just for now, change it later
}

if params["data_source"] == "MND":
    # geting MND data ready and getting the data file paths
    params = get_mnd_data_ready(params)
    data_sources = [
        params["tabular1_source"],
        params["tabular2_source"],
        params["img_source"],
    ]
elif params["data_source"] == "ADNI":
    if not params["cluster"]:
        params["file_root"] = (
            "/Users/florencetownend/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Projects/"
            "multimodal_methods_lightning/"
        )
        params["tabular1_source"] = params["file_root"] + "data/tabular1_adni.csv"
        params["tabular2_source"] = params["file_root"] + "data/tabular2_adni.csv"

        params["img_source"] = params["file_root"] + "mnd_data/Milan_6monthsdelay.pt"

        data_sources = [
            params["tabular1_source"],
            params["tabular2_source"],
            params["img_source"],
        ]

    else:
        raise NotImplementedError("ADNI data not yet available on cluster")


if params["log"]:  # finish any wandb process that is happening
    wandb.finish()
if params["kfold_flag"]:  # if we're doing kfold, we need to use the KFoldDataModule
    params["dataloadermodule"] = KFoldDataModule
if (
    params["cluster"] is False and params["log"] is False
):  # if we're not logging and not on the cluster, we need to make a local figures folder
    params["local_fig_path"] = f"local_figures/{params['timestamp']}"
    os.mkdir(params["local_fig_path"])

repetition_performances = dict().fromkeys(
    fusion_model_names
)  # initialise dict to store the performances of each method over all repetitions
# structure: method: {metric: [] for metric in metric_names} for method in methods}


for rep_n in range(params["num_replications"]):
    k_fold_performances = {}  # initialise dict to store the performances of each
    # method over all kfold folds

    final_rep_flag = rep_n == params["num_replications"] - 1

    for i, fusion_model in enumerate(fusion_models):
        only_one_model_flag = len(fusion_models) == 1
        final_model_flag = i == len(fusion_models) - 1

        method_name = fusion_model_names[i]

        print("#" * 50)
        print("Method: ", method_name)
        print("#" * 50)

        init_model = BaseModel(
            fusion_model(
                params["pred_type"], data_dims=[10, 10, [100, 100, 100]], params=params
            )
        )

        modality_type = init_model.modality_type
        fusion_type = init_model.fusion_type
        metric_1_name = init_model.metrics[init_model.pred_type][0]["name"]
        metric_2_name = init_model.metrics[init_model.pred_type][1]["name"]
        metric_name_list = [metric_1_name, metric_2_name]

        if rep_n == 0:  # create the empty lists for the metrics on the first repetition
            repetition_performances[method_name] = {
                metric_1_name: [],
                metric_2_name: [],
            }

        if init_model.fusion_type == "graph":
            if params["kfold_flag"]:
                dmg = KFoldGraphDataModule(
                    params,
                    modality_type,
                    sources=data_sources,
                    graph_creation_method=init_model.graph_maker,
                )
            else:
                dmg = GraphDataModule(
                    params,
                    modality_type,
                    sources=data_sources,
                    graph_creation_method=init_model.graph_maker,
                )

            dmg.prepare_data()
            dmg.setup()
            dm = dmg.get_lightning_module()

            if params["kfold_flag"]:
                for dm_instance in dm:
                    dm_instance.data_dims = dmg.data_dims
            else:
                dm.data_dims = dmg.data_dims

        else:
            # another other than graph fusion
            dm = params["dataloadermodule"](
                params,
                modality_type,
                sources=data_sources,
                subspace_method=init_model.subspace_method,
            )
            dm.prepare_data()
            dm.setup()

        if params["kfold_flag"]:
            all_kf_val_reals = []
            all_kf_val_preds = []  # list of all the validation preds for each fold
            kfold_metrics = []
            kfold_metrics_2 = []
            # list of the secondary validation metrics for each fold

            for k in range(params["num_k"]):
                print("Fold: ", k)

                (
                    model,
                    trainer,
                    fold_metric_1,
                    fold_metric_2,
                    val_reals,
                    val_preds,
                ) = train_and_test(
                    dm,
                    params,
                    rep_n,
                    k,
                    fusion_model,
                    init_model,
                    metric_name_list,
                    method_name,
                )

                all_kf_val_reals.append(val_reals)
                all_kf_val_preds.append(val_preds)
                kfold_metrics.append(fold_metric_1)
                kfold_metrics_2.append(fold_metric_2)

                if k == params["num_k"] - 1:  # if we're on the last fold
                    # plot the altogether kfold results - move log finish to after this
                    # TODO make this applicable to binary and multiclass applications as well

                    overall_kfold_metrics = model.plot_kfold_eval_figs(
                        all_kf_val_preds,
                        all_kf_val_reals,
                        kfold_metrics,
                        path_suffix=f"_{method_name}_rep{rep_n}",
                    )

                    repetition_performances = update_repetition_results(
                        repetition_performances,
                        method_name,
                        metric_name_list,
                        overall_kfold_metrics,
                    )

                    k_fold_performances[method_name] = {
                        metric_1_name: kfold_metrics,
                        metric_2_name: kfold_metrics_2,
                    }

                    if final_model_flag:
                        pass
                    else:
                        if params["log"]:
                            wandb.finish()

                else:
                    if params["log"]:
                        wandb.finish()

        elif params["kfold_flag"] is False:
            model, trainer, metric_1, metric_2, val_reals, val_preds = train_and_test(
                dm,
                params,
                rep_n,
                None,
                fusion_model,
                init_model,
                metric_name_list,
                method_name,
            )

            repetition_performances = update_repetition_results(
                repetition_performances,
                method_name,
                metric_name_list,
                [metric_1, metric_2],
            )

            if (only_one_model_flag and final_rep_flag) or (
                final_rep_flag and final_model_flag
            ):
                pass
            elif params["log"]:
                wandb.finish()

    eval_one_rep_kfold(k_fold_performances, rep_n, params)


eval_replications(repetition_performances, params)
