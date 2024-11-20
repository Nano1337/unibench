import os.path
from pathlib import Path

import pandas as pd
import torch

from unibench.common_utils.utils import download_all_results, download_only_aggregate
from oslo_concurrency import lockutils

from .benchmarks_zoo.registry import get_benchmark_info, list_benchmarks, get_benchmark_types

from .common_utils.constants import OUTPUT_DIR, LOCK_DIR

from sklearn.metrics import balanced_accuracy_score

import warnings

import json


class OutputHandler(object):
    def __init__(
        self,
        load_all_csv=False,
        round_values=4,
        output_dir=OUTPUT_DIR,
        download_all_precomputed=False,
        download_aggregate_precomputed=False,
    ):
        self.output_dir = Path(output_dir)
        if download_all_precomputed:
            download_all_results(self.output_dir)
        elif download_aggregate_precomputed:
            download_only_aggregate(self.output_dir)
        self.round_values = round_values
        self.reset_local_csv()
        lockutils.set_defaults(lock_path=LOCK_DIR)
        self.load_aggregate_results()
        if load_all_csv:
            self.load_all_csv()

    def reset_local_csv(self):
        self._local_csv = pd.DataFrame()

    def check_if_computed(self, model_name, benchmark_name, **kwargs):
        self.load_aggregate_results()
        res = self.query(
            df=self._aggregate,
            **{"model_name": model_name, "benchmark_name": benchmark_name}
        )
        if len(res) >= 1:
            return True

        self.load_csv(model_name, benchmark_name)
        return len(self.query(**kwargs))

    def load_all_csvs(self, model_names):
        self._model_csv = pd.DataFrame()
        dfs = []
        for model in model_names:
            model_folder = self.output_dir.joinpath(model)
            for benchmark_file in os.listdir(model_folder):
                file = model_folder.joinpath(benchmark_file)
                if ".f" in file.suffix and file.exists():
                    try:
                        dfs.append(pd.read_feather(file))
                    except:
                        print("Error reading file: ", file)
                else:
                    print("File not found: ", file)

        self._model_csv = pd.concat(dfs).reset_index(drop=True).round(self.round_values)

    def load_all_csv(self, model_name, benchmark_name):
        self._model_csv = pd.DataFrame()
        dfs = []
        for model in model_name:
            model_folder = self.output_dir.joinpath(model)
            for benchmark in benchmark_name:
                file = model_folder.joinpath(benchmark + ".f")
                if file.exists():
                    try:
                        dfs.append(pd.read_feather(file))
                    except:
                        print("Error reading file: ", file)
                else:
                    print("File not found: ", file)

        self._model_csv = pd.concat(dfs).reset_index(drop=True).round(self.round_values)

    def load_csv(self, model_name, benchmark_name):
        file_name = str(
            self.output_dir.joinpath(model_name).joinpath(benchmark_name + ".f")
        )

        # Load the csv if it exists
        if os.path.exists(file_name):
            self._model_csv = pd.read_feather(file_name)
        else:
            self._model_csv = pd.DataFrame()

    def load_model_csvs(self, model_name, use_cols=None):
        model_folder = self.output_dir.joinpath(model_name)

        self._model_csv = pd.DataFrame()
        dfs = []
        for file in os.listdir(model_folder):
            if file.endswith(".f"):
                dfs.append(
                    pd.read_feather(model_folder.joinpath(file), columns=use_cols)
                )

        self._model_csv = pd.concat(dfs).reset_index(drop=True).round(self.round_values)


    def _get_cls_metrics(self, df): 
        # drop columns we don't need for this analysis
        df = df.drop(columns=["model_name", "entropy", "benchmark_name", "split", "predictions_top5", "confidence", "image_name"])

        # calculate correctness at top1 and top5
        acc1 = df["correctness"].mean()
        acc5 = df["correctness_top5"].mean()

        # calculate mean per class recall - convert warning to exception
        target = df["image_class"].values
        preds = df["predictions"].values
        
        # Suppress the warning and calculate balanced accuracy
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
            mean_per_class_recall = balanced_accuracy_score(target, preds)
        
        metrics = {
            "acc1": acc1,
            "acc5": acc5,
            "mean_per_class_recall": mean_per_class_recall,
            "main_metric": acc1,
        }
        return metrics
    
    def _get_order_retrieval_metrics(self, df):
        mean_acc = df["correctness"].mean()
        return {"acc1": mean_acc, "main_metric": mean_acc}   

    def _get_sugarcrepe_metrics(self, df):
        # Calculate mean accuracy per attribute
        metrics = df.groupby("attribute")["correctness"].mean().to_dict()
        return metrics
    
    def _get_other_metrics(self, df, benchmark_name):
        if benchmark_name == "countbench":
            return {"acc1": df["correctness"].mean(), "main_metric": df["correctness"].mean()}
        elif benchmark_name == "vg_relation":
            return {"acc1": df["correctness"].mean(), "main_metric": df["correctness"].mean()}
        elif benchmark_name == "winoground":
            return {"text_acc": df["text_correctness"].mean(), "image_acc": df["image_correctness"].mean(), "main_metric": df["text_correctness"].mean()}
        elif benchmark_name == "vg_attribution":
            return {"acc1": df["correctness"].mean(), "main_metric": df["correctness"].mean()}
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def load_model_csvs_and_calculate(self, model_name, use_cols=None):
        # TODO 2: after we get this one working, instead of saving the .f file, we can directly save/concat the jsonl with lockutils
        model_folder = self.output_dir.joinpath(model_name)

        self._model_csv = pd.DataFrame()
        dfs = []
        vtab_group = list_benchmarks("blogpost_vtab")
        order_retrieval_group = list_benchmarks("order_retrieval") # FIXME: will need to manually align this
        sugarcrepe_group = list_benchmarks("blogpost_sugarcrepe")
        other_group = list_benchmarks("other") # requires special processing
        # wilds_group = list_benchmarks("wilds") # TODO: wilds evals don't exist yet, needs to be implemented
        # may need other groups depending on what metrics there are in the df

        # NOTE, example: /fsx/users/amro/projects/openclip_projects/science/outputs/openclip/BP_CLIP-B-32_DC_raw_pool-256m_cls-optimized_size-47m_compute-128m_seed0/eval_results/epoch_0_step_8784
        data = []
        for file in os.listdir(model_folder):
            if file.endswith(".f"):
                df = pd.read_feather(model_folder.joinpath(file), columns=use_cols)
                benchmark_name = df["benchmark_name"].iloc[0] # note that the names here are the "key" names, not the "dataset" display
                try: 
                    if benchmark_name in vtab_group:
                        metrics = self._get_cls_metrics(df)
                        data.append({
                            "key": f"vtab/{benchmark_name}",
                            "dataset": benchmark_name,
                            "metrics": metrics,
                        })
                    elif benchmark_name in order_retrieval_group:
                        metrics = self._get_order_retrieval_metrics(df)
                        data.append({
                            "key": benchmark_name,
                            "dataset": benchmark_name,
                            "metrics": metrics,
                        })
                    elif benchmark_name in sugarcrepe_group:
                        metrics = self._get_sugarcrepe_metrics(df)
                        for k, v in metrics.items():
                            data.append({
                                "key": f"sugar_crepe/{k}",
                                "dataset": f"sugar_crepe_{k}",
                                "metrics": {"acc": metrics[k], "main_metric": metrics[k]},
                            })
                    elif benchmark_name in other_group:
                        metrics = self._get_other_metrics(df, benchmark_name)
                        data.append({
                            "key": benchmark_name,
                            "dataset": benchmark_name,
                            "metrics": metrics,
                        })
                    else:
                        # TODO 1: test this on all evals once finished
                        metrics = self._get_cls_metrics(df)
                except Exception as e: 
                    print(f"Error processing {benchmark_name}: {e}")
                    exit()

        # Writing to a JSONL file
        with open(f"{model_name}_eval_results.jsonl", 'w') as f:
            for item in data:
                # Write each dictionary as a JSON line
                json_line = json.dumps(item)
                f.write(json_line + '\n')

    def get_csv(self):
        return pd.concat([self._local_csv, self._model_csv])

    def add_values(self, **kwargs):
        for k in kwargs.keys():
            if isinstance(kwargs[k], torch.Tensor):
                kwargs[k] = kwargs[k].cpu().squeeze().tolist()
        self._local_csv = pd.concat([self._local_csv, pd.DataFrame(kwargs)])

    def query(self, df=None, **kwargs):
        if df is None:
            df = self._model_csv
        if len(kwargs) == 0:
            return df

        mask = pd.Series([True] * len(df))

        for k, v in kwargs.items():
            if isinstance(v, list):
                mask &= df[k].isin(v)
            else:
                mask &= (df[k] == v)

        return df[mask]

    def delete_rows(self, model_name, benchmark_name, **kwargs):
        # file_name = str(OUTPUT_DIR.joinpath(model_name + ".f"))
        self.output_dir.joinpath(model_name).mkdir(parents=True, exist_ok=True)
        file_name = str(
            self.output_dir.joinpath(model_name).joinpath(benchmark_name + ".f")
        )

        # Load the csv if it exists
        if os.path.exists(file_name):
            self._model_csv = pd.read_feather(file_name)
        else:
            pass

        self._model_csv.drop(self.query(**kwargs).index, inplace=True)
        self._model_csv = self._model_csv.reset_index(drop=True)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._model_csv.to_feather(file_name)

    def _get_benchmark_mappings(self, axis):
        benchmark_mappings = {}
        for benchmark in list_benchmarks():
            if axis is None:
                benchmark_mappings[benchmark] = get_benchmark_info(benchmark)
            else:
                benchmark_mappings[benchmark] = get_benchmark_info(benchmark)[axis]
        return benchmark_mappings

    @lockutils.synchronized(name="aggregate", external=True, fair=True)
    def load_aggregate_results(self):
        file = self.output_dir.joinpath("aggregate.f")
        if file.exists():
            self._aggregate = pd.read_feather(file)

    @lockutils.synchronized(name="aggregate", external=True, fair=True)
    def save_aggregate_results(self, model_name, benchmark_name):
        file_dir = self.output_dir.joinpath("aggregate.f")
        if file_dir.exists():
            self._aggregate = pd.read_feather(file_dir)

        df = self.query(
            self._model_csv,
            **{"model_name": [model_name], "benchmark_name": [benchmark_name]}
        )

        df = (
            df.groupby(["model_name", "benchmark_name"])["correctness"]
            .mean()
            .reset_index()
        )
        
        df = (
            pd.concat([self._aggregate, df])
            .drop_duplicates(subset=["model_name", "benchmark_name"], keep="last")
            .reset_index(drop=True)
        )

        df.to_feather(file_dir)

    def print_dataframe(self, **kwargs):
        self.load_aggregate_results()
        df = self.query(df=self._aggregate, **kwargs)
        benchmark_mappings = self._get_benchmark_mappings("benchmark_type")
        df["benchmark_type"] = df["benchmark_name"].map(benchmark_mappings)
        df = (
            df.groupby(["model_name", "benchmark_name", "benchmark_type"])[
                "correctness"
            ]
            .mean()
            .reset_index()
        )

        df = (
            df.groupby(["model_name", "benchmark_type"])["correctness"]
            .mean()
            .reset_index()
        )
        return df.pivot(
            index="model_name", columns="benchmark_type", values="correctness"
        )

    def save_csv(self, model_name, benchmark_name):
        self.output_dir.joinpath(model_name).mkdir(parents=True, exist_ok=True)
        file_name = str(
            self.output_dir.joinpath(model_name).joinpath(benchmark_name + ".f")
        )

        # Load the csv if it exists
        if os.path.exists(file_name):
            self._model_csv = pd.read_feather(file_name)
        else:
            self._model_csv = pd.DataFrame()

        # Add the local csv to the model csv
        self._model_csv = (
            pd.concat(
                [self._model_csv, self._local_csv.reset_index(drop=True)],
                axis=0,
                ignore_index=True,
            )
            .round(self.round_values)
            .reset_index(drop=True)
        )

        # Save the model csv
        self._model_csv.to_feather(file_name)
        self.reset_local_csv()