from ..module.base_metric import BaseMetric
import pandas as pd
import os

class GeneralGenerationBenchMetric(BaseMetric):
    is_general_generation = True  # marker for eval_score_video_general.py to identify this metric type

    def __init__(self, dataset_name=None, max_size=1024):
        super().__init__()
        components = dataset_name.split('_')
        components = [x.capitalize() for x in components]
        self.score_name = dataset_name
        self.dataset_name = ''.join(components)
        self.max_size = max_size
        self.task_type = None
        if ("t2v_general_generation" in dataset_name) or \
           ("i2v_general_generation" in dataset_name) or \
           ("t2i_general_generation" in dataset_name):
            self.task_type = dataset_name.split("_")[0]

        if os.path.exists(dataset_name) and dataset_name.endswith(".json"):
            # Reference-editing JSON (benchmark-style) with per-case `conditions`.
            # Route every case through the range_condition generation path; cases
            # with no conditions fall back to t2v inside the generation loop, so a
            # single mixed-task JSON (t2v/v2v/ti2v/tiv2v) is supported directly.
            self.task_type = "range_condition"
            self.dataset_name = dataset_name.split("/")[-1].split(".")[0]
        elif os.path.exists(dataset_name) and dataset_name.endswith(".csv"):
            df = pd.read_csv(dataset_name)
            if "task_type" in df.columns:
                task_types = {
                    str(x).strip().lower()
                    for x in df["task_type"].dropna().tolist()
                    if str(x).strip() != ""
                }
                if len(task_types) != 1:
                    raise ValueError(
                        f"GeneralGenerationBenchMetric requires one task_type per CSV, "
                        f"got {sorted(task_types)} in {dataset_name}. Split mixed tasks into separate CSVs."
                    )
                task_type = next(iter(task_types))
                if task_type in {"v2v", "ti2v", "tiv2v"}:
                    task_type = "range_condition"
                if task_type not in {"t2v", "i2v", "fl2v", "range_condition"}:
                    raise ValueError(f"Unsupported task_type={task_type} in {dataset_name}")
                self.task_type = task_type
            elif "condition_spec" in df.columns:
                self.task_type = "range_condition"
            elif "ref_image_path" in df.columns:
                self.task_type = "i2v"
            elif "first" in df.columns and "last" in df.columns:
                self.task_type = "fl2v"
            else:
                self.task_type = "t2v"
            self.dataset_name = dataset_name.split("/")[-1].split(".")[0]

    def load_model(self):
        # No external scoring model needed — this metric only drives generation
        pass
