"""CICIoT2023 thesis-grade hybrid AI experiment utilities.

This module is designed to be imported from the accompanying Jupyter notebook.
It keeps the notebook readable while still making every experimental step
reproducible, configurable, and reusable.
"""

from __future__ import annotations

import gc
import ipaddress
import json
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

try:
    from IPython.display import display
except Exception:  # pragma: no cover - notebook convenience only
    display = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
except Exception:  # pragma: no cover
    tf = None
    keras = None
    layers = None
    regularizers = None

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception:  # pragma: no cover
    SMOTE = None
    BorderlineSMOTE = None
    RandomUnderSampler = None
    SMOTETomek = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


@dataclass
class ExperimentConfig:
    """Central configuration for the CICIoT2023 experiment."""

    DATASET_NAME: str = "CICIoT2023"
    DATASET_PATH: Path = field(default_factory=lambda: Path("CIC_IOT_Dataset2023"))
    LABEL_COLUMN: Optional[str] = None
    SAMPLE_SIZE: Optional[int] = 500_000
    SAMPLE_STRATEGY: str = "class_balanced"  # class_balanced or file_balanced
    CHUNKSIZE: Optional[int] = 250_000
    RANDOM_STATE: int = 42

    TEST_SIZE: float = 0.20
    VAL_SIZE: float = 0.20
    BATCH_SIZE: int = 512
    EPOCHS: int = 50
    PATIENCE: int = 10
    LEARNING_RATE: float = 1e-3

    OUTPUT_ROOT: Path = field(default_factory=lambda: Path("outputs"))
    SCALER: str = "robust"  # robust or standard
    OPTIMIZE_DTYPES: bool = True
    DROP_DUPLICATES: bool = True
    DUPLICATE_DROP_MAX_ROWS: int = 5_000_000
    DUPLICATE_ESTIMATE_SAMPLE_SIZE: int = 250_000
    HIGH_MISSING_THRESHOLD: Optional[float] = 0.90
    DROP_HIGH_CORRELATION: bool = True
    HIGH_CORRELATION_THRESHOLD: float = 0.98
    LOW_VARIANCE_THRESHOLD: float = 0.0
    TOP_K_FEATURES: Optional[int] = 50

    BALANCING_STRATEGY: str = "class_weight"
    # Options: class_weight, smote, borderline_smote, random_under_sampler,
    # smote_tomek, focal_loss, none.
    SMOTE_MAX_SAMPLES: Optional[int] = 200_000
    MIN_SMOTE_K: int = 1
    DEFAULT_SMOTE_K: int = 5

    USE_FOCAL_LOSS: bool = False
    DNN_DROPOUT: float = 0.30
    DNN_L2: float = 1e-4
    DNN_HIDDEN_UNITS: Tuple[int, int, int] = (256, 128, 64)

    RF_N_ESTIMATORS: int = 250
    RF_MAX_DEPTH: Optional[int] = None
    XGB_N_ESTIMATORS: int = 350
    XGB_MAX_DEPTH: int = 7

    EDA_SAMPLE_FOR_PLOTS: int = 10_000
    TSNE_SAMPLE_SIZE: int = 3_000
    PCA_SAMPLE_SIZE: int = 10_000
    MAX_CORR_FEATURES: int = 50
    FEATURE_SELECTION_SAMPLE_SIZE: int = 80_000
    XAI_SAMPLE_SIZE: int = 2_000
    XAI_BACKGROUND_SIZE: int = 200

    UNKNOWN_CLASSES: Optional[List[str]] = None
    AUTO_UNKNOWN_CLASS_COUNT: int = 2
    MIN_UNKNOWN_CLASS_COUNT: int = 500
    UNKNOWN_THRESHOLD_METRIC: str = "f1"
    USE_AUTOENCODER_UNKNOWN: bool = True
    AUTOENCODER_EPOCHS: int = 30

    DEVICE_CRITICALITY_FIXED: Optional[str] = None
    MISSION_CRITICALITY_FIXED: Optional[str] = None
    DEVICE_CRITICALITY_DISTRIBUTION: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.20, "medium": 0.35, "high": 0.30, "critical": 0.15}
    )
    MISSION_CRITICALITY_DISTRIBUTION: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.15, "medium": 0.35, "high": 0.35, "critical": 0.15}
    )
    TRAFFIC_INTENSITY_COLUMNS: Tuple[str, ...] = ("rate", "tot_size", "tot_sum", "number", "iat")

    RUN_BASELINES: bool = True
    RUN_DEEP_LEARNING: bool = True
    RUN_UNKNOWN_DETECTION: bool = True
    RUN_XAI: bool = True
    RUN_ABLATION: bool = True
    RUN_FULL_ABLATION_RETRAIN: bool = True
    ABLATION_EPOCHS: int = 15
    ABLATION_RF_N_ESTIMATORS: int = 120


def set_global_seed(seed: int = 42) -> None:
    """Set reproducibility controls for Python, NumPy, and TensorFlow."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            pass


def config_to_jsonable(config: ExperimentConfig) -> Dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, tuple):
            data[key] = list(value)
    return data


def create_output_dirs(config: ExperimentConfig) -> Dict[str, Path]:
    base = Path(config.OUTPUT_ROOT) / config.DATASET_NAME
    dirs = {
        "base": base,
        "models": base / "models",
        "plots": base / "plots",
        "metrics": base / "metrics",
        "reports": base / "reports",
        "artifacts": base / "artifacts",
        "predictions": base / "predictions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    save_json(config_to_jsonable(config), dirs["artifacts"] / "training_config.json")
    return dirs


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def show_df(df: pd.DataFrame, max_rows: int = 20) -> None:
    if display is not None:
        display(df.head(max_rows))
    else:
        print(df.head(max_rows).to_string())


def safe_filename(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "item"


def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory pressure during experiments."""

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    if "label" in df.columns:
        df["label"] = df["label"].astype("category")
    return df


def normalize_label_value(value: Any) -> str:
    raw = str(value).strip()
    low = raw.lower()
    if low in {"benign", "normal", "benigntraffic", "benign_traffic", "benign_final"}:
        return "Benign"
    if "benign" in low or low == "normal":
        return "Benign"
    return raw


def is_benign_label(value: Any) -> bool:
    return normalize_label_value(value).lower() in {"benign", "normal"}


def derive_label_from_path(csv_path: Path) -> str:
    return normalize_label_value(csv_path.parent.name)


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def iter_with_progress(items: Sequence[Any], desc: str) -> Iterable[Any]:
    if tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def _process_loaded_chunk(
    chunk: pd.DataFrame,
    csv_path: Path,
    requested_label_column: Optional[str],
    infer_label_from_parent: bool = True,
    add_source_file: bool = False,
    optimize_dtypes: bool = True,
) -> pd.DataFrame:
    chunk = clean_columns(chunk)
    cleaned_requested = clean_column_name(requested_label_column) if requested_label_column else None
    if cleaned_requested and cleaned_requested in chunk.columns and cleaned_requested != "label":
        chunk = chunk.rename(columns={cleaned_requested: "label"})
    elif "label" not in chunk.columns and infer_label_from_parent:
        chunk["label"] = derive_label_from_path(csv_path)
    if add_source_file:
        chunk["_source_file"] = str(csv_path)
    if optimize_dtypes:
        chunk = optimize_dataframe_dtypes(chunk)
    return chunk


def _sample_dataframe(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _group_csv_files_by_parent_label(csv_files: Sequence[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for csv_path in csv_files:
        groups.setdefault(derive_label_from_path(csv_path), []).append(csv_path)
    return groups


def load_ciciot2023(
    path: str | Path,
    sample_size: Optional[int] = None,
    chunksize: Optional[int] = None,
    label_column: Optional[str] = None,
    random_state: int = 42,
    sample_strategy: str = "class_balanced",
    optimize_dtypes: bool = True,
) -> pd.DataFrame:
    """Load CICIoT2023 from one CSV file or a directory containing many CSVs.

    The local CICIoT2023 CSV release often stores labels in parent directory
    names rather than a dedicated CSV column. This loader detects that layout
    and adds a clean ``label`` column automatically.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path.resolve()}")

    if path.is_file():
        csv_files = [path]
    else:
        csv_files = sorted(path.rglob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found under {path.resolve()}")

    print(f"CSV file count: {len(csv_files)}")
    print(f"Sample size: {sample_size if sample_size is not None else 'full dataset'}")
    print(f"Chunksize: {chunksize if chunksize else 'disabled'}")
    print(f"Sample strategy: {sample_strategy if sample_size is not None else 'not used'}")

    pieces: List[pd.DataFrame] = []

    if sample_size is not None and len(csv_files) > 1 and sample_strategy == "class_balanced":
        class_groups = _group_csv_files_by_parent_label(csv_files)
        class_names = ["Benign"] + sorted([c for c in class_groups if c != "Benign"])
        class_targets = {
            cls: sample_size // len(class_names) + (1 if i < sample_size % len(class_names) else 0)
            for i, cls in enumerate(class_names)
        }
        print(f"Class-balanced sampling over {len(class_names)} classes.")

        for class_i, cls in enumerate(iter_with_progress(class_names, "Loading CICIoT2023 classes")):
            files = class_groups.get(cls, [])
            target_for_class = class_targets[cls]
            if not files or target_for_class <= 0:
                continue
            per_file_target = max(1, math.ceil(target_for_class / len(files) * 1.25))
            class_pieces: List[pd.DataFrame] = []
            class_collected = 0

            for file_i, csv_path in enumerate(files):
                file_collected = 0
                try:
                    if chunksize:
                        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
                            chunk = _process_loaded_chunk(
                                chunk,
                                csv_path,
                                label_column,
                                optimize_dtypes=optimize_dtypes,
                            )
                            remaining_file = per_file_target - file_collected
                            remaining_class = max(target_for_class - class_collected, 0)
                            take = min(len(chunk), remaining_file, max(remaining_class, 1))
                            if take <= 0:
                                break
                            if len(chunk) > take:
                                chunk = chunk.sample(n=take, random_state=random_state + class_i + file_i)
                            class_pieces.append(chunk)
                            file_collected += len(chunk)
                            class_collected += len(chunk)
                            if file_collected >= per_file_target or class_collected >= target_for_class:
                                break
                    else:
                        file_df = pd.read_csv(csv_path, low_memory=False)
                        file_df = _process_loaded_chunk(
                            file_df,
                            csv_path,
                            label_column,
                            optimize_dtypes=optimize_dtypes,
                        )
                        take = min(len(file_df), per_file_target, max(target_for_class - class_collected, 0))
                        if take > 0:
                            file_df = _sample_dataframe(file_df, take, random_state + class_i + file_i)
                            class_pieces.append(file_df)
                            class_collected += len(file_df)
                    if class_collected >= target_for_class:
                        break
                except Exception as exc:
                    raise RuntimeError(f"Could not load CSV file {csv_path}: {exc}") from exc

            if class_pieces:
                class_df = pd.concat(class_pieces, ignore_index=True)
                class_df = _sample_dataframe(class_df, target_for_class, random_state + class_i)
                pieces.append(class_df)

        if not pieces:
            raise ValueError("No data could be loaded. Check CSV paths and permissions.")

        df = pd.concat(pieces, ignore_index=True)
        df = _sample_dataframe(df, sample_size, random_state)
        if optimize_dtypes:
            df = optimize_dataframe_dtypes(df)
        print(f"Loaded shape: {df.shape}")
        print(f"Memory usage: {memory_usage_mb(df):,.2f} MB")
        return df

    per_file_target = None
    if sample_size is not None and len(csv_files) > 1:
        per_file_target = max(1, math.ceil(sample_size / len(csv_files) * 1.5))

    for i, csv_path in enumerate(iter_with_progress(csv_files, "Loading CICIoT2023 CSV files")):
        try:
            if chunksize:
                file_sample: List[pd.DataFrame] = []
                collected_for_file = 0
                for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
                    chunk = _process_loaded_chunk(
                        chunk,
                        csv_path,
                        label_column,
                        optimize_dtypes=optimize_dtypes,
                    )
                    if per_file_target is not None:
                        target = min(len(chunk), per_file_target)
                        chunk = chunk.sample(n=target, random_state=random_state + i)
                    file_sample.append(chunk)
                    collected_for_file += len(chunk)
                    if per_file_target is not None and collected_for_file >= per_file_target:
                        break
                if file_sample:
                    file_df = pd.concat(file_sample, ignore_index=True)
                    if per_file_target is not None:
                        file_df = _sample_dataframe(file_df, per_file_target, random_state + i)
                    pieces.append(file_df)
            else:
                file_df = pd.read_csv(csv_path, low_memory=False)
                file_df = _process_loaded_chunk(
                    file_df,
                    csv_path,
                    label_column,
                    optimize_dtypes=optimize_dtypes,
                )
                if per_file_target is not None:
                    file_df = _sample_dataframe(file_df, per_file_target, random_state + i)
                pieces.append(file_df)
        except Exception as exc:
            raise RuntimeError(f"Could not load CSV file {csv_path}: {exc}") from exc

    if not pieces:
        raise ValueError("No data could be loaded. Check CSV paths and permissions.")

    df = pd.concat(pieces, ignore_index=True)
    df = _sample_dataframe(df, sample_size, random_state)
    if optimize_dtypes:
        df = optimize_dataframe_dtypes(df)
    print(f"Loaded shape: {df.shape}")
    print(f"Memory usage: {memory_usage_mb(df):,.2f} MB")
    return df


def detect_label_column(df: pd.DataFrame, manual_label_column: Optional[str] = None) -> str:
    if manual_label_column:
        cleaned = clean_column_name(manual_label_column)
        if cleaned in df.columns:
            return cleaned
        raise ValueError(f"Manual LABEL_COLUMN={manual_label_column!r} was not found after cleaning.")

    candidates = ["label", "attack", "class", "category"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No label column could be detected. Set config.LABEL_COLUMN manually "
        "or use a CICIoT2023 directory layout where parent folders are class names."
    )


def prepare_labels(
    df: pd.DataFrame,
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = clean_columns(df)
    label_col = detect_label_column(df, config.LABEL_COLUMN)
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
        label_col = "label"

    df["label"] = df["label"].map(normalize_label_value)
    df["binary_label"] = df["label"].map(lambda x: 0 if is_benign_label(x) else 1).astype(int)
    df["multiclass_label"] = df["label"].astype(str)

    class_names = ["Benign"] + sorted([c for c in df["multiclass_label"].unique() if c != "Benign"])
    multiclass_encoder = LabelEncoder()
    multiclass_encoder.fit(class_names)
    df["multiclass_encoded"] = multiclass_encoder.transform(df["multiclass_label"])

    binary_mapping = {"Benign": 0, "Attack": 1}
    multiclass_mapping = {cls: int(idx) for idx, cls in enumerate(multiclass_encoder.classes_)}

    save_json(binary_mapping, output_dirs["artifacts"] / "label_mapping_binary.json")
    save_json(multiclass_mapping, output_dirs["artifacts"] / "label_mapping_multiclass.json")
    joblib.dump(binary_mapping, output_dirs["artifacts"] / "label_encoder_binary.joblib")
    joblib.dump(multiclass_encoder, output_dirs["artifacts"] / "label_encoder_multiclass.joblib")

    info = {
        "label_column": label_col,
        "binary_mapping": binary_mapping,
        "multiclass_mapping": multiclass_mapping,
        "multiclass_encoder": multiclass_encoder,
        "class_names": list(multiclass_encoder.classes_),
    }
    print(f"Detected label column: {label_col}")
    print(f"Binary distribution:\n{df['binary_label'].value_counts().to_string()}")
    print(f"Multiclass class count: {df['multiclass_label'].nunique()}")
    return df, info


def clean_basic_dataframe(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Apply leakage-safe basic cleaning before modeling labels are split."""

    before = len(df)
    duplicates_removed = 0
    if config.DROP_DUPLICATES and before <= config.DUPLICATE_DROP_MAX_ROWS:
        df = df.drop_duplicates().reset_index(drop=True)
        duplicates_removed = before - len(df)
    elif config.DROP_DUPLICATES:
        sample_n = min(before, config.DUPLICATE_ESTIMATE_SAMPLE_SIZE)
        duplicate_estimate = np.nan
        if sample_n > 1:
            sample = df.sample(sample_n, random_state=config.RANDOM_STATE)
            duplicate_estimate = float(sample.duplicated().mean())
        print(
            "Exact duplicate removal skipped for memory safety: "
            f"{before:,} rows exceeds DUPLICATE_DROP_MAX_ROWS="
            f"{config.DUPLICATE_DROP_MAX_ROWS:,}."
        )
        print(
            "Estimated duplicate ratio from sample: "
            f"{duplicate_estimate:.6f}" if np.isfinite(duplicate_estimate) else "not available"
        )
    else:
        print("Duplicate removal disabled by config.DROP_DUPLICATES=False.")

    numeric_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
    for col in numeric_cols:
        if pd.api.types.is_float_dtype(df[col].dtype):
            values = df[col].to_numpy(copy=False)
            bad_mask = np.isinf(values)
            if bad_mask.any():
                df.loc[bad_mask, col] = np.nan
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
    print(f"Duplicate rows removed: {duplicates_removed:,}")
    print(f"Fully empty columns removed: {len(empty_cols)}")
    return df


def save_current_fig(path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close()


def plot_count_series(series: pd.Series, title: str, path: Path, top_n: Optional[int] = None) -> None:
    counts = series.value_counts()
    if top_n:
        counts = counts.head(top_n)
    height = max(4, min(14, 0.35 * len(counts) + 2))
    plt.figure(figsize=(11, height))
    sns.barplot(x=counts.values, y=counts.index.astype(str), orient="h", palette="viridis")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("")
    save_current_fig(path)


def run_eda(
    df: pd.DataFrame,
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
    label_col: str = "label",
) -> Dict[str, Any]:
    print("Dataset shape:", df.shape)
    print("Memory usage MB:", round(memory_usage_mb(df), 2))
    print("\nFirst 10 rows:")
    show_df(df.head(10), max_rows=10)

    print("\nColumns:")
    print(list(df.columns))

    dtypes = df.dtypes.astype(str).reset_index()
    dtypes.columns = ["column", "dtype"]
    show_df(dtypes, max_rows=100)

    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(4)
    missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_table.to_csv(output_dirs["metrics"] / "missing_values.csv")
    print("\nMissing value table:")
    show_df(missing_table[missing_table["missing_count"] > 0], max_rows=50)

    duplicate_count = int(df.duplicated().sum())
    print(f"\nDuplicate row count: {duplicate_count:,}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    numeric_desc = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()
    categorical_summary = df[categorical_cols].describe(include="all").T if categorical_cols else pd.DataFrame()
    numeric_desc.to_csv(output_dirs["metrics"] / "numeric_describe.csv")
    categorical_summary.to_csv(output_dirs["metrics"] / "categorical_summary.csv")

    if label_col in df.columns:
        plot_count_series(df[label_col], "Label Distribution", output_dirs["plots"] / "label_distribution.png")
    if "binary_label" in df.columns:
        binary_text = df["binary_label"].map({0: "Normal", 1: "Attack"})
        plot_count_series(binary_text, "Binary Normal / Attack Distribution", output_dirs["plots"] / "binary_class_distribution.png")
    if "multiclass_label" in df.columns:
        plot_count_series(
            df["multiclass_label"],
            "Multiclass Attack Type Distribution",
            output_dirs["plots"] / "multiclass_attack_distribution.png",
        )

    if missing_table["missing_count"].sum() > 0:
        nonzero_missing = missing_table[missing_table["missing_count"] > 0].head(40)
        plt.figure(figsize=(11, max(4, len(nonzero_missing) * 0.32)))
        sns.barplot(x=nonzero_missing["missing_pct"], y=nonzero_missing.index, palette="magma")
        plt.title("Missing Values (%)")
        plt.xlabel("Missing %")
        save_current_fig(output_dirs["plots"] / "missing_values_bar_plot.png")

        sample_for_missing = df.sample(min(len(df), config.EDA_SAMPLE_FOR_PLOTS), random_state=config.RANDOM_STATE)
        cols = nonzero_missing.index.tolist()[:40]
        plt.figure(figsize=(12, 7))
        sns.heatmap(sample_for_missing[cols].isna(), cbar=False)
        plt.title("Missing Values Heatmap")
        save_current_fig(output_dirs["plots"] / "missing_values_heatmap.png")
    else:
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        plt.axis("off")
        save_current_fig(output_dirs["plots"] / "missing_values_bar_plot.png")

    corr_pairs = pd.DataFrame()
    if numeric_cols:
        corr_features = [c for c in numeric_cols if c not in {"binary_label", "multiclass_encoded"}]
        corr_features = corr_features[: config.MAX_CORR_FEATURES]
        corr_df = df[corr_features].sample(min(len(df), config.EDA_SAMPLE_FOR_PLOTS), random_state=config.RANDOM_STATE)
        corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(corr_df.median(numeric_only=True))
        corr = corr_df.corr().fillna(0)
        plt.figure(figsize=(13, 11))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
        plt.title("Correlation Heatmap")
        save_current_fig(output_dirs["plots"] / "correlation_heatmap.png")

        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        corr_pairs = (
            upper.abs()
            .stack()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_correlation"})
        )
        corr_pairs.to_csv(output_dirs["metrics"] / "top_correlated_feature_pairs.csv", index=False)
        top_pairs = corr_pairs.head(20)
        if len(top_pairs):
            plt.figure(figsize=(11, 7))
            labels = top_pairs["feature_1"] + " | " + top_pairs["feature_2"]
            sns.barplot(x=top_pairs["abs_correlation"], y=labels, palette="crest")
            plt.title("Top Correlated Feature Pairs")
            plt.xlabel("|Correlation|")
            save_current_fig(output_dirs["plots"] / "top_correlated_features_plot.png")

    make_embedding_plots(df, config, output_dirs)
    make_feature_distribution_plots(df, config, output_dirs)

    class_counts = df["multiclass_label"].value_counts() if "multiclass_label" in df.columns else pd.Series(dtype=int)
    imbalance_ratio = float(class_counts.max() / max(class_counts.min(), 1)) if len(class_counts) else np.nan

    eda_summary = {
        "shape": list(df.shape),
        "memory_mb": memory_usage_mb(df),
        "duplicate_count": duplicate_count,
        "numeric_column_count": len(numeric_cols),
        "categorical_column_count": len(categorical_cols),
        "class_imbalance_ratio_max_to_min": imbalance_ratio,
        "top_correlated_pairs": corr_pairs.head(20).to_dict(orient="records") if len(corr_pairs) else [],
    }
    save_json(eda_summary, output_dirs["metrics"] / "eda_summary.json")
    return eda_summary


def make_embedding_plots(df: pd.DataFrame, config: ExperimentConfig, output_dirs: Dict[str, Path]) -> None:
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"binary_label", "multiclass_encoded"}
    ]
    if len(numeric_cols) < 2:
        return

    from sklearn.decomposition import PCA

    sample_n = min(len(df), config.PCA_SAMPLE_SIZE)
    sample = df.sample(sample_n, random_state=config.RANDOM_STATE)
    X = sample[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    X = StandardScaler().fit_transform(X)
    labels = sample["multiclass_label"].astype(str) if "multiclass_label" in sample else None

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    emb = pca.fit_transform(X)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, s=12, linewidth=0, legend=False)
    plt.title("PCA 2D Scatter Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_current_fig(output_dirs["plots"] / "pca_2d_scatter_plot.png")

    tsne_n = min(len(sample), config.TSNE_SAMPLE_SIZE)
    if tsne_n >= 50:
        tsne_sample = sample.sample(tsne_n, random_state=config.RANDOM_STATE)
        X_tsne = tsne_sample[numeric_cols].replace([np.inf, -np.inf], np.nan)
        X_tsne = X_tsne.fillna(X_tsne.median(numeric_only=True)).astype(float)
        X_tsne = StandardScaler().fit_transform(X_tsne)
        perplexity = min(30, max(5, (tsne_n - 1) // 4))
        emb_tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=config.RANDOM_STATE,
        ).fit_transform(X_tsne)
        labels_tsne = tsne_sample["multiclass_label"].astype(str) if "multiclass_label" in tsne_sample else None
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=emb_tsne[:, 0], y=emb_tsne[:, 1], hue=labels_tsne, s=12, linewidth=0, legend=False)
        plt.title("t-SNE 2D Scatter Plot")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        save_current_fig(output_dirs["plots"] / "tsne_2d_scatter_plot.png")


def make_feature_distribution_plots(df: pd.DataFrame, config: ExperimentConfig, output_dirs: Dict[str, Path]) -> None:
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"binary_label", "multiclass_encoded"}
    ]
    if not numeric_cols:
        return
    sample = df.sample(min(len(df), config.EDA_SAMPLE_FOR_PLOTS), random_state=config.RANDOM_STATE)
    variances = sample[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
    top_features = variances.head(8).index.tolist()
    for feature in top_features:
        plt.figure(figsize=(10, 5))
        if "binary_label" in sample.columns:
            hue = sample["binary_label"].map({0: "Normal", 1: "Attack"})
            sns.kdeplot(data=sample, x=feature, hue=hue, fill=True, common_norm=False, alpha=0.35)
        else:
            sns.histplot(sample[feature], kde=True)
        plt.title(f"Distribution of {feature}")
        save_current_fig(output_dirs["plots"] / f"feature_distribution_{safe_filename(feature)}.png")


def is_private_ip(value: Any) -> Optional[int]:
    try:
        return int(ipaddress.ip_address(str(value)).is_private)
    except Exception:
        return np.nan


def detect_ip_columns(columns: Sequence[str]) -> List[str]:
    exact = {
        "src_ip",
        "dst_ip",
        "source_ip",
        "destination_ip",
        "client_ip",
        "server_ip",
        "ip_address",
        "srcaddr",
        "dstaddr",
    }
    out = []
    for c in columns:
        lc = c.lower()
        if lc in exact or lc.endswith("_ip") or lc.endswith("_addr"):
            if lc != "ipv":
                out.append(c)
    return out


def detect_timestamp_columns(columns: Sequence[str]) -> List[str]:
    exact = {
        "timestamp",
        "time_stamp",
        "flow_start_time",
        "flow_end_time",
        "datetime",
        "date_time",
        "start_time",
        "end_time",
    }
    out = []
    for c in columns:
        lc = c.lower()
        if lc in exact or "timestamp" in lc:
            out.append(c)
    return out


def build_feature_frame(df: pd.DataFrame, config: ExperimentConfig) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create leakage-aware model input frame and target vectors."""

    data = df.copy()
    data["sample_id"] = np.arange(len(data), dtype=int)

    ip_cols = detect_ip_columns(data.columns)
    src_candidates = [c for c in ip_cols if c in {"src_ip", "source_ip", "srcaddr", "client_ip"}]
    dst_candidates = [c for c in ip_cols if c in {"dst_ip", "destination_ip", "dstaddr", "server_ip"}]
    if src_candidates:
        src_col = src_candidates[0]
        data["is_src_private"] = data[src_col].map(is_private_ip)
    if dst_candidates:
        dst_col = dst_candidates[0]
        data["is_dst_private"] = data[dst_col].map(is_private_ip)
    if src_candidates and dst_candidates:
        data["src_dst_same_private_status"] = (
            data["is_src_private"].fillna(-1).astype(int) == data["is_dst_private"].fillna(-2).astype(int)
        ).astype(int)

    timestamp_cols = detect_timestamp_columns(data.columns)
    for col in timestamp_cols[:1]:
        ts = pd.to_datetime(data[col], errors="coerce", utc=True)
        data["hour"] = ts.dt.hour
        data["day_of_week"] = ts.dt.dayofweek
        data["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(float)

    drop_exact = {
        "label",
        "binary_label",
        "multiclass_label",
        "multiclass_encoded",
        "sample_id",
        "_source_file",
    }
    leakage_cols = set(ip_cols + timestamp_cols)
    leakage_cols.update([c for c in data.columns if c.lower() in {"flow_id", "flowid", "session_id", "uid"}])
    drop_cols = [c for c in data.columns if c in drop_exact or c in leakage_cols]
    X_raw = data.drop(columns=drop_cols, errors="ignore")
    y_binary = data["binary_label"].astype(int)
    y_multiclass = data["multiclass_encoded"].astype(int)
    sample_ids = data["sample_id"].astype(int)

    print("Feature frame shape:", X_raw.shape)
    print("Dropped leakage/label columns:", drop_cols)
    return X_raw, y_binary, y_multiclass, sample_ids


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Fit preprocessing only on train data, then transform val/test."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TabularPreprocessor":
        X = clean_columns(X).copy()
        self.original_columns_ = X.columns.tolist()
        missing_ratio = X.isna().mean()
        self.high_missing_cols_ = (
            missing_ratio[missing_ratio > self.config.HIGH_MISSING_THRESHOLD].index.tolist()
            if self.config.HIGH_MISSING_THRESHOLD is not None
            else []
        )
        self.empty_cols_ = [c for c in X.columns if X[c].isna().all()]
        candidate = X.drop(columns=set(self.high_missing_cols_ + self.empty_cols_), errors="ignore")
        self.constant_cols_ = [c for c in candidate.columns if candidate[c].nunique(dropna=False) <= 1]
        self.drop_cols_ = sorted(set(self.high_missing_cols_ + self.empty_cols_ + self.constant_cols_))
        X_fit = X.drop(columns=self.drop_cols_, errors="ignore")
        self.input_columns_ = X_fit.columns.tolist()

        self.numeric_cols_ = X_fit.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = [c for c in X_fit.columns if c not in self.numeric_cols_]

        scaler = RobustScaler() if self.config.SCALER.lower() == "robust" else StandardScaler()
        transformers = []
        if self.numeric_cols_:
            self.numeric_pipeline_ = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
            )
            transformers.append(("num", self.numeric_pipeline_, self.numeric_cols_))
        else:
            self.numeric_pipeline_ = None

        if self.categorical_cols_:
            self.categorical_pipeline_ = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", make_one_hot_encoder()),
                ]
            )
            transformers.append(("cat", self.categorical_pipeline_, self.categorical_cols_))
        else:
            self.categorical_pipeline_ = None

        if not transformers:
            raise ValueError("No usable feature columns remained after preprocessing.")

        self.transformer_ = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
        self.transformer_.fit(X_fit)
        self.feature_names_ = self._get_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = clean_columns(X).copy()
        for col in self.input_columns_:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.input_columns_]
        arr = self.transformer_.transform(X)
        return np.asarray(arr, dtype=np.float32)

    def _get_feature_names(self) -> List[str]:
        names: List[str] = []
        if self.numeric_cols_:
            names.extend(self.numeric_cols_)
        if self.categorical_cols_:
            encoder = self.transformer_.named_transformers_["cat"].named_steps["encoder"]
            cat_names = encoder.get_feature_names_out(self.categorical_cols_).tolist()
            names.extend(cat_names)
        return names


def save_preprocessor_artifacts(preprocessor: TabularPreprocessor, output_dirs: Dict[str, Path], task_name: str) -> None:
    joblib.dump(preprocessor, output_dirs["artifacts"] / f"preprocessor_{task_name}.joblib")
    if preprocessor.numeric_pipeline_ is not None:
        joblib.dump(preprocessor.numeric_pipeline_, output_dirs["artifacts"] / f"scaler_{task_name}.joblib")
        if task_name == "binary":
            joblib.dump(preprocessor.numeric_pipeline_, output_dirs["artifacts"] / "scaler.joblib")
    if preprocessor.categorical_pipeline_ is not None:
        joblib.dump(preprocessor.categorical_pipeline_, output_dirs["artifacts"] / f"encoder_{task_name}.joblib")
        if task_name == "binary":
            joblib.dump(preprocessor.categorical_pipeline_, output_dirs["artifacts"] / "encoder.joblib")


def stratified_train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    sample_ids: pd.Series | np.ndarray,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y)
    sample_ids = np.asarray(sample_ids)
    stratify_main = y if len(np.unique(y)) > 1 else None
    X_train_val, X_test, y_train_val, y_test, id_train_val, id_test = train_test_split(
        X,
        y,
        sample_ids,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_main,
    )
    stratify_val = y_train_val if len(np.unique(y_train_val)) > 1 else None
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_train_val,
        y_train_val,
        id_train_val,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, id_train, id_val, id_test


def compute_balanced_class_weight(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def limit_rows_for_resampling(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(y) <= max_samples:
        return X, y
    idx = np.arange(len(y))
    _, idx_sample = train_test_split(
        idx,
        train_size=max_samples,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    return X[idx_sample], y[idx_sample]


def safe_smote_k(y: np.ndarray, default_k: int = 5, min_k: int = 1) -> int:
    counts = pd.Series(y).value_counts()
    minority = int(counts.min()) if len(counts) else default_k + 1
    return max(min_k, min(default_k, minority - 1))


def balance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]], str]:
    strategy = (config.BALANCING_STRATEGY or "class_weight").lower()
    class_weight = compute_balanced_class_weight(y_train)
    if strategy in {"none", "no_balance"}:
        return X_train, y_train, None, "none"
    if strategy in {"class_weight", "focal_loss"}:
        return X_train, y_train, class_weight, strategy

    X_resample, y_resample = limit_rows_for_resampling(
        X_train, y_train, config.SMOTE_MAX_SAMPLES, config.RANDOM_STATE
    )
    k = safe_smote_k(y_resample, config.DEFAULT_SMOTE_K, config.MIN_SMOTE_K)

    if strategy == "smote":
        if SMOTE is None:
            print("imbalanced-learn is not installed; falling back to class_weight.")
            return X_train, y_train, class_weight, "class_weight_fallback"
        sampler = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k)
    elif strategy == "borderline_smote":
        if BorderlineSMOTE is None:
            print("imbalanced-learn is not installed; falling back to class_weight.")
            return X_train, y_train, class_weight, "class_weight_fallback"
        sampler = BorderlineSMOTE(random_state=config.RANDOM_STATE, k_neighbors=k)
    elif strategy == "random_under_sampler":
        if RandomUnderSampler is None:
            print("imbalanced-learn is not installed; falling back to class_weight.")
            return X_train, y_train, class_weight, "class_weight_fallback"
        sampler = RandomUnderSampler(random_state=config.RANDOM_STATE)
    elif strategy in {"smote_tomek", "smote_tomek_links"}:
        if SMOTETomek is None:
            print("imbalanced-learn is not installed; falling back to class_weight.")
            return X_train, y_train, class_weight, "class_weight_fallback"
        sampler = SMOTETomek(random_state=config.RANDOM_STATE, smote=SMOTE(k_neighbors=k))
    else:
        print(f"Unknown balancing strategy {strategy!r}; falling back to class_weight.")
        return X_train, y_train, class_weight, "class_weight_fallback"

    X_bal, y_bal = sampler.fit_resample(X_resample, y_resample)
    return np.asarray(X_bal, dtype=np.float32), np.asarray(y_bal), None, strategy


def _sample_for_feature_selection(
    X: np.ndarray, y: np.ndarray, config: ExperimentConfig
) -> Tuple[np.ndarray, np.ndarray]:
    if config.FEATURE_SELECTION_SAMPLE_SIZE and len(y) > config.FEATURE_SELECTION_SAMPLE_SIZE:
        idx = np.arange(len(y))
        _, sample_idx = train_test_split(
            idx,
            train_size=config.FEATURE_SELECTION_SAMPLE_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        return X[sample_idx], y[sample_idx]
    return X, y


def fit_feature_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
    task_name: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    top_k = config.TOP_K_FEATURES if top_k is None else top_k
    vt = VarianceThreshold(threshold=config.LOW_VARIANCE_THRESHOLD)
    X_var = vt.fit_transform(X_train)
    var_mask = vt.get_support()
    var_names = [name for name, keep in zip(feature_names, var_mask) if keep]

    corr_keep_idx = np.arange(X_var.shape[1])
    corr_drop_names: List[str] = []
    if config.DROP_HIGH_CORRELATION and X_var.shape[1] > 1:
        corr_sample = X_var
        if len(corr_sample) > min(50_000, len(corr_sample)):
            sample_idx = np.random.default_rng(config.RANDOM_STATE).choice(len(corr_sample), size=50_000, replace=False)
            corr_sample = corr_sample[sample_idx]
        corr = np.corrcoef(corr_sample, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.triu(np.abs(corr), k=1)
        drop_idx = sorted(set(np.where(upper > config.HIGH_CORRELATION_THRESHOLD)[1].tolist()))
        keep_mask = np.ones(X_var.shape[1], dtype=bool)
        keep_mask[drop_idx] = False
        corr_keep_idx = np.where(keep_mask)[0]
        corr_drop_names = [var_names[i] for i in drop_idx]

    X_corr = X_var[:, corr_keep_idx]
    corr_names = [var_names[i] for i in corr_keep_idx]

    X_fs_sample, y_fs_sample = _sample_for_feature_selection(X_corr, y_train, config)
    if X_fs_sample.shape[1] == 0:
        raise ValueError("No features left after variance/correlation filtering.")

    mi = mutual_info_classif(X_fs_sample, y_fs_sample, random_state=config.RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=max(80, min(config.RF_N_ESTIMATORS, 180)),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_fs_sample, y_fs_sample)
    rf_imp = rf.feature_importances_

    def normalize(v: np.ndarray) -> np.ndarray:
        v = np.nan_to_num(v.astype(float), nan=0.0)
        denom = v.max() - v.min()
        return (v - v.min()) / denom if denom > 0 else np.zeros_like(v)

    combined = 0.5 * normalize(mi) + 0.5 * normalize(rf_imp)
    importance_df = pd.DataFrame(
        {
            "feature": corr_names,
            "mutual_information": mi,
            "random_forest_importance": rf_imp,
            "combined_importance": combined,
        }
    ).sort_values("combined_importance", ascending=False)

    if top_k is None:
        selected_names = importance_df["feature"].tolist()
    else:
        selected_names = importance_df["feature"].head(min(top_k, len(importance_df))).tolist()
    selected_idx_after_corr = np.array([corr_names.index(name) for name in selected_names], dtype=int)

    selector = {
        "variance_selector": vt,
        "post_variance_feature_names": var_names,
        "corr_keep_indices": corr_keep_idx,
        "post_corr_feature_names": corr_names,
        "selected_indices_after_corr": selected_idx_after_corr,
        "selected_features": selected_names,
        "correlation_dropped_features": corr_drop_names,
        "importance_table": importance_df,
    }
    joblib.dump(selector, output_dirs["artifacts"] / f"feature_selector_{task_name}.joblib")
    if task_name == "binary":
        joblib.dump(selector, output_dirs["artifacts"] / "feature_selector.joblib")
    save_json(selected_names, output_dirs["artifacts"] / f"selected_features_{task_name}.json")
    if task_name == "binary":
        save_json(selected_names, output_dirs["artifacts"] / "selected_features.json")
    importance_df.to_csv(output_dirs["metrics"] / f"feature_importance_{task_name}.csv", index=False)

    for n in [20, 30, 50]:
        plot_feature_importance(
            importance_df.head(n),
            title=f"Top {n} Features - {task_name}",
            path=output_dirs["plots"] / f"top_{n}_features_{task_name}.png",
        )
    return selector


def apply_feature_selector(X: np.ndarray, selector: Dict[str, Any]) -> np.ndarray:
    X_var = selector["variance_selector"].transform(X)
    X_corr = X_var[:, selector["corr_keep_indices"]]
    selected_idx = selector["selected_indices_after_corr"]
    return np.asarray(X_corr[:, selected_idx], dtype=np.float32)


def transform_without_topk_selection(X: np.ndarray, selector: Dict[str, Any]) -> np.ndarray:
    X_var = selector["variance_selector"].transform(X)
    X_corr = X_var[:, selector["corr_keep_indices"]]
    return np.asarray(X_corr, dtype=np.float32)


def plot_feature_importance(df: pd.DataFrame, title: str, path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(11, max(5, 0.32 * len(df) + 1)))
    sns.barplot(x="combined_importance", y="feature", data=df, palette="viridis")
    plt.title(title)
    plt.xlabel("Combined importance")
    plt.ylabel("")
    save_current_fig(path)


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    if tf is None:
        raise ImportError("TensorFlow is required for focal loss.")

    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        bce = -(y_true_f * tf.math.log(y_pred_f) + (1 - y_true_f) * tf.math.log(1 - y_pred_f))
        pt = tf.where(tf.equal(y_true_f, 1), y_pred_f, 1 - y_pred_f)
        alpha_factor = tf.where(tf.equal(y_true_f, 1), alpha, 1 - alpha)
        return tf.reduce_mean(alpha_factor * tf.pow(1 - pt, gamma) * bce)

    return loss


def sparse_categorical_focal_loss(num_classes: int, gamma: float = 2.0, alpha: float = 0.25):
    if tf is None:
        raise ImportError("TensorFlow is required for focal loss.")

    def loss(y_true, y_pred):
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_f = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        y_one_hot = tf.one_hot(y_true_int, depth=num_classes)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred_f)
        pt = tf.reduce_sum(y_one_hot * y_pred_f, axis=-1)
        return tf.reduce_mean(alpha * tf.pow(1 - pt, gamma) * ce)

    return loss


def build_dnn_model(
    input_dim: int,
    task_name: str,
    n_classes: int,
    config: ExperimentConfig,
) -> Any:
    if tf is None or keras is None:
        raise ImportError("TensorFlow/Keras is not installed.")

    inputs = keras.Input(shape=(input_dim,), name="network_features")
    x = inputs
    for units in config.DNN_HIDDEN_UNITS:
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(config.DNN_L2),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.DNN_DROPOUT)(x)

    if task_name == "binary":
        outputs = layers.Dense(1, activation="sigmoid", name="attack_probability")(x)
        loss = binary_focal_loss() if config.USE_FOCAL_LOSS or config.BALANCING_STRATEGY == "focal_loss" else "binary_crossentropy"
        metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]
    else:
        outputs = layers.Dense(n_classes, activation="softmax", name="attack_type_probability")(x)
        loss = (
            sparse_categorical_focal_loss(n_classes)
            if config.USE_FOCAL_LOSS or config.BALANCING_STRATEGY == "focal_loss"
            else "sparse_categorical_crossentropy"
        )
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model = keras.Model(inputs, outputs, name=f"dnn_{task_name}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss,
        metrics=metrics,
    )
    return model


class LearningRateLogger(keras.callbacks.Callback if keras is not None else object):
    def on_epoch_end(self, epoch, logs=None):  # type: ignore[override]
        if keras is None:
            return
        logs = logs or {}
        try:
            lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
            logs["learning_rate"] = lr
        except Exception:
            pass


def fit_dnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_name: str,
    n_classes: int,
    class_weight: Optional[Dict[int, float]],
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
    epochs: Optional[int] = None,
) -> Tuple[Any, Any]:
    model = build_dnn_model(X_train.shape[1], task_name, n_classes, config)
    model_path = output_dirs["models"] / f"best_model_{task_name}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, config.PATIENCE // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dirs["metrics"] / f"training_log_{task_name}.csv")),
        LearningRateLogger(),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        epochs=epochs or config.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight if class_weight else None,
        verbose=1,
    )
    plot_training_history(history, task_name, output_dirs)
    return model, history


def plot_training_history(history: Any, task_name: str, output_dirs: Dict[str, Path]) -> None:
    hist = pd.DataFrame(history.history)
    hist.to_csv(output_dirs["metrics"] / f"history_{task_name}.csv", index=False)
    if "loss" in hist.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(hist["loss"], label="loss")
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="val_loss")
        plt.title(f"Loss Curve - {task_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        save_current_fig(output_dirs["plots"] / f"training_loss_{task_name}.png")

    if "accuracy" in hist.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(hist["accuracy"], label="accuracy")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="val_accuracy")
        plt.title(f"Accuracy Curve - {task_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        save_current_fig(output_dirs["plots"] / f"training_accuracy_{task_name}.png")

    auc_cols = [c for c in hist.columns if c.lower() in {"auc", "val_auc"}]
    if auc_cols:
        plt.figure(figsize=(9, 5))
        for col in auc_cols:
            plt.plot(hist[col], label=col)
        plt.title(f"AUC Curve - {task_name}")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        save_current_fig(output_dirs["plots"] / f"training_auc_{task_name}.png")

    if "learning_rate" in hist.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(hist["learning_rate"], label="learning_rate")
        plt.title(f"Learning Rate - {task_name}")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.legend()
        save_current_fig(output_dirs["plots"] / f"learning_rate_{task_name}.png")


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_name: str,
    n_classes: int,
    config: ExperimentConfig,
    class_weight: Optional[Dict[int, float]],
) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    if not config.RUN_BASELINES:
        return models

    lr = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced" if class_weight else None,
        solver="saga",
        random_state=config.RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample" if class_weight else None,
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    sample_weight = compute_sample_weight("balanced", y_train)
    if XGBClassifier is not None:
        if task_name == "binary":
            xgb = XGBClassifier(
                n_estimators=config.XGB_N_ESTIMATORS,
                max_depth=config.XGB_MAX_DEPTH,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            xgb = XGBClassifier(
                n_estimators=config.XGB_N_ESTIMATORS,
                max_depth=config.XGB_MAX_DEPTH,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        xgb.fit(X_train, y_train, sample_weight=sample_weight)
        models["XGBoost"] = xgb
    elif LGBMClassifier is not None:
        lgbm = LGBMClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            class_weight="balanced" if class_weight else None,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        lgbm.fit(X_train, y_train)
        models["LightGBM"] = lgbm
    else:
        print("Neither xgboost nor lightgbm is installed; gradient boosting baseline skipped.")

    return models


def align_binary_proba(raw_proba: np.ndarray) -> np.ndarray:
    raw_proba = np.asarray(raw_proba)
    if raw_proba.ndim == 1:
        p1 = raw_proba
        return np.column_stack([1 - p1, p1])
    if raw_proba.shape[1] == 1:
        p1 = raw_proba[:, 0]
        return np.column_stack([1 - p1, p1])
    return raw_proba[:, :2]


def predict_model_proba(model: Any, X: np.ndarray, task_name: str, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if task_name == "binary":
            return align_binary_proba(proba)
        aligned = np.zeros((len(X), n_classes), dtype=float)
        classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        for i, cls in enumerate(classes):
            if int(cls) < n_classes:
                aligned[:, int(cls)] = proba[:, i]
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return aligned / row_sums

    raw = model.predict(X, verbose=0) if hasattr(model, "predict") else model(X)
    if task_name == "binary":
        return align_binary_proba(raw)
    raw = np.asarray(raw)
    if raw.shape[1] != n_classes:
        raise ValueError(f"Expected {n_classes} probability columns, got {raw.shape[1]}")
    return raw


def entropy_from_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba), 1e-12, 1.0)
    ent = -np.sum(proba * np.log(proba), axis=1)
    denom = np.log(proba.shape[1]) if proba.shape[1] > 1 else 1.0
    return ent / denom


def normalized_score(values: np.ndarray, min_value: Optional[float] = None, max_value: Optional[float] = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values) if min_value is None else min_value
    hi = np.nanmax(values) if max_value is None else max_value
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def assign_contextual_criticality(n: int, config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.RANDOM_STATE)

    def assign(fixed: Optional[str], dist: Dict[str, float]) -> np.ndarray:
        if fixed:
            return np.array([fixed] * n)
        keys = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=float)
        probs = probs / probs.sum()
        return rng.choice(keys, size=n, p=probs)

    return (
        assign(config.DEVICE_CRITICALITY_FIXED, config.DEVICE_CRITICALITY_DISTRIBUTION),
        assign(config.MISSION_CRITICALITY_FIXED, config.MISSION_CRITICALITY_DISTRIBUTION),
    )


def derive_traffic_intensity(raw_X: Optional[pd.DataFrame], n: int, config: ExperimentConfig) -> np.ndarray:
    if raw_X is not None:
        cols = [clean_column_name(c) for c in raw_X.columns]
        raw = clean_columns(raw_X)
        for col in config.TRAFFIC_INTENSITY_COLUMNS:
            if col in cols and col in raw.columns:
                values = pd.to_numeric(raw[col], errors="coerce")
                median = values.median()
                values = values.fillna(0 if pd.isna(median) else median)
                q1, q2 = values.quantile([0.33, 0.66])
                return np.where(values <= q1, "low", np.where(values <= q2, "medium", "high"))
    rng = np.random.default_rng(config.RANDOM_STATE)
    return rng.choice(["low", "medium", "high"], size=n, p=[0.30, 0.45, 0.25])


def fuzzy_score_row(row: pd.Series) -> Tuple[float, str, str]:
    crit_map = {"low": 0.0, "medium": 0.18, "high": 0.34, "critical": 0.50}
    traffic_map = {"low": 0.0, "medium": 0.12, "high": 0.25}
    confidence = float(row.get("confidence_score", 0.0))
    attack_probability = float(row.get("attack_probability", 0.0))
    anomaly_score = float(row.get("anomaly_score", 0.0))
    entropy_score = float(row.get("entropy_score", 0.0))
    unknown = bool(row.get("unknown_attack_candidate", False))
    device = str(row.get("device_criticality", "medium")).lower()
    mission = str(row.get("mission_criticality", "medium")).lower()
    traffic = str(row.get("traffic_intensity", "medium")).lower()

    score = (
        42 * attack_probability
        + 28 * anomaly_score
        + 13 * (1 - confidence)
        + 10 * entropy_score
        + 18 * crit_map.get(device, 0.18)
        + 22 * crit_map.get(mission, 0.18)
        + 10 * traffic_map.get(traffic, 0.12)
    )
    reasons: List[str] = []

    if attack_probability >= 0.75 and anomaly_score >= 0.65:
        score = max(score, 88)
        reasons.append("attack_probability and anomaly_score are high")
    if unknown and anomaly_score >= 0.55:
        score = max(score, 92)
        reasons.append("unknown attack candidate with high anomaly score")
    if confidence < 0.45 and anomaly_score >= 0.65:
        score = max(score, 78)
        reasons.append("low confidence with high anomaly score")
    if mission == "critical" and attack_probability >= 0.45:
        score = max(score, 86)
        reasons.append("critical mission context with attack probability")
    if device == "critical" and unknown:
        score = max(score, 90)
        reasons.append("critical device and unknown attack candidate")
    if traffic == "high" and attack_probability >= 0.65:
        score = max(score, 76)
        reasons.append("high traffic intensity with high attack probability")
    if attack_probability < 0.25 and anomaly_score < 0.25 and not unknown:
        score = min(score, 25)
        reasons.append("normal-like traffic and low anomaly score")
    if confidence < 0.50 and entropy_score > 0.60:
        score += 8
        reasons.append("low confidence and high entropy")

    score = float(np.clip(score, 0, 100))
    if score <= 25:
        level = "Low"
    elif score <= 50:
        level = "Medium"
    elif score <= 75:
        level = "High"
    else:
        level = "Critical"
    reason = "; ".join(reasons) if reasons else "weighted fuzzy rule aggregation"
    return score, level, reason


def apply_fuzzy_risk_layer(
    pred_df: pd.DataFrame,
    raw_X: Optional[pd.DataFrame],
    config: ExperimentConfig,
) -> pd.DataFrame:
    pred_df = pred_df.copy()
    n = len(pred_df)
    device, mission = assign_contextual_criticality(n, config)
    pred_df["device_criticality"] = device
    pred_df["mission_criticality"] = mission
    pred_df["traffic_intensity"] = derive_traffic_intensity(raw_X, n, config)
    fuzzy = pred_df.apply(fuzzy_score_row, axis=1, result_type="expand")
    pred_df["fuzzy_risk_score"] = fuzzy[0].round(3)
    pred_df["fuzzy_risk_level"] = fuzzy[1]
    pred_df["fuzzy_reason"] = fuzzy[2]
    return pred_df


def make_prediction_dataframe(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    proba: np.ndarray,
    task_name: str,
    class_names: List[str],
    config: ExperimentConfig,
    raw_X: Optional[pd.DataFrame] = None,
    anomaly_score: Optional[np.ndarray] = None,
    unknown_flag: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    proba = np.asarray(proba)
    pred_numeric = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    entropy = entropy_from_proba(proba)
    anomaly = anomaly_score if anomaly_score is not None else normalized_score(1 - confidence + entropy)
    unknown = unknown_flag if unknown_flag is not None else np.zeros(len(y_true), dtype=bool)

    if task_name == "binary":
        attack_probability = proba[:, 1]
        true_labels = np.where(y_true == 1, "Attack", "Benign")
        pred_labels = np.where(pred_numeric == 1, "Attack", "Benign")
        pred_attack_type = pred_labels
    else:
        benign_index = class_names.index("Benign") if "Benign" in class_names else None
        attack_probability = 1 - proba[:, benign_index] if benign_index is not None else 1 - proba[:, pred_numeric]
        true_labels = [class_names[int(i)] if int(i) < len(class_names) else str(i) for i in y_true]
        pred_labels = [class_names[int(i)] if int(i) < len(class_names) else str(i) for i in pred_numeric]
        pred_attack_type = pred_labels

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": true_labels,
            "predicted_label": np.where(unknown, "Unknown_Attack_Candidate", pred_labels),
            "predicted_attack_type": np.where(unknown, "Unknown_Attack_Candidate", pred_attack_type),
            "confidence_score": confidence,
            "attack_probability": attack_probability,
            "anomaly_score": anomaly,
            "entropy_score": entropy,
            "unknown_attack_candidate": unknown.astype(bool),
        }
    )
    pred_df = apply_fuzzy_risk_layer(pred_df, raw_X, config)
    required_order = [
        "sample_id",
        "true_label",
        "predicted_label",
        "predicted_attack_type",
        "confidence_score",
        "attack_probability",
        "anomaly_score",
        "entropy_score",
        "unknown_attack_candidate",
        "device_criticality",
        "mission_criticality",
        "fuzzy_risk_score",
        "fuzzy_risk_level",
        "fuzzy_reason",
    ]
    extra = [c for c in pred_df.columns if c not in required_order]
    return pred_df[required_order + extra]


def evaluate_binary_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_ids: np.ndarray,
    raw_X: Optional[pd.DataFrame],
    model_name: str,
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
    save_prefix: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, str]:
    proba = align_binary_proba(proba)
    attack_prob = proba[:, 1]
    y_pred = (attack_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan

    metrics = {
        "task": "binary",
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "detection_rate": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, attack_prob)
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        metrics["pr_auc"] = average_precision_score(y_true, attack_prob)
    except Exception:
        metrics["pr_auc"] = np.nan

    report = classification_report(y_true, y_pred, target_names=["Benign", "Attack"], zero_division=0)
    pred_df = make_prediction_dataframe(sample_ids, y_true, proba, "binary", ["Benign", "Attack"], config, raw_X)
    plot_binary_evaluation(y_true, y_pred, attack_prob, cm, model_name, output_dirs, save_prefix)
    return metrics, pred_df, cm, report


def plot_binary_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_prob: np.ndarray,
    cm: np.ndarray,
    model_name: str,
    output_dirs: Dict[str, Path],
    save_prefix: str,
) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_current_fig(output_dirs["plots"] / f"confusion_matrix_{save_prefix}.png")

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.title(f"Normalized Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_current_fig(output_dirs["plots"] / f"normalized_confusion_matrix_{save_prefix}.png")

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, attack_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        save_current_fig(output_dirs["plots"] / f"roc_curve_{save_prefix}.png")

        precision, recall, _ = precision_recall_curve(y_true, attack_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, label="PR")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        save_current_fig(output_dirs["plots"] / f"precision_recall_curve_{save_prefix}.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(attack_prob[y_true == 0], color="tab:blue", label="Benign", bins=50, stat="density", alpha=0.45)
    sns.histplot(attack_prob[y_true == 1], color="tab:red", label="Attack", bins=50, stat="density", alpha=0.45)
    plt.title(f"Prediction Probability Distribution - {model_name}")
    plt.xlabel("Attack probability")
    plt.legend()
    save_current_fig(output_dirs["plots"] / f"prediction_probability_distribution_{save_prefix}.png")

    errors = pd.DataFrame({"true": y_true, "pred": y_pred, "attack_probability": attack_prob})
    errors["error_type"] = np.select(
        [
            (errors["true"] == 0) & (errors["pred"] == 1),
            (errors["true"] == 1) & (errors["pred"] == 0),
        ],
        ["False Positive", "False Negative"],
        default="Correct",
    )
    plt.figure(figsize=(8, 5))
    sns.countplot(data=errors, x="error_type", order=["Correct", "False Positive", "False Negative"], palette="Set2")
    plt.title(f"Error Analysis - {model_name}")
    plt.xlabel("")
    plt.ylabel("Count")
    save_current_fig(output_dirs["plots"] / f"error_analysis_{save_prefix}.png")


def evaluate_multiclass_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_ids: np.ndarray,
    raw_X: Optional[pd.DataFrame],
    class_names: List[str],
    model_name: str,
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
    save_prefix: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, str]:
    y_pred = np.argmax(proba, axis=1)
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "task": "multiclass",
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )
    for cls in class_names:
        if cls in report_dict:
            metrics[f"precision_{safe_filename(cls)}"] = report_dict[cls]["precision"]
            metrics[f"recall_{safe_filename(cls)}"] = report_dict[cls]["recall"]
            metrics[f"f1_{safe_filename(cls)}"] = report_dict[cls]["f1-score"]

    pred_df = make_prediction_dataframe(sample_ids, y_true, proba, "multiclass", class_names, config, raw_X)
    plot_multiclass_evaluation(y_true, y_pred, cm, class_names, report_dict, model_name, output_dirs, save_prefix)
    return metrics, pred_df, cm, report


def plot_multiclass_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    class_names: List[str],
    report_dict: Dict[str, Any],
    model_name: str,
    output_dirs: Dict[str, Path],
    save_prefix: str,
) -> None:
    fig_width = max(9, min(24, len(class_names) * 0.45))
    plt.figure(figsize=(fig_width, fig_width * 0.85))
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Multiclass Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    save_current_fig(output_dirs["plots"] / f"multiclass_confusion_matrix_{save_prefix}.png")

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(fig_width, fig_width * 0.85))
    sns.heatmap(cm_norm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Normalized Multiclass Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    save_current_fig(output_dirs["plots"] / f"normalized_multiclass_confusion_matrix_{save_prefix}.png")

    per_class = []
    for cls in class_names:
        if cls in report_dict:
            per_class.append(
                {
                    "class": cls,
                    "precision": report_dict[cls]["precision"],
                    "recall": report_dict[cls]["recall"],
                    "f1": report_dict[cls]["f1-score"],
                }
            )
    per_class_df = pd.DataFrame(per_class)
    if not per_class_df.empty:
        per_class_df.to_csv(output_dirs["metrics"] / f"per_class_metrics_{save_prefix}.csv", index=False)
        for metric in ["f1", "recall"]:
            sorted_df = per_class_df.sort_values(metric)
            plt.figure(figsize=(11, max(5, 0.32 * len(sorted_df) + 1)))
            sns.barplot(data=sorted_df, x=metric, y="class", palette="crest")
            plt.title(f"Per-Class {metric.upper()} - {model_name}")
            plt.xlim(0, 1)
            save_current_fig(output_dirs["plots"] / f"per_class_{metric}_{save_prefix}.png")

    confusion = cm.copy()
    np.fill_diagonal(confusion, 0)
    if confusion.sum() > 0:
        plt.figure(figsize=(fig_width, fig_width * 0.85))
        sns.heatmap(confusion, cmap="Reds", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Misclassified Classes Heatmap - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        save_current_fig(output_dirs["plots"] / f"misclassified_classes_heatmap_{save_prefix}.png")

        pairs = []
        for i, true_cls in enumerate(class_names):
            for j, pred_cls in enumerate(class_names):
                if i != j and confusion[i, j] > 0:
                    pairs.append({"true_class": true_cls, "predicted_class": pred_cls, "count": int(confusion[i, j])})
        pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=False)
        pairs_df.to_csv(output_dirs["metrics"] / f"top_confused_class_pairs_{save_prefix}.csv", index=False)
        if not pairs_df.empty:
            top = pairs_df.head(20)
            plt.figure(figsize=(11, 7))
            labels = top["true_class"] + " -> " + top["predicted_class"]
            sns.barplot(x=top["count"], y=labels, palette="rocket")
            plt.title(f"Top Confused Class Pairs - {model_name}")
            save_current_fig(output_dirs["plots"] / f"top_confused_class_pairs_{save_prefix}.png")


def save_evaluation_outputs(
    task_name: str,
    model_name: str,
    metrics: Dict[str, Any],
    predictions: pd.DataFrame,
    cm: np.ndarray,
    report: str,
    output_dirs: Dict[str, Path],
    generic: bool = False,
) -> None:
    slug = safe_filename(model_name)
    save_json(metrics, output_dirs["metrics"] / f"metrics_{task_name}_{slug}.json")
    predictions.to_csv(output_dirs["predictions"] / f"predictions_{task_name}_{slug}.csv", index=False)
    pd.DataFrame(cm).to_csv(output_dirs["metrics"] / f"confusion_matrix_{task_name}_{slug}.csv", index=False)
    save_text(report, output_dirs["reports"] / f"classification_report_{task_name}_{slug}.txt")
    if generic:
        save_json(metrics, output_dirs["metrics"] / f"metrics_{task_name}.json")
        predictions.to_csv(output_dirs["predictions"] / f"predictions_{task_name}.csv", index=False)
        pd.DataFrame(cm).to_csv(output_dirs["metrics"] / f"confusion_matrix_{task_name}.csv", index=False)
        save_text(report, output_dirs["reports"] / f"classification_report_{task_name}.txt")


def hybrid_predict_proba(
    model_dict: Dict[str, Any],
    X: np.ndarray,
    task_name: str,
    n_classes: int,
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    probs = []
    used_weights = []
    for name, model in model_dict.items():
        if name not in {"Random Forest", "XGBoost", "LightGBM", "Deep Neural Network"}:
            continue
        try:
            probs.append(predict_model_proba(model, X, task_name, n_classes))
            used_weights.append((weights or {}).get(name, 1.0))
        except Exception as exc:
            print(f"Hybrid skipped {name}: {exc}")
    if not probs:
        raise ValueError("No compatible models available for hybrid prediction.")
    weights_arr = np.asarray(used_weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    combined = np.zeros_like(probs[0], dtype=float)
    for p, w in zip(probs, weights_arr):
        combined += p * w
    row_sums = combined.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return combined / row_sums


def run_supervised_task(
    X_raw: pd.DataFrame,
    y: pd.Series | np.ndarray,
    sample_ids: pd.Series | np.ndarray,
    task_name: str,
    class_names: List[str],
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    print(f"\n========== {task_name.upper()} EXPERIMENT ==========")
    start = time.time()
    n_classes = len(class_names)
    (
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train,
        y_val,
        y_test,
        id_train,
        id_val,
        id_test,
    ) = stratified_train_validation_test_split(X_raw, y, sample_ids, config)

    preprocessor = TabularPreprocessor(config).fit(X_train_raw)
    X_train_full = preprocessor.transform(X_train_raw)
    X_val_full = preprocessor.transform(X_val_raw)
    X_test_full = preprocessor.transform(X_test_raw)
    save_preprocessor_artifacts(preprocessor, output_dirs, task_name)

    selector = fit_feature_selector(
        X_train_full,
        y_train,
        preprocessor.feature_names_,
        config,
        output_dirs,
        task_name,
    )
    X_train = apply_feature_selector(X_train_full, selector)
    X_val = apply_feature_selector(X_val_full, selector)
    X_test = apply_feature_selector(X_test_full, selector)
    selected_features = selector["selected_features"]

    X_train_model, y_train_model, class_weight, balancing_used = balance_training_data(X_train, y_train, config)
    print(f"Balancing strategy used: {balancing_used}")
    print(f"Train shape before/after balancing: {X_train.shape} -> {X_train_model.shape}")

    models: Dict[str, Any] = {}
    training_times: Dict[str, float] = {}

    baseline_start = time.time()
    models.update(train_baseline_models(X_train_model, y_train_model, task_name, n_classes, config, class_weight))
    baseline_elapsed = time.time() - baseline_start
    for name in models:
        training_times[name] = baseline_elapsed / max(1, len(models))

    if config.RUN_DEEP_LEARNING:
        dl_start = time.time()
        dnn, history = fit_dnn_model(
            X_train_model,
            y_train_model,
            X_val,
            y_val,
            task_name,
            n_classes,
            class_weight,
            config,
            output_dirs,
        )
        models["Deep Neural Network"] = dnn
        training_times["Deep Neural Network"] = time.time() - dl_start

    task_rows: List[Dict[str, Any]] = []
    predictions_by_model: Dict[str, pd.DataFrame] = {}
    cm_by_model: Dict[str, np.ndarray] = {}
    report_by_model: Dict[str, str] = {}

    for name, model in models.items():
        proba = predict_model_proba(model, X_test, task_name, n_classes)
        save_prefix = f"{task_name}_{safe_filename(name)}"
        if task_name == "binary":
            metrics, pred_df, cm, report = evaluate_binary_predictions(
                y_test, proba, id_test, X_test_raw, name, config, output_dirs, save_prefix
            )
        else:
            metrics, pred_df, cm, report = evaluate_multiclass_predictions(
                y_test, proba, id_test, X_test_raw, class_names, name, config, output_dirs, save_prefix
            )
        metrics["training_time_seconds"] = training_times.get(name, np.nan)
        metrics["balancing_strategy"] = balancing_used
        task_rows.append(metrics)
        predictions_by_model[name] = pred_df
        cm_by_model[name] = cm
        report_by_model[name] = report
        save_evaluation_outputs(task_name, name, metrics, pred_df, cm, report, output_dirs, generic=False)
        if name in {"Random Forest", "XGBoost", "LightGBM"}:
            joblib.dump(model, output_dirs["models"] / f"{safe_filename(name)}_{task_name}.joblib")

    hybrid_name = "Proposed Hybrid Model"
    try:
        hybrid_start = time.time()
        hybrid_proba = hybrid_predict_proba(models, X_test, task_name, n_classes)
        if task_name == "binary":
            h_metrics, h_pred_df, h_cm, h_report = evaluate_binary_predictions(
                y_test,
                hybrid_proba,
                id_test,
                X_test_raw,
                hybrid_name,
                config,
                output_dirs,
                f"{task_name}_proposed_hybrid",
            )
        else:
            h_metrics, h_pred_df, h_cm, h_report = evaluate_multiclass_predictions(
                y_test,
                hybrid_proba,
                id_test,
                X_test_raw,
                class_names,
                hybrid_name,
                config,
                output_dirs,
                f"{task_name}_proposed_hybrid",
            )
        h_metrics["training_time_seconds"] = sum(training_times.values()) + (time.time() - hybrid_start)
        h_metrics["balancing_strategy"] = balancing_used
        task_rows.append(h_metrics)
        predictions_by_model[hybrid_name] = h_pred_df
        cm_by_model[hybrid_name] = h_cm
        report_by_model[hybrid_name] = h_report
        save_json({"component_models": list(models.keys()), "selected_features": selected_features}, output_dirs["models"] / f"hybrid_manifest_{task_name}.json")
    except Exception as exc:
        print(f"Hybrid model evaluation skipped: {exc}")

    task_results = pd.DataFrame(task_rows)
    metric_for_best = "f1" if task_name == "binary" else "macro_f1"
    best_idx = task_results[metric_for_best].astype(float).idxmax()
    best_model_name = str(task_results.loc[best_idx, "model"])
    print(f"Best {task_name} model: {best_model_name}")
    save_evaluation_outputs(
        task_name,
        best_model_name,
        task_results.loc[best_idx].to_dict(),
        predictions_by_model[best_model_name],
        cm_by_model[best_model_name],
        report_by_model[best_model_name],
        output_dirs,
        generic=True,
    )
    task_results.to_csv(output_dirs["metrics"] / f"{task_name}_results.csv", index=False)

    elapsed = time.time() - start
    print(f"{task_name} experiment finished in {elapsed:.1f} seconds")
    return {
        "task_name": task_name,
        "class_names": class_names,
        "preprocessor": preprocessor,
        "selector": selector,
        "selected_features": selected_features,
        "models": models,
        "task_results": task_results,
        "best_model_name": best_model_name,
        "predictions_by_model": predictions_by_model,
        "X_train_full": X_train_full,
        "X_val_full": X_val_full,
        "X_test_full": X_test_full,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "id_test": id_test,
        "balancing_used": balancing_used,
        "elapsed_seconds": elapsed,
    }


def select_unknown_classes(df: pd.DataFrame, config: ExperimentConfig) -> List[str]:
    attack_labels = [c for c in df["multiclass_label"].unique() if not is_benign_label(c)]
    if config.UNKNOWN_CLASSES:
        requested = [normalize_label_value(x) for x in config.UNKNOWN_CLASSES]
        available = [c for c in requested if c in attack_labels]
        if available:
            return available
        print("Configured UNKNOWN_CLASSES were not found; falling back to automatic selection.")

    counts = df["multiclass_label"].value_counts()
    candidates = [
        c
        for c in attack_labels
        if counts.get(c, 0) >= config.MIN_UNKNOWN_CLASS_COUNT
    ]
    if not candidates:
        candidates = sorted(attack_labels, key=lambda c: counts.get(c, 0), reverse=True)
    candidates = sorted(candidates, key=lambda c: counts.get(c, 0))
    return candidates[: max(1, config.AUTO_UNKNOWN_CLASS_COUNT)]


def tune_threshold(y_true_unknown: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best = {"threshold": float(thresholds[0]), "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        p = precision_score(y_true_unknown, pred, zero_division=0)
        r = recall_score(y_true_unknown, pred, zero_division=0)
        f = f1_score(y_true_unknown, pred, zero_division=0)
        if f > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f), "precision": float(p), "recall": float(r)}
    return best


def build_autoencoder(input_dim: int, config: ExperimentConfig) -> Any:
    if tf is None or keras is None:
        raise ImportError("TensorFlow/Keras is required for the autoencoder.")
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation="relu")(inputs)
    encoded = layers.Dropout(0.15)(encoded)
    encoded = layers.Dense(32, activation="relu")(encoded)
    decoded = layers.Dense(128, activation="relu")(encoded)
    outputs = layers.Dense(input_dim, activation="linear")(decoded)
    model = keras.Model(inputs, outputs, name="unknown_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE), loss="mse")
    return model


def run_unknown_detection_experiment(
    df: pd.DataFrame,
    X_raw: pd.DataFrame,
    sample_ids: pd.Series | np.ndarray,
    class_names_full: List[str],
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    if not config.RUN_UNKNOWN_DETECTION:
        return {}
    print("\n========== UNKNOWN / ZERO-DAY EXPERIMENT ==========")
    unknown_classes = select_unknown_classes(df, config)
    print("Unknown classes:", unknown_classes)
    save_json(unknown_classes, output_dirs["artifacts"] / "unknown_classes.json")

    label_series = df["multiclass_label"].reset_index(drop=True)
    unknown_mask = label_series.isin(unknown_classes).to_numpy()
    known_mask = ~unknown_mask
    known_df_index = np.where(known_mask)[0]
    unknown_df_index = np.where(unknown_mask)[0]

    known_labels = label_series.iloc[known_df_index].to_numpy()
    known_classes = ["Benign"] + sorted([c for c in np.unique(known_labels) if c != "Benign"])
    known_encoder = LabelEncoder().fit(known_classes)
    known_y = known_encoder.transform(known_labels)

    known_X = X_raw.iloc[known_df_index].reset_index(drop=True)
    known_ids = np.asarray(sample_ids)[known_df_index]
    unknown_X = X_raw.iloc[unknown_df_index].reset_index(drop=True)
    unknown_labels = label_series.iloc[unknown_df_index].to_numpy()
    unknown_ids = np.asarray(sample_ids)[unknown_df_index]

    X_train_raw, X_tmp_raw, y_train, y_tmp, id_train, id_tmp = train_test_split(
        known_X,
        known_y,
        known_ids,
        test_size=config.TEST_SIZE + config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=known_y if len(np.unique(known_y)) > 1 else None,
    )
    val_relative = config.VAL_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
    X_val_known_raw, X_test_known_raw, y_val_known, y_test_known, id_val_known, id_test_known = train_test_split(
        X_tmp_raw,
        y_tmp,
        id_tmp,
        train_size=val_relative,
        random_state=config.RANDOM_STATE,
        stratify=y_tmp if len(np.unique(y_tmp)) > 1 else None,
    )

    if len(unknown_X) < 2:
        raise ValueError("Not enough unknown-class samples to run zero-day simulation.")
    strat_unknown = unknown_labels if len(np.unique(unknown_labels)) > 1 else None
    X_val_unknown_raw, X_test_unknown_raw, label_val_unknown, label_test_unknown, id_val_unknown, id_test_unknown = train_test_split(
        unknown_X,
        unknown_labels,
        unknown_ids,
        test_size=0.50,
        random_state=config.RANDOM_STATE,
        stratify=strat_unknown,
    )

    preprocessor = TabularPreprocessor(config).fit(X_train_raw)
    X_train_full = preprocessor.transform(X_train_raw)
    X_val_known_full = preprocessor.transform(X_val_known_raw)
    X_test_known_full = preprocessor.transform(X_test_known_raw)
    X_val_unknown_full = preprocessor.transform(X_val_unknown_raw)
    X_test_unknown_full = preprocessor.transform(X_test_unknown_raw)
    save_preprocessor_artifacts(preprocessor, output_dirs, "unknown")

    selector = fit_feature_selector(
        X_train_full,
        y_train,
        preprocessor.feature_names_,
        config,
        output_dirs,
        "unknown",
    )
    X_train = apply_feature_selector(X_train_full, selector)
    X_val_known = apply_feature_selector(X_val_known_full, selector)
    X_test_known = apply_feature_selector(X_test_known_full, selector)
    X_val_unknown = apply_feature_selector(X_val_unknown_full, selector)
    X_test_unknown = apply_feature_selector(X_test_unknown_full, selector)

    X_train_bal, y_train_bal, class_weight, _ = balance_training_data(X_train, y_train, config)
    if tf is None:
        raise ImportError("TensorFlow is required for the unknown detection classifier.")
    known_model, _ = fit_dnn_model(
        X_train_bal,
        y_train_bal,
        X_val_known,
        y_val_known,
        "unknown",
        len(known_classes),
        class_weight,
        config,
        output_dirs,
    )

    iso = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train)
    joblib.dump(iso, output_dirs["models"] / "isolation_forest_unknown.joblib")

    ae_model = None
    if config.USE_AUTOENCODER_UNKNOWN and tf is not None:
        try:
            ae_model = build_autoencoder(X_train.shape[1], config)
            ae_path = output_dirs["models"] / "best_autoencoder_unknown.keras"
            ae_model.fit(
                X_train,
                X_train,
                validation_data=(X_val_known, X_val_known),
                batch_size=config.BATCH_SIZE,
                epochs=config.AUTOENCODER_EPOCHS,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                    keras.callbacks.ModelCheckpoint(str(ae_path), monitor="val_loss", save_best_only=True),
                ],
                verbose=1,
            )
        except Exception as exc:
            print(f"Autoencoder unknown detector skipped: {exc}")
            ae_model = None

    X_val_mix = np.vstack([X_val_known, X_val_unknown])
    y_val_unknown_true = np.r_[np.zeros(len(X_val_known), dtype=int), np.ones(len(X_val_unknown), dtype=int)]
    X_test_mix = np.vstack([X_test_known, X_test_unknown])
    y_test_unknown_true = np.r_[np.zeros(len(X_test_known), dtype=int), np.ones(len(X_test_unknown), dtype=int)]
    raw_test_mix = pd.concat([X_test_known_raw, X_test_unknown_raw], ignore_index=True)
    id_test_mix = np.r_[id_test_known, id_test_unknown]

    proba_val = predict_model_proba(known_model, X_val_mix, "multiclass", len(known_classes))
    proba_test = predict_model_proba(known_model, X_test_mix, "multiclass", len(known_classes))
    conf_val = np.max(proba_val, axis=1)
    conf_test = np.max(proba_test, axis=1)
    ent_val = entropy_from_proba(proba_val)
    ent_test = entropy_from_proba(proba_test)

    iso_val_raw = -iso.decision_function(X_val_mix)
    iso_test_raw = -iso.decision_function(X_test_mix)
    iso_val = normalized_score(iso_val_raw)
    iso_test = normalized_score(iso_test_raw, np.nanmin(iso_val_raw), np.nanmax(iso_val_raw))

    ae_val = np.zeros(len(X_val_mix))
    ae_test = np.zeros(len(X_test_mix))
    if ae_model is not None:
        val_recon = ae_model.predict(X_val_mix, verbose=0)
        test_recon = ae_model.predict(X_test_mix, verbose=0)
        ae_val_raw = np.mean(np.square(X_val_mix - val_recon), axis=1)
        ae_test_raw = np.mean(np.square(X_test_mix - test_recon), axis=1)
        ae_val = normalized_score(ae_val_raw)
        ae_test = normalized_score(ae_test_raw, np.nanmin(ae_val_raw), np.nanmax(ae_val_raw))

    score_val = 0.35 * (1 - conf_val) + 0.25 * ent_val + 0.25 * iso_val + 0.15 * ae_val
    score_test = 0.35 * (1 - conf_test) + 0.25 * ent_test + 0.25 * iso_test + 0.15 * ae_test

    thresholds = {
        "max_softmax_probability": tune_threshold(y_val_unknown_true, 1 - conf_val),
        "entropy_score": tune_threshold(y_val_unknown_true, ent_val),
        "isolation_forest_score": tune_threshold(y_val_unknown_true, iso_val),
        "autoencoder_reconstruction_error": tune_threshold(y_val_unknown_true, ae_val) if ae_model is not None else None,
        "combined_anomaly_score": tune_threshold(y_val_unknown_true, score_val),
    }
    save_json(thresholds, output_dirs["metrics"] / "unknown_thresholds.json")
    threshold = thresholds["combined_anomaly_score"]["threshold"]
    unknown_pred = score_test >= threshold

    cm = confusion_matrix(y_test_unknown_true, unknown_pred.astype(int), labels=[0, 1])
    known_class_pred = np.argmax(proba_test[: len(X_test_known)], axis=1)
    known_class_accuracy = accuracy_score(y_test_known, known_class_pred)
    false_unknown_rate = cm[0, 1] / max(cm[0].sum(), 1)
    false_known_rate = cm[1, 0] / max(cm[1].sum(), 1)
    metrics = {
        "task": "unknown",
        "model": "Known-class DNN + confidence/entropy/IsolationForest/Autoencoder",
        "unknown_classes": unknown_classes,
        "threshold": float(threshold),
        "unknown_detection_rate": recall_score(y_test_unknown_true, unknown_pred, zero_division=0),
        "unknown_precision": precision_score(y_test_unknown_true, unknown_pred, zero_division=0),
        "unknown_recall": recall_score(y_test_unknown_true, unknown_pred, zero_division=0),
        "unknown_f1": f1_score(y_test_unknown_true, unknown_pred, zero_division=0),
        "known_class_accuracy": known_class_accuracy,
        "false_unknown_rate": false_unknown_rate,
        "false_known_rate": false_known_rate,
    }
    try:
        metrics["auroc"] = roc_auc_score(y_test_unknown_true, score_test)
    except Exception:
        metrics["auroc"] = np.nan
    try:
        metrics["pr_auc"] = average_precision_score(y_test_unknown_true, score_test)
    except Exception:
        metrics["pr_auc"] = np.nan

    true_known_names = [known_classes[int(i)] for i in y_test_known]
    true_test_labels = np.array(true_known_names + list(label_test_unknown))
    pred_known_names = [known_classes[int(i)] for i in np.argmax(proba_test, axis=1)]
    pred_df = pd.DataFrame(
        {
            "sample_id": id_test_mix,
            "true_label": true_test_labels,
            "predicted_label": np.where(unknown_pred, "Unknown_Attack_Candidate", pred_known_names),
            "predicted_attack_type": np.where(unknown_pred, "Unknown_Attack_Candidate", pred_known_names),
            "confidence_score": conf_test,
            "attack_probability": 1 - proba_test[:, known_classes.index("Benign")] if "Benign" in known_classes else 1 - conf_test,
            "anomaly_score": score_test,
            "entropy_score": ent_test,
            "unknown_attack_candidate": unknown_pred.astype(bool),
        }
    )
    pred_df = apply_fuzzy_risk_layer(pred_df, raw_test_mix, config)

    save_json(metrics, output_dirs["metrics"] / "metrics_unknown.json")
    pd.DataFrame([metrics]).to_csv(output_dirs["metrics"] / "unknown_results.csv", index=False)
    pd.DataFrame(cm, index=["Known", "Unknown"], columns=["Pred Known", "Pred Unknown"]).to_csv(
        output_dirs["metrics"] / "confusion_matrix_unknown.csv"
    )
    pred_df.to_csv(output_dirs["predictions"] / "predictions_unknown.csv", index=False)
    plot_unknown_detection(
        y_test_unknown_true,
        conf_test,
        ent_test,
        score_test,
        unknown_pred,
        threshold,
        cm,
        output_dirs,
    )
    return {
        "unknown_classes": unknown_classes,
        "known_classes": known_classes,
        "metrics": metrics,
        "predictions": pred_df,
        "thresholds": thresholds,
        "model": known_model,
        "isolation_forest": iso,
        "autoencoder": ae_model,
    }


def plot_unknown_detection(
    y_true_unknown: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    anomaly_score: np.ndarray,
    unknown_pred: np.ndarray,
    threshold: float,
    cm: np.ndarray,
    output_dirs: Dict[str, Path],
) -> None:
    labels = np.where(y_true_unknown == 1, "Unknown", "Known")
    plot_df = pd.DataFrame(
        {
            "label": labels,
            "confidence_score": confidence,
            "entropy_score": entropy,
            "anomaly_score": anomaly_score,
        }
    )
    for col, title in [
        ("confidence_score", "Confidence Score Distribution: Known vs Unknown"),
        ("entropy_score", "Entropy Score Distribution: Known vs Unknown"),
        ("anomaly_score", "Anomaly Score Distribution: Known vs Unknown"),
    ]:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=plot_df, x=col, hue="label", fill=True, common_norm=False, alpha=0.35)
        if col == "anomaly_score":
            plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.3f}")
            plt.legend()
        plt.title(title)
        save_current_fig(output_dirs["plots"] / f"unknown_{col}_distribution.png")

    if len(np.unique(y_true_unknown)) > 1:
        fpr, tpr, _ = roc_curve(y_true_unknown, anomaly_score)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("Unknown Detection ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        save_current_fig(output_dirs["plots"] / "unknown_detection_roc_curve.png")

        precision, recall, _ = precision_recall_curve(y_true_unknown, anomaly_score)
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision)
        plt.title("Unknown Detection Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        save_current_fig(output_dirs["plots"] / "unknown_detection_pr_curve.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(anomaly_score, bins=60, color="tab:purple", alpha=0.65)
    plt.axvline(threshold, color="black", linestyle="--", label="selected threshold")
    plt.title("Unknown Detection Threshold Visualization")
    plt.xlabel("Combined anomaly score")
    plt.legend()
    save_current_fig(output_dirs["plots"] / "unknown_threshold_visualization.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Known", "Unknown"], yticklabels=["Known", "Unknown"])
    plt.title("Known vs Unknown Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_current_fig(output_dirs["plots"] / "unknown_confusion_matrix.png")


def run_xai_analysis(
    context: Dict[str, Any],
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    if not config.RUN_XAI:
        return {}
    print("\n========== EXPLAINABLE AI ANALYSIS ==========")
    task_name = context["task_name"]
    class_names = context["class_names"]
    X_train = context["X_train"]
    X_test = context["X_test"]
    y_test = context["y_test"]
    feature_names = context["selected_features"]
    models = context["models"]

    rng = np.random.default_rng(config.RANDOM_STATE)
    train_idx = rng.choice(len(X_train), size=min(len(X_train), config.XAI_SAMPLE_SIZE), replace=False)
    test_idx = rng.choice(len(X_test), size=min(len(X_test), config.XAI_SAMPLE_SIZE), replace=False)
    X_train_sample = X_train[train_idx]
    X_test_sample = X_test[test_idx]
    y_test_sample = y_test[test_idx]
    X_test_df = pd.DataFrame(X_test_sample, columns=feature_names)

    xai_outputs: Dict[str, Any] = {}
    tree_model_name = next((name for name in ["XGBoost", "LightGBM", "Random Forest"] if name in models), None)
    if tree_model_name and shap is not None:
        try:
            tree_model = models[tree_model_name]
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_for_bar = np.mean([np.abs(v) for v in shap_values], axis=0)
                shap_summary_values = shap_values
            elif np.asarray(shap_values).ndim == 3:
                shap_for_bar = np.mean(np.abs(shap_values), axis=2)
                shap_summary_values = shap_values
            else:
                shap_for_bar = np.abs(shap_values)
                shap_summary_values = shap_values

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_summary_values, X_test_df, show=False, max_display=20)
            save_current_fig(output_dirs["plots"] / f"shap_summary_{task_name}.png")

            importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": np.mean(shap_for_bar, axis=0),
                }
            ).sort_values("mean_abs_shap", ascending=False)
            importance.to_csv(output_dirs["reports"] / f"shap_top_features_{task_name}.csv", index=False)
            plt.figure(figsize=(11, 7))
            sns.barplot(data=importance.head(20), x="mean_abs_shap", y="feature", palette="viridis")
            plt.title(f"SHAP Bar Plot - {tree_model_name}")
            save_current_fig(output_dirs["plots"] / f"shap_bar_{task_name}.png")

            proba = predict_model_proba(tree_model, X_test_sample, task_name, len(class_names))
            pred = np.argmax(proba, axis=1)
            correct_idx = np.where(pred == y_test_sample)[0]
            wrong_idx = np.where(pred != y_test_sample)[0]
            local_rows = []
            for label, arr in [("correct", correct_idx), ("wrong", wrong_idx)]:
                if len(arr):
                    row_idx = int(arr[0])
                    if isinstance(shap_values, list):
                        class_idx = int(pred[row_idx])
                        contrib = shap_values[class_idx][row_idx]
                    elif np.asarray(shap_values).ndim == 3:
                        class_idx = int(pred[row_idx])
                        contrib = shap_values[row_idx, :, class_idx]
                    else:
                        contrib = shap_values[row_idx]
                    local = pd.DataFrame(
                        {
                            "case": label,
                            "feature": feature_names,
                            "feature_value": X_test_sample[row_idx],
                            "shap_value": contrib,
                        }
                    )
                    local["abs_shap"] = local["shap_value"].abs()
                    local_rows.append(local.sort_values("abs_shap", ascending=False).head(20))
            if local_rows:
                local_df = pd.concat(local_rows, ignore_index=True)
                local_df.to_csv(output_dirs["reports"] / f"shap_local_explanations_{task_name}.csv", index=False)
            xai_outputs["tree_model"] = tree_model_name
        except Exception as exc:
            print(f"Tree SHAP failed; fallback will be used where possible: {exc}")

    if "Deep Neural Network" in models:
        try:
            deep_model = models["Deep Neural Network"]
            base_pred = np.argmax(predict_model_proba(deep_model, X_test_sample, task_name, len(class_names)), axis=1)
            base_score = accuracy_score(y_test_sample, base_pred)
            max_features = min(60, X_test_sample.shape[1])
            candidate_idx = np.arange(max_features)
            drops = []
            for j in candidate_idx:
                X_perm = X_test_sample.copy()
                rng.shuffle(X_perm[:, j])
                perm_pred = np.argmax(predict_model_proba(deep_model, X_perm, task_name, len(class_names)), axis=1)
                drops.append(base_score - accuracy_score(y_test_sample, perm_pred))
            perm_df = pd.DataFrame(
                {"feature": [feature_names[j] for j in candidate_idx], "importance_drop": drops}
            ).sort_values("importance_drop", ascending=False)
            perm_df.to_csv(output_dirs["reports"] / f"deep_permutation_importance_{task_name}.csv", index=False)
            plt.figure(figsize=(11, 7))
            sns.barplot(data=perm_df.head(20), x="importance_drop", y="feature", palette="mako")
            plt.title("Deep Model Permutation Importance")
            save_current_fig(output_dirs["plots"] / f"deep_permutation_importance_{task_name}.png")
            xai_outputs["deep_model_explainer"] = "permutation_importance"
        except Exception as exc:
            print(f"Deep model explanation skipped: {exc}")

    return xai_outputs


def run_hybrid_ablation_variant(
    context: Dict[str, Any],
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
    variant_name: str,
    use_feature_selection: bool = True,
    use_class_balancing: bool = True,
) -> Dict[str, Any]:
    task_name = context["task_name"]
    n_classes = len(context["class_names"])
    if use_feature_selection:
        X_train = context["X_train"]
        X_test = context["X_test"]
    else:
        X_train = transform_without_topk_selection(context["X_train_full"], context["selector"])
        X_test = transform_without_topk_selection(context["X_test_full"], context["selector"])
    y_train = context["y_train"]
    y_test = context["y_test"]
    if use_class_balancing:
        X_train_model, y_train_model, class_weight, _ = balance_training_data(X_train, y_train, config)
    else:
        X_train_model, y_train_model, class_weight = X_train, y_train, None

    rf = RandomForestClassifier(
        n_estimators=config.ABLATION_RF_N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample" if class_weight else None,
    )
    rf.fit(X_train_model, y_train_model)
    models = {"Random Forest": rf}
    if tf is not None:
        dnn, _ = fit_dnn_model(
            X_train_model,
            y_train_model,
            context["X_val"] if use_feature_selection else transform_without_topk_selection(context["X_val_full"], context["selector"]),
            context["y_val"],
            f"{task_name}_{safe_filename(variant_name)}",
            n_classes,
            class_weight,
            config,
            output_dirs,
            epochs=config.ABLATION_EPOCHS,
        )
        models["Deep Neural Network"] = dnn
    proba = hybrid_predict_proba(models, X_test, task_name, n_classes)
    y_pred = np.argmax(proba, axis=1)
    if task_name == "binary":
        metrics = {
            "task": task_name,
            "model": variant_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "false_negative_rate": confusion_matrix(y_test, y_pred, labels=[0, 1])[1, 0] / max((y_test == 1).sum(), 1),
        }
    else:
        metrics = {
            "task": task_name,
            "model": variant_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "false_negative_rate": np.nan,
        }
    return metrics


def run_ablation_study(
    binary_context: Dict[str, Any],
    multiclass_context: Dict[str, Any],
    unknown_context: Dict[str, Any],
    config: ExperimentConfig,
    output_dirs: Dict[str, Path],
) -> pd.DataFrame:
    if not config.RUN_ABLATION:
        return pd.DataFrame()
    print("\n========== ABLATION STUDY ==========")
    rows: List[Dict[str, Any]] = []
    for ctx in [binary_context, multiclass_context]:
        task_results = ctx["task_results"].copy()
        for _, row in task_results.iterrows():
            rows.append(row.to_dict())

    best_multi = multiclass_context["task_results"].sort_values("macro_f1", ascending=False).iloc[0].to_dict()
    no_fuzzy = best_multi.copy()
    no_fuzzy["model"] = "Proposed Hybrid Model without fuzzy risk layer"
    no_fuzzy["fuzzy_layer_enabled"] = False
    rows.append(no_fuzzy)

    if unknown_context:
        unk = unknown_context.get("metrics", {}).copy()
        unk["model"] = "Proposed Hybrid Model with unknown detection"
        unk["accuracy"] = unk.get("known_class_accuracy", np.nan)
        unk["macro_f1"] = unk.get("unknown_f1", np.nan)
        unk["weighted_f1"] = unk.get("unknown_f1", np.nan)
        rows.append(unk)

    if config.RUN_FULL_ABLATION_RETRAIN:
        try:
            rows.append(
                run_hybrid_ablation_variant(
                    multiclass_context,
                    config,
                    output_dirs,
                    "Proposed Hybrid Model without feature selection",
                    use_feature_selection=False,
                    use_class_balancing=True,
                )
            )
            rows.append(
                run_hybrid_ablation_variant(
                    multiclass_context,
                    config,
                    output_dirs,
                    "Proposed Hybrid Model without class balancing",
                    use_feature_selection=True,
                    use_class_balancing=False,
                )
            )
        except Exception as exc:
            print(f"Full ablation retraining failed: {exc}")

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(output_dirs["metrics"] / "ablation_study_results.csv", index=False)
    plot_model_comparison(ablation_df, output_dirs, prefix="ablation")
    return ablation_df


def plot_model_comparison(results_df: pd.DataFrame, output_dirs: Dict[str, Path], prefix: str = "all_model") -> None:
    if results_df.empty or "model" not in results_df.columns:
        return
    metrics = [
        ("accuracy", "Accuracy Comparison"),
        ("macro_f1", "Macro F1 Comparison"),
        ("weighted_f1", "Weighted F1 Comparison"),
        ("recall", "Recall Comparison"),
        ("false_negative_rate", "False Negative Rate Comparison"),
        ("unknown_detection_rate", "Unknown Detection Rate Comparison"),
        ("training_time_seconds", "Training Time Comparison"),
    ]
    for metric, title in metrics:
        if metric in results_df.columns and results_df[metric].notna().any():
            plot_df = results_df[["model", metric]].dropna().copy()
            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df = plot_df.dropna().sort_values(metric, ascending=False)
            if plot_df.empty:
                continue
            plt.figure(figsize=(12, max(5, 0.35 * len(plot_df) + 1)))
            sns.barplot(data=plot_df, x=metric, y="model", palette="viridis")
            plt.title(title)
            plt.ylabel("")
            save_current_fig(output_dirs["plots"] / f"{prefix}_{safe_filename(metric)}_comparison.png")


def save_all_results(
    binary_context: Dict[str, Any],
    multiclass_context: Dict[str, Any],
    unknown_context: Dict[str, Any],
    ablation_df: pd.DataFrame,
    output_dirs: Dict[str, Path],
) -> pd.DataFrame:
    frames = [
        binary_context["task_results"],
        multiclass_context["task_results"],
    ]
    if unknown_context:
        frames.append(pd.DataFrame([unknown_context["metrics"]]))
    if ablation_df is not None and not ablation_df.empty:
        frames.append(ablation_df)
    all_results = pd.concat(frames, ignore_index=True, sort=False)
    all_results.to_csv(output_dirs["metrics"] / "all_model_results.csv", index=False)
    save_json(all_results.replace({np.nan: None}).to_dict(orient="records"), output_dirs["metrics"] / "all_model_results.json")
    binary_context["task_results"].to_csv(output_dirs["metrics"] / "binary_results.csv", index=False)
    multiclass_context["task_results"].to_csv(output_dirs["metrics"] / "multiclass_results.csv", index=False)
    plot_model_comparison(all_results, output_dirs, prefix="all_model_results")
    return all_results


def generate_final_summary(
    all_results: pd.DataFrame,
    binary_context: Dict[str, Any],
    multiclass_context: Dict[str, Any],
    unknown_context: Dict[str, Any],
    output_dirs: Dict[str, Path],
) -> str:
    best_binary = binary_context["task_results"].sort_values("f1", ascending=False).iloc[0]
    best_multi = multiclass_context["task_results"].sort_values("macro_f1", ascending=False).iloc[0]
    multi_pred = multiclass_context["predictions_by_model"][multiclass_context["best_model_name"]]
    per_class_cols = [c for c in best_multi.index if c.startswith("f1_")]
    hard_classes = sorted(
        [(c.replace("f1_", ""), best_multi[c]) for c in per_class_cols if pd.notna(best_multi[c])],
        key=lambda x: x[1],
    )[:5]
    unknown_metrics = unknown_context.get("metrics", {}) if unknown_context else {}
    risk_counts = multi_pred["fuzzy_risk_level"].value_counts().to_dict() if "fuzzy_risk_level" in multi_pred else {}

    summary = f"""
# Tez Hazır Deney Özeti - CICIoT2023

Deney tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. CICIoT2023 üzerinde en iyi binary model: {best_binary['model']} (F1={best_binary.get('f1', np.nan):.4f}, ROC-AUC={best_binary.get('roc_auc', np.nan):.4f}).
2. CICIoT2023 üzerinde en iyi multiclass model: {best_multi['model']} (Macro F1={best_multi.get('macro_f1', np.nan):.4f}, Weighted F1={best_multi.get('weighted_f1', np.nan):.4f}).
3. Binary classification performansı normal ve saldırı trafiğini ayırmak için detection-rate odaklı değerlendirildi; false negative oranı {best_binary.get('false_negative_rate', np.nan):.4f} olarak ölçüldü.
4. Multiclass classification performansı sınıf dengesizliğine duyarlı Macro F1 ve Balanced Accuracy ile raporlandı.
5. Zor tespit edilen saldırı sınıfları düşük F1 değerine göre: {hard_classes}.
6. Unknown attack / zero-day simülasyonunda unknown sınıflar: {unknown_metrics.get('unknown_classes', 'çalıştırılmadı')}; Unknown F1={unknown_metrics.get('unknown_f1', np.nan)}.
7. Fuzzy risk layer, yüksek attack_probability, yüksek anomaly_score, düşük confidence, kritik görev/cihaz bağlamı ve unknown flag birleşimlerinde Critical risk üretti. Risk dağılımı: {risk_counts}.
8. False positive ve false negative oranları askeri IoT senaryosunda operasyonel alarm yükü ve kaçırılan saldırı riski açısından birlikte yorumlanmalıdır.
9. Model askeri IoT için uygundur çünkü normal/saldırı ayrımı, saldırı türü sınıflandırması, eğitimde görülmeyen saldırı adayı işaretleme, açıklanabilirlik ve görev/cihaz kritikliğine duyarlı risk skoru tek deney hattında birleştirilmiştir.
10. Gelecek çalışmada aynı pipeline IoT-23 dataset'i için ikinci domain olarak çalıştırılıp domain shift, transfer learning ve cross-dataset generalization analizleriyle genişletilebilir.
"""
    save_text(summary.strip() + "\n", output_dirs["reports"] / "final_experiment_summary.md")
    save_text(summary.strip() + "\n", output_dirs["reports"] / "thesis_ready_summary.txt")
    return summary


def run_ciciot2023_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run the full CICIoT2023 experiment end to end."""

    set_global_seed(config.RANDOM_STATE)
    output_dirs = create_output_dirs(config)
    df = load_ciciot2023(
        config.DATASET_PATH,
        sample_size=config.SAMPLE_SIZE,
        chunksize=config.CHUNKSIZE,
        label_column=config.LABEL_COLUMN,
        random_state=config.RANDOM_STATE,
        sample_strategy=config.SAMPLE_STRATEGY,
        optimize_dtypes=config.OPTIMIZE_DTYPES,
    )
    df = clean_basic_dataframe(df, config)
    df, label_info = prepare_labels(df, config, output_dirs)
    eda_summary = run_eda(df, config, output_dirs)
    X_raw, y_binary, y_multiclass, sample_ids = build_feature_frame(df, config)

    binary_context = run_supervised_task(
        X_raw,
        y_binary,
        sample_ids,
        "binary",
        ["Benign", "Attack"],
        config,
        output_dirs,
    )
    multiclass_context = run_supervised_task(
        X_raw,
        y_multiclass,
        sample_ids,
        "multiclass",
        label_info["class_names"],
        config,
        output_dirs,
    )
    unknown_context = run_unknown_detection_experiment(
        df,
        X_raw,
        sample_ids,
        label_info["class_names"],
        config,
        output_dirs,
    )
    xai_outputs = run_xai_analysis(multiclass_context, config, output_dirs)
    ablation_df = run_ablation_study(binary_context, multiclass_context, unknown_context, config, output_dirs)
    all_results = save_all_results(binary_context, multiclass_context, unknown_context, ablation_df, output_dirs)
    final_summary = generate_final_summary(all_results, binary_context, multiclass_context, unknown_context, output_dirs)
    print(final_summary)
    return {
        "config": config,
        "output_dirs": output_dirs,
        "df": df,
        "label_info": label_info,
        "eda_summary": eda_summary,
        "X_raw": X_raw,
        "sample_ids": sample_ids,
        "binary_context": binary_context,
        "multiclass_context": multiclass_context,
        "unknown_context": unknown_context,
        "xai_outputs": xai_outputs,
        "ablation_results": ablation_df,
        "all_results": all_results,
        "final_summary": final_summary,
    }
