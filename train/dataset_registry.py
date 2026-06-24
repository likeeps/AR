#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Sequence, Tuple

from utils.preprocess_adult import preprocess_adult
from utils.preprocess_claims import preprocess_claims
from utils.preprocess_dblp10k import preprocess_dblp10k
from utils.preprocess_hospital import preprocess_hospital
from utils.preprocess_tax import preprocess_tax
from utils.preprocess_biocase import preprocess_biocase
from utils.preprocess_biocase_gathering import preprocess_biocase_gathering
from utils.preprocess_biocase_namedareas import preprocess_biocase_namedareas
from utils.preprocess_biocase_highertaxon import preprocess_biocase_highertaxon
from utils.preprocess_biocase_identification import preprocess_biocase_identification
from utils.preprocess_generic import PreprocessConfig, preprocess_generic


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _normalize_meta(meta: Dict) -> Dict[str, object]:
    return {
        "all_cols": list(meta.get("all_cols") or meta.get("columns") or []),
        "numeric_cols": list(meta.get("numeric_cols", meta.get("numerical_cols", []))),
        "categorical_cols": list(meta.get("categorical_cols", [])),
        "discrete_numeric_cols": list(meta.get("discrete_numeric_cols", [])),
        "continuous_cols": list(meta.get("continuous_cols", [])),
        "date_cols": list(meta.get("date_cols", [])),
        "dropped_columns": list(meta.get("dropped_columns", [])),
        "target_col": str(meta.get("target_col", "")),
        "category_maps": dict(meta.get("category_maps", {})),
    }


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlowHyperparameters:
    run_id: int
    hidden_features: int = 256
    num_flow_steps: int = 8
    num_transform_blocks: int = 2
    num_bins: int = 8
    tail_bound: float = 8.0
    dropout_probability: float = 0.1
    use_batch_norm: bool = False
    train_batch_size: int = 512
    val_batch_size: int = 4096
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    monitor_interval: int = 500
    num_training_steps: int = 60000
    grad_norm_clip_value: float = 5.0
    anneal_learning_rate: bool = True
    seed: int = 1638128
    val_ratio: float = 0.1
    num_workers: int = 0
    min_training_steps: int = 10000
    early_stop_patience: int = 30
    validation_add_noise: bool = True
    validation_noise_seed_offset: int = 2026
    train_loss_window: int = 200

    @property
    def validation_noise_seed(self) -> int:
        return self.seed + self.validation_noise_seed_offset


@dataclass(frozen=True)
class TrainingSchema:
    columns: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    target_col: str
    discrete_cols: List[str]
    continuous_cols: List[str]
    log1p_cols: List[str]
    dropped_columns: List[str]


@dataclass(frozen=True)
class DatasetTrainingSpec:
    dataset_name: str
    preprocess_fn: Callable[..., Tuple[object, list[str]]]
    schema_builder: Callable[[Dict], TrainingSchema]
    hyperparameters: FlowHyperparameters
    input_csv: str
    output_npy: str
    meta_json: str
    notes: Tuple[str, ...]
    search_overrides: dict = field(default_factory=dict)
    structural_overrides: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

COMMON_HPARAMS = FlowHyperparameters(run_id=1)


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------

def _build_simple_schema(raw_meta: Dict) -> TrainingSchema:
    """Default schema builder — works for all datasets.

    Date columns (when present in metadata) are merged into discrete_cols so that
    datasets like claims automatically get correct treatment without a custom builder.
    """
    meta = _normalize_meta(raw_meta)
    columns = list(meta["all_cols"])
    categorical_cols = [col for col in meta["categorical_cols"] if col in columns]
    numeric_cols = [col for col in meta["numeric_cols"] if col in columns]
    target_col = str(meta["target_col"])
    discrete_cols = _dedupe_keep_order(
        list(meta["discrete_numeric_cols"]) + list(meta["date_cols"]) + categorical_cols
    )
    if target_col and target_col in columns and target_col not in discrete_cols:
        discrete_cols.append(target_col)
    continuous_cols = [col for col in meta["continuous_cols"] if col in columns]
    if not continuous_cols:
        # Fallback: treat remaining numeric columns as continuous.
        discrete_set = set(discrete_cols)
        continuous_cols = [col for col in numeric_cols if col not in discrete_set]
    return TrainingSchema(
        columns=columns,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        discrete_cols=discrete_cols,
        continuous_cols=continuous_cols,
        log1p_cols=[],
        dropped_columns=list(meta["dropped_columns"]),
    )


# ---------------------------------------------------------------------------
# Generic preprocessor wrapper
# ---------------------------------------------------------------------------

def _make_generic_preprocess(config: PreprocessConfig):
    """Create a preprocess_fn from a PreprocessConfig."""
    def _preprocess(input_file: str = config.input_csv,
                    output_file: str = config.output_npy,
                    project_path: str | None = None):
        active_config = replace(config, input_csv=input_file, output_npy=output_file)
        return preprocess_generic(active_config, project_path)
    return _preprocess


# ---------------------------------------------------------------------------
# Declarative dataset specs
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Minimal declarative config for adding a new dataset."""
    dataset_name: str
    input_csv: str
    output_npy: str
    meta_json: str
    all_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str] = field(default_factory=list)
    continuous_cols: list[str] = field(default_factory=list)
    discrete_numeric_cols: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    target_col: str = ""
    notes: str = ""
    hparams: FlowHyperparameters | None = None
    schema_builder: Callable | None = None
    preprocess_fn: Callable | None = None
    search_overrides: dict = field(default_factory=dict)
    structural_overrides: dict = field(default_factory=dict)


# All standard datasets — add new entries here only.
DATASET_SPEC_LIST: list[DatasetSpec] = [
    DatasetSpec(
        dataset_name="biocase",
        input_csv="t_biocase_gathering_agent_r72738_c18.csv",
        output_npy="biocase.npy", meta_json="biocase_meta.json",
        all_cols=["_unitguid", "_datasetguid", "PersonFullName", "PersonInheritedName",
                  "Sequence", "Gath_Country_Name"],
        categorical_cols=["_unitguid", "_datasetguid", "PersonFullName",
                          "PersonInheritedName", "Gath_Country_Name"],
        discrete_numeric_cols=["Sequence"],
        dropped_columns=["_unitguid_id", "Gath_LocalityText", "Gath_HigherGeography",
                         "Gath_Country_NameISO3166_A2", "Gath_ScientificName",
                         "Gath_DateTime_Begin", "Gath_DateTime_End",
                         "Gath_AreaDetail", "Gath_LocalityVerbatim",
                         "Gath_HigherGeographySea", "Gath_HigherGeographyWaterBody"],
        notes="BioCase gathering agent data (6 effective columns). "
              "Identifiers and 100%-null columns are dropped.",
    ),
    DatasetSpec(
        dataset_name="biocase_gathering",
        input_csv="t_biocase_gathering_r90992_c35.csv",
        output_npy="biocase_gathering.npy", meta_json="biocase_gathering_meta.json",
        all_cols=["_unitguid", "_datasetguid", "Gath_AreaDetail",
                  "Gath_Country_Name", "Gath_DateTime_Begin", "Gath_DateTime_End"],
        categorical_cols=["_unitguid", "_datasetguid", "Gath_AreaDetail",
                          "Gath_Country_Name", "Gath_DateTime_Begin", "Gath_DateTime_End"],
        dropped_columns=["Gath_LocalityText", "Gath_HigherGeography",
                         "Gath_Country_NameISO3166_A2", "Gath_ScientificName",
                         "Gath_LocalityVerbatim", "Gath_HigherGeographySea",
                         "Gath_HigherGeographyWaterBody", "Gath_AreaName",
                         "Gath_AreaClass", "Gath_AreaCode",
                         "Gath_DateTime_DateText", "Gath_DateTime_TimeZone",
                         "Gath_DateTime_ISO8601DateTimeBegin",
                         "Gath_DateTime_ISO8601DateTimeEnd",
                         "Gath_DateTime_DayNumberBegin",
                         "Gath_DateTime_DayNumberEnd",
                         "Gath_DateTime_TimeOfDayBegin",
                         "Gath_DateTime_TimeOfDayEnd",
                         "Gath_DateTime_PeriodExplicit",
                         "Gath_DateTime_Method", "Gath_DateTime_Method_language",
                         "Gath_DateTime_Notes"],
        notes="BioCase gathering data (r90992, 6 effective columns). "
              "Ground truth FD: Gath_AreaDetail -> Gath_Country_Name.",
    ),
    DatasetSpec(
        dataset_name="biocase_namedareas",
        input_csv="t_biocase_gathering_namedareas_r137711_c11.csv",
        output_npy="biocase_namedareas.npy", meta_json="biocase_namedareas_meta.json",
        all_cols=["_unitguid", "_datasetguid", "Gath_AreaName", "Gath_AreaClass",
                  "Gath_AreaCode", "Gath_AreaDetail", "Sequence"],
        categorical_cols=["_unitguid", "_datasetguid", "Gath_AreaName",
                          "Gath_AreaClass", "Gath_AreaCode", "Gath_AreaDetail"],
        discrete_numeric_cols=["Sequence"],
        dropped_columns=["Gath_DateTime_Begin", "Gath_DateTime_End"],
        notes="BioCase namedareas data (r137711, 6 effective columns). "
              "GT FDs: AreaName->AreaClass, AreaName->AreaCode, AreaCode->AreaName/AreaClass.",
    ),
    DatasetSpec(
        dataset_name="biocase_highertaxon",
        input_csv="t_biocase_identification_highertaxon_r562959_c3.csv",
        output_npy="biocase_highertaxon.npy", meta_json="biocase_highertaxon_meta.json",
        all_cols=["_identificationguid", "HigherTaxonName", "HigherTaxonRank"],
        categorical_cols=["_identificationguid", "HigherTaxonName", "HigherTaxonRank"],
        notes="BioCase highertaxon data (r562959, 3 columns, no drops). "
              "GT FD: HigherTaxonName -> HigherTaxonRank.",
    ),
    DatasetSpec(
        dataset_name="biocase_identification",
        input_csv="t_biocase_identification_r91800_c38.csv",
        output_npy="biocase_identification.npy", meta_json="biocase_identification_meta.json",
        all_cols=["ScientificName", "FullScientificNameString", "GenusOrMonomial",
                  "Subgenus", "SpeciesEpithet", "SubspeciesEpithet",
                  "AuthorTeamOriginalAndYear", "AuthorTeamParenthesisAndYear"],
        categorical_cols=["ScientificName", "FullScientificNameString", "GenusOrMonomial",
                          "Subgenus", "SpeciesEpithet", "SubspeciesEpithet",
                          "AuthorTeamOriginalAndYear", "AuthorTeamParenthesisAndYear"],
        dropped_columns=["_unitguid", "_identificationguid", "_timestamp", "PreferredFlag",
                         "CombinationAuthorTeamAndYear", "Breed", "NamedIndividual",
                         "IdentificationQualifier", "IdentificationQualifier_insertionpoint",
                         "NameAddendum", "InformalNameString", "InformalNameString_language",
                         "Code", "NonFlag", "StoredUnderFlag", "ResultRole",
                         "Date_DateText", "Date_TimeZone", "Date_ISODateTimeBegin",
                         "Date_TimeOfDayBegin", "DayNumberBegin", "Date_ISODateTimeEnd",
                         "Date_TimeOfDayEnd", "Date_DayNumberEnd", "PeriodExplicit",
                         "Method", "Method_language", "Notes", "VerificationLevel",
                         "IdentificationHistory"],
        notes="BioCase identification data (r91800, 8 effective columns). "
              "14 GT FDs among ScientificName, FullScientificNameString, etc.",
    ),
    DatasetSpec(
        dataset_name="tax",
        input_csv="tax.csv", output_npy="tax.npy", meta_json="tax_meta.json",
        all_cols=["FName", "LName", "Gender", "AreaCode", "Phone", "City", "State", "Zip",
                  "MaritalStatus", "HasChild", "Salary", "Rate",
                  "SingleExemp", "MarriedExemp", "ChildExemp"],
        categorical_cols=["FName", "LName", "Gender", "AreaCode", "Phone", "City", "State",
                          "Zip", "MaritalStatus", "HasChild"],
        numeric_cols=["Salary", "Rate"],
        continuous_cols=["Salary", "Rate"],
        discrete_numeric_cols=["SingleExemp", "MarriedExemp", "ChildExemp"],
        notes="Name/location/code columns are label-encoded as categorical. "
              "Salary and Rate are continuous. Exemptions are discrete numeric.",
    ),
    DatasetSpec(
        dataset_name="adult",
        input_csv="adult.csv", output_npy="adult.npy", meta_json="adult_meta.json",
        all_cols=["age", "workclass", "fnlwgt", "education", "education_num",
                  "marital_status", "occupation", "relationship", "race", "sex",
                  "capital_gain", "capital_loss", "hours_per_week", "native_country", "outcome"],
        categorical_cols=["workclass", "education", "marital_status", "occupation",
                          "relationship", "race", "sex", "native_country"],
        numeric_cols=["fnlwgt", "age", "capital_gain", "capital_loss",
                      "hours_per_week", "education_num"],
        continuous_cols=["fnlwgt"],
        discrete_numeric_cols=["age", "capital_gain", "capital_loss",
                               "hours_per_week", "education_num"],
        target_col="outcome",
        preprocess_fn=preprocess_adult,
        notes="Adult income data. Categorical columns are label-encoded; numeric columns "
              "keep raw values (standardized during training).",
    ),
    DatasetSpec(
        dataset_name="claims",
        input_csv="claims.csv", output_npy="claims.npy", meta_json="claims_meta.json",
        all_cols=["DateReceived", "IncidentDate", "AirportCode", "AirportName", "AirlineName",
                  "ClaimType", "ClaimSite", "Item", "ClaimAmount", "Status",
                  "CloseAmount", "Disposition"],
        categorical_cols=["AirportCode", "AirportName", "AirlineName", "ClaimType",
                          "ClaimSite", "Item", "Status", "Disposition"],
        numeric_cols=["ClaimAmount", "CloseAmount"],
        continuous_cols=["ClaimAmount", "CloseAmount"],
        preprocess_fn=preprocess_claims,
        notes="TSA Claims data. ClaimNumber excluded; dates converted to days since epoch; "
              "currency parsed. Date columns treated as discrete features.",
    ),
    DatasetSpec(
        dataset_name="hospital",
        input_csv="hospital.csv", output_npy="hospital.npy", meta_json="hospital_meta.json",
        all_cols=["ProviderNumber", "HospitalName", "City", "State", "ZIPCode", "CountyName",
                  "PhoneNumber", "HospitalType", "HospitalOwner", "EmergencyService",
                  "Condition", "MeasureCode", "MeasureName", "Sample", "StateAvg"],
        categorical_cols=["ProviderNumber", "HospitalName", "City", "State", "ZIPCode",
                          "CountyName", "PhoneNumber", "HospitalType", "HospitalOwner",
                          "EmergencyService", "Condition", "MeasureCode", "MeasureName",
                          "StateAvg"],
        numeric_cols=["Sample"],
        discrete_numeric_cols=["Sample"],
        preprocess_fn=preprocess_hospital,
        notes="Hospital data. Categorical columns are label-encoded; Sample parsed as "
              "discrete patient count.",
    ),
    DatasetSpec(
        dataset_name="dblp10k",
        input_csv="dblp10k.csv", output_npy="dblp10k.npy", meta_json="dblp10k_meta.json",
        all_cols=["sameentity", "samename", "author1", "author2", "key1", "key2",
                  "p1type", "p1author", "p1title", "p1booktitle", "p1booktitlefull",
                  "p1year", "p1journal", "p1journalfull", "p1publisher", "p1series",
                  "p1id", "p1key", "p2type", "p2author", "p2title", "p2booktitle",
                  "p2booktitlefull", "p2year", "p2journal", "p2journalfull",
                  "p2publisher", "p2series", "p2id", "p2key"],
        categorical_cols=["sameentity", "samename", "author1", "author2", "key1", "key2",
                          "p1type", "p1author", "p1title", "p1booktitle", "p1booktitlefull",
                          "p1journal", "p1journalfull", "p1publisher", "p1series",
                          "p1id", "p1key", "p2type", "p2author", "p2title", "p2booktitle",
                          "p2booktitlefull", "p2journal", "p2journalfull",
                          "p2publisher", "p2series", "p2id", "p2key"],
        numeric_cols=["p1year", "p2year"],
        discrete_numeric_cols=["p1year", "p2year"],
        dropped_columns=["p1editor", "p1address", "p2editor", "p2address"],
        preprocess_fn=preprocess_dblp10k,
        notes="DBLP10K entity resolution data. Four fully-missing columns dropped; "
              "identifier-like columns remain categorical; publication years are discrete numeric.",
    ),
    # DatasetSpec(
    #     dataset_name="newdataset",
    #     input_csv="newdataset.csv",
    #     output_npy="newdataset.npy",
    #     meta_json="newdataset_meta.json",
    #     all_cols=["col_a", "col_b", "col_c"],
    #     categorical_cols=["col_a", "col_b"],
    #     continuous_cols=["col_c"],
    #     # dropped_columns=[],  # 可选
    #     # target_col="",        # 可选
    #     # search_overrides={},  # 可选
    #     # structural_overrides={},  # 可选
    #     notes="Description of the dataset.",
    # ),
]


def _build_dataset_spec(spec: DatasetSpec) -> DatasetTrainingSpec:
    """Convert a declarative DatasetSpec to a full DatasetTrainingSpec."""
    preprocess_fn = spec.preprocess_fn or _make_generic_preprocess(
        PreprocessConfig(
            dataset_name=spec.dataset_name,
            input_csv=spec.input_csv, output_npy=spec.output_npy, meta_json=spec.meta_json,
            all_cols=spec.all_cols, categorical_cols=spec.categorical_cols,
            numeric_cols=spec.numeric_cols, continuous_cols=spec.continuous_cols,
            discrete_numeric_cols=spec.discrete_numeric_cols,
            dropped_columns=spec.dropped_columns, target_col=spec.target_col, note=spec.notes,
        )
    )
    schema_builder = spec.schema_builder or _build_simple_schema
    hparams = spec.hparams or COMMON_HPARAMS
    return DatasetTrainingSpec(
        dataset_name=spec.dataset_name,
        preprocess_fn=preprocess_fn,
        schema_builder=schema_builder,
        hyperparameters=hparams,
        input_csv=spec.input_csv,
        output_npy=spec.output_npy,
        meta_json=spec.meta_json,
        notes=(spec.notes,) if isinstance(spec.notes, str) else tuple(spec.notes),
        search_overrides=spec.search_overrides,
        structural_overrides=spec.structural_overrides,
    )


# ---------------------------------------------------------------------------
# Full registry: auto-generated from DATASET_SPEC_LIST
# ---------------------------------------------------------------------------

DATASET_SPECS: Dict[str, DatasetTrainingSpec] = {
    spec.dataset_name: _build_dataset_spec(spec) for spec in DATASET_SPEC_LIST
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_spec(dataset_name: str) -> DatasetTrainingSpec:
    key = dataset_name.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {sorted(DATASET_SPECS)}")
    return DATASET_SPECS[key]


def get_supported_datasets() -> tuple[str, ...]:
    """Return all registered dataset names. Used by AR/config.py for SUPPORTED_DATASETS."""
    return tuple(sorted(DATASET_SPECS.keys()))


__all__ = [
    "DATASET_SPECS",
    "DatasetTrainingSpec",
    "DatasetSpec",
    "FlowHyperparameters",
    "TrainingSchema",
    "get_dataset_spec",
    "get_supported_datasets",
]
