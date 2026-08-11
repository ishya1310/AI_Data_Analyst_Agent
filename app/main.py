from pathlib import Path
from uuid import uuid4
import json
import re

import duckdb
import pandas as pd

from fastapi import FastAPI, File, HTTPException, UploadFile


BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DUCKDB_DIR = BASE_DIR / "data" / "duckdb"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_DIR.mkdir(parents=True, exist_ok=True)



app = FastAPI(
    title="AI Data Analyst Agent",
    description="Week 1 - Dataset Engine",
    version="0.1.0",
)


def create_safe_table_name(filename: str) -> str:
    """
    Convert a filename into a safe DuckDB table name.

    Example:
        sales-data.csv -> sales_data
    """

    name = Path(filename).stem

    # Replace special characters with _
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # Prevent empty names
    if not name:
        name = "dataset"

    # Prevent table names starting with numbers
    if name[0].isdigit():
        name = f"table_{name}"

    return name.lower()


def read_dataset(file_path: Path) -> pd.DataFrame:
    """
    Read CSV or Excel file into a Pandas DataFrame.
    """

    extension = file_path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    elif extension in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Only CSV and Excel files are supported."
        )


def detect_column_type(series: pd.Series) -> str:
    """
    Detect a semantic type for a column.

    Possible types:
        numeric
        categorical
        date
        boolean
    """

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    return "categorical"




def generate_dataset_profile(df: pd.DataFrame) -> dict:
    """
    Generate a complete dataset profile.
    """


    rows = len(df)
    columns = len(df.columns)


    missing_values = df.isnull().sum()

    total_missing = int(missing_values.sum())

    if rows > 0 and columns > 0:
        missing_percentage = round(
            (total_missing / (rows * columns)) * 100,
            2
        )
    else:
        missing_percentage = 0.0



    duplicate_rows = int(df.duplicated().sum())



    numeric_columns = []
    categorical_columns = []
    date_columns = []
    boolean_columns = []

    for column in df.columns:

        column_type = detect_column_type(df[column])

        if column_type == "numeric":
            numeric_columns.append(column)

        elif column_type == "categorical":
            categorical_columns.append(column)

        elif column_type == "date":
            date_columns.append(column)

        elif column_type == "boolean":
            boolean_columns.append(column)


    schema = []

    for column in df.columns:

        schema.append({
            "column": column,
            "pandas_dtype": str(df[column].dtype),
            "semantic_type": detect_column_type(df[column]),
            "nullable": bool(df[column].isnull().any()),
            "unique_values": int(
                df[column].nunique(dropna=True)
            )
        })


    numeric_summary = {}

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) == 0:

            numeric_summary[column] = {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None
            }

            continue

        numeric_summary[column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max())
        }


    categorical_summary = {}

    for column in categorical_columns:

        series = df[column].astype("string")

        value_counts = (
            series
            .value_counts(dropna=True)
            .head(10)
        )

        top_values = []

        for value, count in value_counts.items():

            top_values.append({
                "value": str(value),
                "count": int(count)
            })

        categorical_summary[column] = {
            "unique_values": int(
                series.nunique(dropna=True)
            ),
            "top_values": top_values
        }



    profile = {

        "rows": rows,

        "columns": columns,

        "column_names": list(df.columns),

        "missing_cells": total_missing,

        "missing_percentage": missing_percentage,

        "duplicate_rows": duplicate_rows,

        "column_groups": {

            "numeric": numeric_columns,

            "categorical": categorical_columns,

            "date": date_columns,

            "boolean": boolean_columns
        },

        "schema": schema,

        "numeric_summary": numeric_summary,

        "categorical_summary": categorical_summary
    }

    return profile




@app.get("/")
def root():

    return {
        "project": "AI Data Analyst Agent",
        "phase": "Week 1 - Dataset Engine",
        "status": "running",
        "documentation": "/docs"
    }



@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):

    filename = file.filename or ""

    extension = Path(filename).suffix.lower()


    allowed_extensions = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload CSV or Excel files."
            )
        )



    dataset_id = str(uuid4())

    stored_filename = f"{dataset_id}{extension}"

    file_path = UPLOAD_DIR / stored_filename



    file_content = await file.read()

    file_path.write_bytes(file_content)

    try:


        df = read_dataset(file_path)



        if df.empty:

            raise ValueError(
                "The uploaded dataset is empty."
            )



        table_name = create_safe_table_name(filename)



        database_path = (
            DUCKDB_DIR /
            f"{dataset_id}.duckdb"
        )

        connection = duckdb.connect(
            str(database_path)
        )

        try:

            # Register Pandas DataFrame
            connection.register(
                "uploaded_dataframe",
                df
            )

            # Create DuckDB table
            connection.execute(
                f'''
                CREATE TABLE "{table_name}"
                AS
                SELECT *
                FROM uploaded_dataframe
                '''
            )

        finally:

            connection.close()



        profile = generate_dataset_profile(df)


        metadata = {

            "dataset_id": dataset_id,

            "original_filename": filename,

            "stored_filename": stored_filename,

            "table_name": table_name,

            "database": str(
                database_path.relative_to(BASE_DIR)
            ),

            "profile": profile
        }

        metadata_path = (
            UPLOAD_DIR /
            f"{dataset_id}.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=4,
                default=str
            ),
            encoding="utf-8"
        )



        return {

            "message": "Dataset uploaded successfully.",

            "dataset_id": dataset_id,

            "filename": filename,

            "table_name": table_name,

            "rows": profile["rows"],

            "columns": profile["columns"],

            "profile_url": (
                f"/datasets/{dataset_id}/profile"
            ),

            "schema_url": (
                f"/datasets/{dataset_id}/schema"
            )
        }

    except Exception as error:

        # Remove uploaded file if processing fails
        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=f"Could not process dataset: {error}"
        )


def load_metadata(dataset_id: str) -> dict:

    metadata_path = (
        UPLOAD_DIR /
        f"{dataset_id}.json"
    )

    if not metadata_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    return json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )




@app.get("/datasets/{dataset_id}/profile")
def get_dataset_profile(
    dataset_id: str
):

    metadata = load_metadata(
        dataset_id
    )

    return metadata["profile"]




@app.get("/datasets/{dataset_id}/schema")
def get_dataset_schema(
    dataset_id: str
):

    metadata = load_metadata(
        dataset_id
    )

    return {

        "dataset_id": dataset_id,

        "table_name": metadata["table_name"],

        "schema": metadata["profile"]["schema"]
    }