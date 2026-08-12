from pathlib import Path

import pandas as pd


class DatasetValidator:
    """
    Validates uploaded datasets before they are processed
    and stored in DuckDB.
    """



    ALLOWED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls"
    }

    MAX_FILE_SIZE_MB = 50

    MAX_ROWS = 1_000_000

    MIN_COLUMNS = 1

    MAX_COLUMNS = 200


    @classmethod
    def validate_extension(
        cls,
        filename: str
    ):
        """
        Validate the uploaded file extension.
        """

        extension = Path(
            filename
        ).suffix.lower()

        if extension not in cls.ALLOWED_EXTENSIONS:

            raise ValueError(
                "Unsupported file type. "
                "Only CSV and Excel files are supported."
            )

        return extension



    @classmethod
    def validate_file_size(
        cls,
        file_size: int
    ):
        """
        Validate uploaded file size.

        file_size is measured in bytes.
        """

        max_size = (
            cls.MAX_FILE_SIZE_MB
            * 1024
            * 1024
        )

        if file_size > max_size:

            raise ValueError(
                f"File size exceeds the maximum "
                f"limit of {cls.MAX_FILE_SIZE_MB} MB."
            )



    @classmethod
    def validate_dataframe(
        cls,
        dataframe: pd.DataFrame
    ):
        """
        Validate the Pandas DataFrame.
        """



        if dataframe.empty:

            raise ValueError(
                "The uploaded dataset is empty."
            )



        column_count = len(
            dataframe.columns
        )

        if column_count < cls.MIN_COLUMNS:

            raise ValueError(
                "Dataset must contain at least "
                "one column."
            )

        if column_count > cls.MAX_COLUMNS:

            raise ValueError(
                f"Dataset contains too many columns. "
                f"Maximum allowed is {cls.MAX_COLUMNS}."
            )



        row_count = len(
            dataframe
        )

        if row_count > cls.MAX_ROWS:

            raise ValueError(
                f"Dataset contains too many rows. "
                f"Maximum allowed is {cls.MAX_ROWS}."
            )


        duplicated_columns = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .tolist()
        )

        if duplicated_columns:

            raise ValueError(
                "Dataset contains duplicate column names: "
                + ", ".join(
                    map(
                        str,
                        duplicated_columns
                    )
                )
            )


        empty_column_names = [

            str(column)

            for column in dataframe.columns

            if not str(column).strip()
        ]

        if empty_column_names:

            raise ValueError(
                "Dataset contains empty column names."
            )



        empty_columns = [

            str(column)

            for column in dataframe.columns

            if dataframe[column].isna().all()
        ]

        if empty_columns:

            raise ValueError(
                "Dataset contains completely empty "
                "columns: "
                + ", ".join(empty_columns)
            )



    @classmethod
    def validate(
        cls,
        filename: str,
        file_size: int,
        dataframe: pd.DataFrame
    ):
        """
        Run all dataset validations.
        """

        cls.validate_extension(
            filename
        )

        cls.validate_file_size(
            file_size
        )

        cls.validate_dataframe(
            dataframe
        )

        return True