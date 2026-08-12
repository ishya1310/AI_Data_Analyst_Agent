import re

import pandas as pd


class DataTypeInference:
    """
    Detect and convert semantic data types in a DataFrame.

    Supported semantic types:

        numeric
        categorical
        date
        boolean
    """


    IDENTIFIER_KEYWORDS = {
        "id",
        "code",
        "zip",
        "zipcode",
        "postal",
        "phone",
        "account",
        "number"
    }

    @classmethod
    def looks_like_identifier(
        cls,
        column_name: str
    ) -> bool:
        """
        Determine whether a column name looks like an
        identifier.

        Examples:

            customer_id       -> True
            product_code      -> True
            postal_code       -> True
            revenue           -> False
        """

        name = str(column_name).lower()

        # Split snake_case / spaces / hyphens
        tokens = re.split(
            r"[_\s\-]+",
            name
        )

        return any(
            token in cls.IDENTIFIER_KEYWORDS
            for token in tokens
        )



    @staticmethod
    def detect_boolean(
        series: pd.Series
    ) -> bool:
        """
        Detect boolean-like string columns.
        """

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        non_null = series.dropna()

        if non_null.empty:
            return False

        normalized = (
            non_null
            .astype(str)
            .str.strip()
            .str.lower()
        )

        allowed_values = {
            "true",
            "false",
            "yes",
            "no"
        }

        return (
            normalized.isin(
                allowed_values
            ).all()
        )



    @staticmethod
    def detect_date(
        series: pd.Series
    ) -> bool:
        """
        Detect whether a string/object column contains dates.

        At least 90% of non-null values must be successfully
        parsed as dates.
        """

        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        non_null = series.dropna()

        if non_null.empty:
            return False

        parsed = pd.to_datetime(
            non_null,
            errors="coerce",
            format="mixed"
        )

        valid_ratio = (
            parsed.notna().sum()
            / len(non_null)
        )

        return valid_ratio >= 0.90



    @classmethod
    def detect_numeric(
        cls,
        series: pd.Series,
        column_name: str
    ) -> bool:
        """
        Detect numeric-looking string columns.

        Identifier-like columns are deliberately protected
        from numeric conversion.
        """

        if cls.looks_like_identifier(
            column_name
        ):
            return False

        if pd.api.types.is_numeric_dtype(series):
            return True

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False

        non_null = series.dropna()

        if non_null.empty:
            return False

        numeric_values = pd.to_numeric(
            non_null,
            errors="coerce"
        )

        valid_ratio = (
            numeric_values.notna().sum()
            / len(non_null)
        )

        return valid_ratio >= 0.95



    @classmethod
    def infer_column_type(
        cls,
        series: pd.Series,
        column_name: str
    ) -> str:
        """
        Infer the semantic type of a column.
        """

        # Boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if cls.detect_boolean(series):
            return "boolean"

        # Already numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Date
        if cls.detect_date(series):
            return "date"

        # Numeric strings
        if cls.detect_numeric(
            series,
            column_name
        ):
            return "numeric"

        # Everything else
        return "categorical"



    @classmethod
    def convert_column(
        cls,
        series: pd.Series,
        column_name: str
    ) -> pd.Series:
        """
        Convert a column to its inferred data type.
        """

        semantic_type = cls.infer_column_type(
            series,
            column_name
        )



        if semantic_type == "boolean":

            normalized = (
                series
                .astype("string")
                .str.strip()
                .str.lower()
            )

            mapping = {
                "true": True,
                "false": False,
                "yes": True,
                "no": False
            }

            return normalized.map(
                mapping
            ).astype("boolean")


        if semantic_type == "date":

            return pd.to_datetime(
                series,
                errors="coerce",
                format="mixed"
            )



        if semantic_type == "numeric":

            return pd.to_numeric(
                series,
                errors="coerce"
            )



        return series



    @classmethod
    def convert_dataframe(
        cls,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Infer and convert all columns.

        Returns a new DataFrame.
        """

        dataframe = dataframe.copy()

        for column in dataframe.columns:

            dataframe[column] = (
                cls.convert_column(
                    dataframe[column],
                    column
                )
            )

        return dataframe



    @classmethod
    def generate_type_report(
        cls,
        dataframe: pd.DataFrame
    ) -> list:
        """
        Generate semantic type information for every column.
        """

        report = []

        for column in dataframe.columns:

            semantic_type = (
                cls.infer_column_type(
                    dataframe[column],
                    column
                )
            )

            report.append({
                "column": column,
                "pandas_dtype": str(
                    dataframe[column].dtype
                ),
                "semantic_type": semantic_type
            })

        return report