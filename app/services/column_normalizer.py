import re

import pandas as pd


class ColumnNormalizer:
    """
    Normalizes dataset column names into SQL-friendly names.
    """

    @staticmethod
    def normalize_column_name(
        column_name: str
    ) -> str:
        """
        Convert a single column name into snake_case.

        Examples:
            Customer Name -> customer_name
            Order-Date -> order_date
            Total Revenue ($) -> total_revenue
        """

        # Convert to string
        name = str(column_name)

        # Remove leading/trailing whitespace
        name = name.strip()

        # Convert to lowercase
        name = name.lower()

        # Replace any sequence of non-alphanumeric
        # characters with an underscore
        name = re.sub(
            r"[^a-z0-9]+",
            "_",
            name
        )

        # Remove leading/trailing underscores
        name = name.strip("_")

        # Prevent empty column names
        if not name:
            name = "column"

        return name

    @classmethod
    def normalize_columns(
        cls,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize all DataFrame column names.

        Returns a copy of the DataFrame.
        """

        dataframe = dataframe.copy()

        original_columns = list(
            dataframe.columns
        )

        normalized_columns = [
            cls.normalize_column_name(
                column
            )
            for column in original_columns
        ]

        # Handle duplicate names created
        # during normalization.
        normalized_columns = (
            cls.make_unique(
                normalized_columns
            )
        )

        dataframe.columns = normalized_columns

        return dataframe

    @staticmethod
    def make_unique(
        column_names: list[str]
    ) -> list[str]:
        """
        Make column names unique.

        Example:

            ["name", "name", "name"]

        becomes:

            ["name", "name_2", "name_3"]
        """

        counts = {}

        unique_names = []

        for name in column_names:

            if name not in counts:

                counts[name] = 1

                unique_names.append(
                    name
                )

            else:

                counts[name] += 1

                unique_names.append(
                    f"{name}_{counts[name]}"
                )

        return unique_names