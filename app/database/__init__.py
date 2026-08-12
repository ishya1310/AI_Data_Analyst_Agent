from pathlib import Path

import duckdb


class DuckDBManager:
    """
    Handles all DuckDB operations for uploaded datasets.
    """

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self, read_only: bool = False):
        """
        Create a DuckDB connection.
        """

        return duckdb.connect(
            str(self.database_path),
            read_only=read_only
        )

    # ========================================================
    # CREATE TABLE FROM DATAFRAME
    # ========================================================

    def create_table_from_dataframe(
        self,
        dataframe,
        table_name: str
    ):
        """
        Create a DuckDB table from a Pandas DataFrame.
        """

        connection = self.connect()

        try:

            connection.register(
                "uploaded_dataframe",
                dataframe
            )

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

    # ========================================================
    # GET TABLE SCHEMA
    # ========================================================

    def get_table_schema(
        self,
        table_name: str
    ) -> list:

        connection = self.connect(
            read_only=True
        )

        try:

            result = connection.execute(
                f'''
                DESCRIBE "{table_name}"
                '''
            ).fetchall()

            schema = []

            for row in result:

                schema.append({
                    "column": row[0],
                    "type": row[1],
                    "null": row[2],
                    "key": row[3],
                    "default": row[4],
                    "extra": row[5]
                })

            return schema

        finally:

            connection.close()

    # ========================================================
    # PREVIEW TABLE
    # ========================================================

    def preview_table(
        self,
        table_name: str,
        rows: int = 10
    ):

        connection = self.connect(
            read_only=True
        )

        try:

            result = connection.execute(
                f'''
                SELECT *
                FROM "{table_name}"
                LIMIT ?
                ''',
                [rows]
            ).fetchdf()

            return result

        finally:

            connection.close()

    # ========================================================
    # COUNT ROWS
    # ========================================================

    def count_rows(
        self,
        table_name: str
    ) -> int:

        connection = self.connect(
            read_only=True
        )

        try:

            result = connection.execute(
                f'''
                SELECT COUNT(*)
                FROM "{table_name}"
                '''
            ).fetchone()

            return int(result[0])

        finally:

            connection.close()

    # ========================================================
    # LIST TABLES
    # ========================================================

    def list_tables(self) -> list:

        connection = self.connect(
            read_only=True
        )

        try:

            result = connection.execute(
                "SHOW TABLES"
            ).fetchall()

            return [
                row[0]
                for row in result
            ]

        finally:

            connection.close()

    # ========================================================
    # EXECUTE QUERY
    # ========================================================

    def execute_query(
        self,
        query: str
    ):

        connection = self.connect(
            read_only=True
        )

        try:

            result = connection.execute(
                query
            ).fetchdf()

            return result

        finally:

            connection.close()