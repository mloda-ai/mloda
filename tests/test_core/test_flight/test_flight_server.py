import threading
from typing import Any
import unittest
from unittest.mock import MagicMock
import pyarrow as pa
import pyarrow.flight as flight
from mloda_core.runtime.flight.flight_server import FlightServer


class TestFlightServerUnit(unittest.TestCase):
    server: Any = None
    context: Any = None
    table: Any = None
    table_key: Any = None
    descriptor: Any = None
    ticket: Any = None
    location: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment and instantiate server"""
        cls.server = FlightServer()
        cls.context = MagicMock()
        cls.table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        cls.table_key = "test_table"
        cls.descriptor = flight.FlightDescriptor.for_path(cls.table_key)
        cls.ticket = flight.Ticket(cls.table_key.encode("utf-8"))
        cls.location = cls.server.location

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def test_do_put_and_get(self) -> None:
        """Test storing a table in the server."""
        reader = MagicMock()
        reader.read_all.return_value = self.table
        writer = MagicMock()

        # Execute do_put method
        self.server.do_put(self.context, self.descriptor, reader, writer)

        self.assertIn(self.table_key.encode("utf-8"), self.server.tables)
        self.assertEqual(self.server.tables[self.table_key.encode("utf-8")], self.table)

        result_stream = self.server.do_get(self.context, self.ticket)
        self.assertIsInstance(result_stream, flight.RecordBatchStream)

    def test_do_get_not_found(self) -> None:
        """Test error handling when table is not found."""
        with self.assertRaises(ValueError):
            self.server.do_get(self.context, self.ticket)


class TestFlightServerIntegration(unittest.TestCase):
    server: Any = None
    table: Any = None
    table_key: Any = None
    location: Any = None
    server_thread: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = FlightServer()
        cls.server_thread = threading.Thread(target=cls.server.serve)
        cls.server_thread.start()
        cls.location = cls.server.location

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server_thread.join()

    def test_server_operations(self) -> None:
        """Test operations involving actual server-client interaction"""

        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        table_key = "test_table"

        FlightServer.upload_table(self.location, table, table_key)
        result_table = FlightServer.download_table(self.location, table_key)
        self.assertEqual(table, result_table)
        assert result_table == table

        FlightServer.drop_tables(self.location, {table_key})
        with self.assertRaises(ValueError):
            FlightServer.download_table(self.location, table_key)
