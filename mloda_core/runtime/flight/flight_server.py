from typing import Any, Dict, Set
from uuid import uuid4
from pyarrow import flight
import pyarrow as pa
import socket
import os

import logging

logger = logging.getLogger(__name__)


def create_location(host: str = "0.0.0.0") -> str:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"grpc://{host}:{port}"


class FlightServer(flight.FlightServerBase):  # type: ignore
    def __init__(self, location: Any = create_location()) -> None:
        self.tables: Dict[str, Any] = {}  # Dictionary to store tables
        self.location = location

        # the default with 4 MB leads to crashes of the test suite
        os.environ["GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH"] = str(1024 * 1024 * 10)  # 10 MB
        os.environ["GRPC_ARG_SEND_MESSAGE_LENGTH"] = str(1024 * 1024 * 10)  # 10 MB

        super().__init__(
            self.location,
        )

        self.uuid = uuid4()

    def do_put(self, context: Any, descriptor: Any, reader: Any, writer: Any) -> None:
        path = descriptor.path[0]  # Path descriptor to identify the dataset
        table = reader.read_all()  # Reading the data sent by the client

        # if path in self.tables:
        #    raise ValueError("Path exists already. Overwritting currently not supported")

        self.tables[path] = table  # Storing the data in memory

    @staticmethod
    def upload_table(location: str, table: Any, table_key: str) -> None:
        with flight.FlightClient(location) as client:
            descriptor = flight.FlightDescriptor.for_path(table_key)
            writer, _ = client.do_put(descriptor, table.schema)

            try:
                writer.write_table(table)
                writer.close()
            finally:
                try:
                    writer.close()
                except Exception:  # nosec
                    pass

    def do_get(self, context: Any, ticket: Any) -> Any:
        if len(self.tables.keys()) == 0:
            raise ValueError("Try to get an empty apache flight.")

        key = ticket.ticket
        if key in self.tables:
            return flight.RecordBatchStream(self.tables[key])
        raise KeyError(f"Table with key {key} not found")

    @staticmethod
    def download_table(location: str, table_key: Any) -> Any:
        with flight.FlightClient(location) as client:
            ticket = flight.Ticket(table_key.encode("utf-8"))
            reader = client.do_get(ticket)
            table = reader.read_all()
        client.close()
        return table

    def do_action(self, context: Any, action: Any) -> None:
        if action.type == "drop_table":
            path = action.body.to_pybytes()
            self.drop_table(path)
        else:
            raise ValueError("Unsupported action")

    def drop_table(self, table_key: Any) -> None:
        if table_key in self.tables:
            del self.tables[table_key]

    @staticmethod
    def drop_tables(location: str, table_key: Set[str]) -> None:
        with flight.FlightClient(location) as client:
            for key in table_key:
                action = flight.Action("drop_table", key.encode("utf-8"))
                for _ in client.do_action(action):
                    pass

    @staticmethod
    def sent_shutdown_signal(location: str) -> None:
        with flight.FlightClient(location) as client:
            action = flight.Action("shutdown", b"")
            for _ in client.do_action(action):
                pass

    def list_flights(self, context: Any, criteria: Any) -> Any:
        """List the available datasets (keys of the tables)."""
        for key in self.tables.keys():
            descriptor = flight.FlightDescriptor.for_path(key)
            endpoint = flight.FlightEndpoint(key, [self.location])
            # Dummy schema and empty metadata
            schema = pa.schema([])
            yield flight.FlightInfo(schema, descriptor, [endpoint], -1, -1)

    @staticmethod
    def list_flight_infos(location: str) -> Set[str]:
        with flight.FlightClient(location) as client:
            flight_infos = {x.descriptor.path[0] for x in client.list_flights()}
        client.close()
        return flight_infos
