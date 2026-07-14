from collections.abc import Callable
from typing import Any
from uuid import uuid4

import os

import logging

from mloda.core.optional_dependency import require

logger = logging.getLogger(__name__)

_FLIGHT_REASON = "Flight-based (multiprocessing/distributed) data transport"


def _flight() -> Any:
    return require("pyarrow.flight", _FLIGHT_REASON)


def create_location(host: str = "127.0.0.1") -> str:
    return f"grpc://{host}:0"


def _require_pyarrow_flight() -> None:
    _flight()


_SERVING_CLASSES: dict[type, type] = {}


def _serving_class(base: type) -> type:
    """FlightServerBase is only needed to serve: mixing it in here keeps the module import pyarrow-free."""
    cached = _SERVING_CLASSES.get(base)
    if cached is None:
        flight = _flight()
        cached = type(f"_Serving{base.__name__}", (base, flight.FlightServerBase), {})
        _SERVING_CLASSES[base] = cached
    return cached


class FlightServer:
    # Provided by flight.FlightServerBase, which is mixed in on instantiation.
    port: int
    serve: Callable[[], None]
    shutdown: Callable[[], None]

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if not issubclass(cls, _flight().FlightServerBase):
            cls = _serving_class(cls)
        return super().__new__(cls)

    def __init__(self, location: Any = create_location()) -> None:
        self.tables: dict[str, Any] = {}  # Dictionary to store tables
        self.location = location

        # the default with 4 MB leads to crashes of the test suite
        os.environ["GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH"] = str(1024 * 1024 * 10)  # 10 MB
        os.environ["GRPC_ARG_SEND_MESSAGE_LENGTH"] = str(1024 * 1024 * 10)  # 10 MB

        super().__init__(  # type: ignore[call-arg]
            self.location,
        )
        location_prefix = str(self.location).rsplit(":", 1)[0]
        self.location = f"{location_prefix}:{self.port}"

        self.uuid = uuid4()

    def do_put(self, context: Any, descriptor: Any, reader: Any, writer: Any) -> None:
        path = descriptor.path[0]  # Path descriptor to identify the dataset
        table = reader.read_all()  # Reading the data sent by the client

        # if path in self.tables:
        #    raise ValueError("Path exists already. Overwritting currently not supported")

        self.tables[path] = table  # Storing the data in memory

    @staticmethod
    def upload_table(location: str, table: Any, table_key: str) -> None:
        flight = _flight()
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
            return _flight().RecordBatchStream(self.tables[key])
        raise KeyError(f"Table with key {key} not found")

    @staticmethod
    def download_table(location: str, table_key: Any) -> Any:
        flight = _flight()
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
    def drop_tables(location: str, table_key: set[str]) -> None:
        flight = _flight()
        with flight.FlightClient(location) as client:
            for key in table_key:
                action = flight.Action("drop_table", key.encode("utf-8"))
                for _ in client.do_action(action):
                    pass

    @staticmethod
    def sent_shutdown_signal(location: str) -> None:
        flight = _flight()
        with flight.FlightClient(location) as client:
            action = flight.Action("shutdown", b"")
            for _ in client.do_action(action):
                pass

    def list_flights(self, context: Any, criteria: Any) -> Any:
        """List the available datasets (keys of the tables)."""
        pa = require("pyarrow", _FLIGHT_REASON)
        flight = _flight()

        for key in self.tables.keys():
            descriptor = flight.FlightDescriptor.for_path(key)
            endpoint = flight.FlightEndpoint(key, [self.location])
            # Dummy schema and empty metadata
            schema = pa.schema([])
            yield flight.FlightInfo(schema, descriptor, [endpoint], -1, -1)

    @staticmethod
    def list_flight_infos(location: str) -> set[str]:
        flight = _flight()
        with flight.FlightClient(location) as client:
            flight_infos = {x.descriptor.path[0] for x in client.list_flights()}
        client.close()
        return flight_infos
