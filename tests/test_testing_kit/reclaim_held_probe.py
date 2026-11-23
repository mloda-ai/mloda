"""Check a held class's complete lifetime in a fresh interpreter."""

import gc
import json

from mloda._testing.source import reclaim
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup


def collect() -> dict[str, str]:
    mark = reclaim.mark()
    held = reclaim.track(type("ReclaimHeldFG", (FeatureGroup,), {}))
    try:
        reported = reclaim.reclaim_since(mark)
    finally:
        del held
        gc.collect()
        gc.collect()
    assert reported == ["ReclaimHeldFG"]
    assert "ReclaimHeldFG" not in {cls.__name__ for cls in get_all_subclasses(FeatureGroup)}
    return {"reported": reported[0], "registration": "released"}


if __name__ == "__main__":
    print(json.dumps(collect()))
