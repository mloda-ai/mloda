## 1. Install mloda via pip
You can install mloda using either pip or by cloning the repository from Git.
```bash
pip install mloda
```

## 2. Install mloda via Git
Alternatively, you can clone the repository and install from source:
```bash
git clone https://github.com/mloda-ai/mloda.git
cd mloda
pip install .
```

## 3. Verify Installation
To verify the installation, run:
``` python
from importlib import metadata
print(metadata.version("mloda"))
```

## 4. Quick Start

Try this 30-second example:

``` python
import mloda
from mloda.user import PluginLoader
PluginLoader.all()

result = mloda.run_all(
    features=["customer_id", "income", "income__sum_aggr"],
    compute_frameworks=["PandasDataFrame"],
    api_data={"SampleData": {
        "customer_id": ["C001", "C002", "C003"],
        "income": [50000, 75000, 90000]
    }}
)
```

Next: [API Request](https://mloda-ai.github.io/mloda/chapter1/api-request/) | [Feature Groups](https://mloda-ai.github.io/mloda/chapter1/feature-groups/)