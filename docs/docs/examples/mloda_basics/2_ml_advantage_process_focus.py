import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What makes mloda unique?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At first glance, data systems may appear straightforward: input data goes in and output data goes out.

    The inherent complexity, however, lies in the details:

    - ensuring data reusability
    - supporting a wide range of technologies across software and ML lifecycles
    - addressing the needs of different user groups

    mloda redefines traditional data management by shifting focus from the data itself to the dynamic processes surrounding it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data should be reuseable

    In most organizations, data pipelines are often built with a single-purpose focus. This tight coupling of data and data transformation within the same process limits reusability. For example:

    ```SQL
    select lower(data) from table into other_table;
    ```

    In this scenario, embedding data specifics into the process prevents sharing or adapting it for other tasks, leading to repeated rewrites and wasted effort.

    mloda addresses this issue by decoupling data and process definitions - the features. Instead of combining them in a single process, data and the features are treated as separate entities that come together only during calculation time. The following code snippet of mloda brings together data and features:

    ```python
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        ...
    ```

    In this model:

    - Data remains independent and can be sourced from various locations
    - Features represent the process definitions (transformations) applied to the data

    Benefits of Decoupling Data and Features:

    - Process Flexibility: Switching between different processing modes is made simple and intuitive, such as batch and stream processing could be done using the same features
    - Artifact and Validation Reusability: Input and output feature artifacts, along with their validations, can be repurposed across projects or teams
    - Production-Ready Functionality: Centralized features such as global logging, monitoring of data usage, and lineage tracking can be consistently applied across all pipelines

    This modular approach not only reduces redundancy but also creates a robust framework for scaling data transformations efficiently across diverse use cases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accommodating different technologies across software and ML lifecycles
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In current systems, data processes are tightly coupled with the technologies used at specific lifecycle stages—ETL systems for batch pipelines, streaming platforms for real-time applications, and ML frameworks for training. This rigid coupling makes it challenging to adapt to changing technologies or switch between processing modes.

    Metadata management further complicates the situation, it is kept fragmented across separate tools:

    | **Tool/Feature**                | **Purpose**                              |
    |----------------------------------|------------------------------------------|
    | **Feature Stores**               | Managing ML features                     |
    | **Data Catalogs**                | Organizing base data                     |
    | **Pipeline Orchestration Tools** | Managing workflows                       |
    | **Monitoring Systems**           | Tracking data quality and performance    |

    This siloed approach leads to redundancy, inefficiencies, and considerable complexity when maintaining metadata and integrating tools.
    mloda tackles these issues by decoupling data and processes, where processes are inherently reusable and adaptable, thereby reducing redundancy and ensuring lifecycle consistency.

    As shown in the previous notebook, each Feature can be calculated using its own technology!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Addressing the Needs of Diverse User Groups with mloda
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Traditional data systems are fragmented and often inefficient. mloda tackles these challenges for providers, users, and stewards:

    | User            | Problem                          | mloda Framework                               |
    |------------------|----------------------------------|-----------------------------------------------|
    | **Providers** | Limited freedom in tools      | Decouples processes, allowing tool flexibility. |
    |                  | Cumbersome APIs                 | Simplifies with standardized, user-friendly APIs. |
    |                  | Knowledge gaps for meta data    | Adds metadata tracking through whole lifecycle  |
    | **Data Users**   | Fragmented access               | Unified mloda simplifies navigation.             |
    |                  | Reusability gaps                | Central repository enables easy reuse.        |
    |                  | Customization constraints       | Offers configurable workflows and templates.  |
    | **Stewards**  | Siloed governance               | Centralizes governance for consistency.       |
    |                  | Ambiguous accountability        | Tracks ownership for transparency.            |
    |                  | Compliance risks                | Standardizes processes for compliance.        |

    mloda is designed to prioritize seamless interfaces between these roles, unifying workflows and aiming at simplified collaboration. Our current development efforts are focused on enhancing the experiences for providers and data users, with improvements for stewards actively underway.
    """)
    return


if __name__ == "__main__":
    app.run()
