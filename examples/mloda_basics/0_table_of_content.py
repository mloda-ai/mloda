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
    ## Table of content

    1) Intro to the core interfaces of mloda
       - Introduction to mloda's core interfaces and practical demonstration of its capabilities.
    2) What makes mloda unique?
       - Explanation of mloda's unique aspects, including data reusability and support for diverse technologies.
    3) What is data, Feature, FeatureSets and FeatureGroups in mloda
       - Detailed overview of key objects in mloda and their relationships.
    4) Roles in mloda
       - Exploration of the roles of Providers, Data Users, and Stewards within mloda.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisite Setup for This Example
    ------------------------------------
    This example assumes that you have cloned the repository and followed the
    [Local Development Setup](https://github.com/mloda-ai/mloda/blob/main/CONTRIBUTING.md#local-development-setup):

    ```bash
    uv sync --all-extras
    source .venv/bin/activate
    ```

    This installs all dependencies, including the docs extras needed for these notebooks.
    """)
    return


if __name__ == "__main__":
    app.run()
