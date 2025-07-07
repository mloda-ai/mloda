## Development Setup (for Contributors)

If you are contributing to mloda or working on its development, follow these steps to set up your environment, including using the provided dev container and running tests with tox.

By contributing, you agree that your contributions will be licensed under the [Apache License, Version 2.0](https://github.com/mloda-ai/mloda/blob/main/LICENSE.TXT).


#### 1. Clone the repository:
```bash
git clone https://github.com/mloda-ai/mloda.git
```

#### 2. Dev Container (Optional)
Prerequisite:
docker

-   Open the project in a dev container (in VS Code or a compatible tool).
-   Start developing: The dev container includes all necessary dependencies and tools to work on mloda.

If you don't want to or can't use docker, skip step 2.
Instead you can run it in the current machine with
```bash
pip install -r tests/requirements-test.txt && pip install tox
```

#### 3. Running Tests with Tox
We use tox to run tests in multiple environments.

This is to ensure that adjustments are working and integrated well.

Run main test suite:
```bash
tox
```

Run core test suite:
```bash
tox -e core
```

Run installed module test suite: 
```bash
tox -e installed
```

#### 4. Docs 
You can adjust the docs under the folder docs.
To view local changes, use:
```bash
mkdocs serve --config-file docs/mkdocs.yml
```

#### 5. Contribute

Want to contribute? You can:

-   Create an Issue: If you find a bug or have a feature request or have an idea, feel free to create an issue.
-   Submit a Pull Request (PR): If you'd like to contribute code, you can open a pull request with your changes.


