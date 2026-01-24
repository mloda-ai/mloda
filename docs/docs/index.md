# mloda

## Open Data Access for AI Agents

> **Describe what you need, mloda delivers it.**

mloda provides **declarative data access** for AI agents, ML pipelines, and data teams. Instead of writing complex data retrieval code, users describe WHAT they need, mloda resolves HOW to get it through its plugin system.

[Get started with mloda here.](https://mloda-ai.github.io/mloda/chapter1/installation/)

mloda's plugin system **automatically selects the right plugins for each task**, enabling efficient querying and processing of complex features. [Learn more about the mloda API here.](https://mloda-ai.github.io/mloda/in_depth/mloda-api/) By defining feature dependencies, transformations, and metadata processes, mloda minimizes duplication and fosters reusability.

Plugins are **small, template-like structures** - easy to test, easy to debug, and AI-friendly. Let AI generate plugins for you, or share them across projects, teams, and organizations.

## Key Benefits

**AI & Agent Integration**

- declarative API - agents describe WHAT, not HOW
- JSON-based feature requests for LLM tool functions
- built-in lineage for traceability

**Data Processing**

- automated feature engineering and dependency resolution
- data cleaning and synthetic data generation

**Data Management**

- rich metadata including data lineage and usage tracking
- clear role separation: providers, users, and stewards

**Data Quality and Security**

- data quality definitions
- unit- and integration tests
- secure queries

**Scalability**

- switch compute framework without changing feature logic
- same plugins work from notebook to production

**Community Engagement by Design**

- shareable plugin ecosystem
- fostering community

## Core Components and Architecture

mloda addresses common challenges in data and feature engineering by leveraging two key components:

#### Plugins
  - Feature Groups: **Define feature dependencies**, such as creating a composite label based on features e.g. user activity, purchase history, and support interactions. Once defined, only the label needs to be requested, as dependencies are resolved automatically, simplifying processing. [Learn more here.](https://mloda-ai.github.io/mloda/chapter1/feature-groups/)

  - Compute Frameworks: Defines the **technology stack**, like Spark or Pandas, along with support for different storage engines such as Parquet, Delta Lake, or PostgreSQL, to execute feature transformations and computations, ensuring efficient processing at scale. [Learn more here.](https://mloda-ai.github.io/mloda/chapter1/compute-frameworks/)

  - Extenders: Automates **metadata extraction processes**, helping you enhance data governance, compliance, and traceability, such as analyzing how often features are used by models or analysts, or understanding where the data is coming from. [Learn more here.](https://mloda-ai.github.io/mloda/chapter1/extender/)

#### Core
  - Core Engine: **Handles dependencies between features and computations** by coordinating linking, joining, filtering, and ordering operations to ensure optimized data processing. For example, in customer segmentation, the core engine would link and filter different data sources, such as demographics, purchasing history, and online behavior, to create relevant features.

## Contributing to mloda

We welcome contributions from the community to help us improve and expand mloda. Whether you're interested in developing plugins or adding new features, your input is invaluable. [Learn more here.](https://mloda-ai.github.io/mloda/development/)


## Frequently Asked Questions (FAQ)

If you have additional questions about mloda visit our [FAQ](https://mloda-ai.github.io/mloda/faq) section, raise an [issue](https://github.com/mloda-ai/mloda/issues/) on our GitHub repository, or email us at [info@mloda.ai](mailto:info@mloda.ai).


## License

This project is licensed under the [Apache License, Version 2.0](https://github.com/mloda-ai/mloda/blob/main/LICENSE.TXT).
