# FAQ

## General questions

#### Who is mloda intended for? 

mloda is designed for data engineers, data scientists, and data practitioners who want to streamline feature engineering. It helps implement repeatable feature creation and apply best practices borrowed from fields like software engineering, data modeling, and testing.

#### What problems does mloda solve in data and feature engineering? 

mloda tackles redundant data work, complex feature dependencies, and inconsistent data management. It simplifies the process of creating reusable features, optimizes feature processing, and maintains high-quality data practices.

#### How does mloda differ from traditional data engineering frameworks? 

Unlike traditional tools, mloda focuses on defining transformations rather than managing static states. This approach allows you to share data transformations or pipeline steps without exposing sensitive data, as the true value lies in the state of the data—which remains protected.

## Getting Started

#### How do I install mloda? How do I get started with mloda?

Installation is straightforward via pip or by cloning the repository. [Installation guide and step-by-step guide](https://tomkaltofen.github.io/mloda/chapter1/installation/)

#### What are the prerequisites for using mloda? 

You'll need Python and a bit of time to understand the framework. mloda introduces a different way of thinking compared to standard approaches, but it builds on standard practices while bridging multiple disciplines—data engineering, software engineering, data science, machine learning, and business analysis. 

You'll learn a lot along the way!

#### How do I create my first feature group using mloda? 

Feature Groups define dependencies and calculations. To create your first Feature Group, check out this [basic example here](https://tomkaltofen.github.io/mloda/chapter1/feature-groups/).

## Features and Functionality

#### What are Feature Groups in mloda and why are they important? 

Feature Groups are inspired by the concept of feature stores, where they represent logical groupings of reusable features. With mloda, Feature Groups define how features are calculated rather than stored, enabling hierarchical relationships between features that are resolved automatically—simplifying both creation and use.

#### How does mloda handle feature dependencies? 

mloda automatically manages feature dependencies, ensuring transformations are handled efficiently without manual intervention.

#### What is the role of the Core Engine in mloda? 

The Core Engine orchestrates dependencies between Feature Groups, compute frameworks, and extenders. Dependencies can include data collections, links, joins, filters, or hierarchical feature relations. 

The Core Engine creates an execution plan based on these dependencies and then runs the plan to execute feature calculations. This approach has the side effect that users describe the solution rather than the programm the solution itself. It is in that regard closer to databases paradigm than database paradigm.

#### How does mloda ensure data governance and quality control? 

mloda uses multiple mechanisms for governance and quality control. The extender feature logs relevant technical details, and users can incorporate data quality checks directly into feature definitions. Unit and integration testing are also simple to set up, enhancing reliability.

#### How do plug-ins work in mloda, and can I develop my own plug-ins?

mloda automatically selects the appropriate plug-ins for a given feature. 

Users can develop custom plug-ins to extend the platform's capabilities, and contributions are always welcome—feedback and design critiques are also greatly appreciated!

## Technical and Integration

#### How do I integrate mloda into my existing data infrastructure? 

mloda is a straightforward Python module. You can install it and run it within your existing projects, or you can create a microservice that exposes the mloda API for broader integration.

## If you have other questions not answered

Please raise an [issue](https://github.com/TomKaltofen/mloda/issues/) on our GitHub repository or email us at [mloda.info@gmail.com](mailto:mloda.info@gmail.com). We value community feedback and will do our best to address your questions and suggestions promptly.



