# System Patterns

## System Architecture

mloda's architecture revolves around two key components:

*   **Plugins:** Feature Groups, Compute Frameworks, and Extenders
*   **Core:** Core Engine

The Core Engine handles dependencies between features and computations by coordinating linking, joining, filtering, and ordering operations.

## Key Technical Decisions

*   Focus on defining transformations rather than static states.
*   Automatic plugin selection.
*   Decoupling feature definitions from specific computation technologies.

## Design Patterns

*   Plugin-based architecture
*   Dependency injection
*   Data transformation pipelines

## Component Relationships

*   **Feature Groups** define feature dependencies and calculations.
*   **Compute Frameworks** define the technology stack used to execute feature transformations and computations.
*   **Extenders** automate metadata extraction processes.
*   The **Core Engine** orchestrates the interactions between these components.
