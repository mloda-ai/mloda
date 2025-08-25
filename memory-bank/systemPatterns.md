# System Patterns

## Architecture Overview

```mermaid
graph TB
    subgraph Core
        CE[Core Engine]
    end
    
    subgraph Plugins
        FG[Feature Groups]
        CF[Compute Frameworks]
        EX[Extenders]
    end
    
    FG --> |Dependencies| CE
    CF --> |Execution| CE
    EX --> |Metadata| CE
    CE --> |Orchestrates| Output[Transformed Features]
    
    style CE fill:#f9f,stroke:#333,stroke-width:3px
    style Output fill:#dfd,stroke:#333,stroke-width:2px
```

## Key Design Decisions

- **Transformations over static states**: Define how data changes
- **Plugin-based architecture**: Automatic plugin selection
- **Decoupled execution**: Features independent of compute technology

## Component Roles

```mermaid
flowchart LR
    FG[Feature Groups] -->|Define| Trans[Transformations]
    CF[Compute Frameworks] -->|Execute| Trans
    EX[Extenders] -->|Extract| Meta[Metadata]
    CE[Core Engine] -->|Orchestrate| All[All Components]
