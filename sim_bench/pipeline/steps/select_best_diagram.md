# SelectBestStep overview

This diagram shows the flow in plain language so you can read it top to bottom.

```mermaid
flowchart TD
    A[Start: pick a cluster] --> B{Use face subclusters?}
    B -->|yes| C[Iterate face subclusters]
    B -->|no| D[Use scene clusters]

    C --> E{Cluster has faces?}
    D --> E{Cluster has faces?}

    E -->|yes| F[Score with face composite<br/>eyes + pose + smile + AVA]
    E -->|no| G[Score with non-face mix<br/>0.7*AVA + 0.3*IQA]

    F --> H[Sort by score (best first)]
    G --> H[Sort by score (best first)]

    H --> I{Top scores within tiebreaker range?}
    I -->|yes| J[Use Siamese to reorder top 2]
    I -->|no| K[Keep order]

    J --> L[Always take #1]
    K --> L[Always take #1]

    L --> M{Take #2?}
    M -->|no| N[Done with this cluster]
    M -->|yes| O[Check: score >= min]
    O --> P[Check: gap <= max]
    P --> Q[Check: not near-duplicate]
    Q --> N[Done with this cluster]

    N --> R{More clusters?}
    R -->|yes| A
    R -->|no| S[Output: selected_images]
```

Legend (short):
- Face composite = weighted eyes/pose/smile/AVA.
- Non-face mix = 0.7*AVA + 0.3*IQA.
- Tiebreaker uses Siamese only if top scores are very close.
- #2 is kept only if it passes all three checks.
