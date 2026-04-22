# JaxARC benchmark flow figure notes

- Primary asset: `Figures/jaxarc_benchmark_flow.svg`
- Intended export: `Figures/jaxarc_benchmark_flow.pdf`
- Intended use: main-text architecture / information-flow figure for the JaxARC environment section

## Figure design rule

Keep hidden-query exact scoring in a distinct benchmark-evaluator lane. The figure should not imply that the masked `target_grid` visible to the core query episode is itself the hidden target used for final exact-match evaluation.

## Suggested caption draft

JaxARC separates host-side task preparation, a compiled JAX environment core, and evaluator-only hidden-query scoring. Parsers and the registry normalize ARC-style JSON tasks into a fixed-shape task buffer with padded grids, masks, support examples, query inputs, and stored hidden query targets. During `reset`, the environment samples a task and pair from this buffer. Support episodes expose visible targets for adaptation, while query episodes expose only the query input and a masked target in the agent-visible state. The simulator itself remains fixed, while action and observation wrappers vary the agent-facing interface. Hidden-query metrics are computed in a separate benchmark-evaluator path by comparing the terminal submitted grid against stored hidden targets from the task buffer.
