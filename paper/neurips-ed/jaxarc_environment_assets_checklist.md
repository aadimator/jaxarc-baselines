# JaxARC environment section assets checklist

This checklist is for the `paper/neurips-ed/sections/jaxarc_environment.tex` section.

## First principle

Do **not** let this section read like a throughput advertisement.

Use visuals here to explain:
- what an ARC task looks like as an RL environment;
- how JaxARC preserves support/query semantics;
- why hidden-query scoring is correct;
- how action/observation interfaces are varied without changing task semantics.

Save most performance plots for the results or practicality section.

## Must-have assets

- [ ] **Figure: one ARC task, two evaluation views**
  - **Purpose:** Make the support/query split concrete in one glance.
  - **Show:**
    - 2--3 support pairs with visible outputs;
    - 1 query input;
    - a masked or blank query target in the agent-facing view;
    - the true query target in a clearly separated evaluator-only box.
  - **Key message:** JaxARC exposes support targets to the agent during adaptation, but keeps query targets hidden while still storing them for correct scoring.
  - **Why it belongs in a figure:** This is spatial and semantic. A paragraph alone makes people work too hard.
  - **Implementation hint:** The existing SVG helpers in `JaxARC/src/jaxarc/utils/visualization/display.py`, especially `draw_task_pair_svg` and `draw_parsed_task_data_svg`, look like the right starting point.
  - **Current reusable stand-in:** `Figures/viz_complete_task.pdf` is already good enough to use in the draft right now. It clearly shows support pairs on the left and a held-out query with hidden output on the right. A future revision could improve it by adding an explicit evaluator-only true-target pane.

- [x] **Figure: JaxARC execution and scoring pipeline**
  - **Purpose:** Explain what is actually inside the compiled environment and what is scored outside the policy observation.
  - **Show:**
    - dataset/parser -> fixed-shape task buffer;
    - task sample -> pair sample;
    - support mode vs query mode;
    - environment state (`working_grid`, masks, pair index, task index);
    - wrapper layer for action and observation interfaces;
    - submit step -> hidden-target scoring -> metrics.
  - **Key message:** JaxARC is not just a fast simulator; it is a benchmark substrate that keeps evaluation-correct information flow.
  - **Why it belongs in a figure:** This is an information-flow story.
  - **Current draft:** `Figures/jaxarc_benchmark_flow.svg` with export target `Figures/jaxarc_benchmark_flow.pdf`.
  - **Design decision:** Keep hidden-query exact scoring in a distinct evaluator-only lane fed by the terminal submitted grid and the stored hidden targets in the task buffer. This avoids implying that the query-time masked `target_grid` shown to the agent is itself the evaluator target.

- [ ] **Table: design choices and why they matter**
  - **Purpose:** Tie implementation decisions directly to scientific consequences.
  - **Recommended columns:** Design choice | What JaxARC does | Why it matters for the benchmark.
  - **Recommended rows:**
    - fixed-shape padded task buffers;
    - masks for valid grid regions;
    - stored true query targets;
    - masked query target in agent observation;
    - base mask action API;
    - wrapper-based action/observation interfaces;
    - evaluation reward with no similarity shaping;
    - episode metrics beyond reward.
  - **Why it belongs in a table:** This is a comparison of discrete design decisions. A table will be tighter and more reviewer-friendly than a long paragraph.

- [ ] **Figure or compact schematic: action and observation interfaces**
  - **Purpose:** Show that the simulator stays fixed while the agent-facing interface changes.
  - **Show:**
    - base action = operation + selection mask;
    - point wrapper -> mask;
    - bounding-box wrapper -> mask;
    - flatten wrapper -> single discrete index;
    - base observation = working grid;
    - optional channels = input, answer, clipboard, contextual demos.
  - **Key message:** JaxARC supports controlled interface ablations without changing the underlying task semantics.
  - **If space is tight:** Convert this into a small table instead of a full figure.

## Strongly recommended assets

- [ ] **Figure: support episode vs query episode snapshots from the same task**
  - **Purpose:** Ground the paper in concrete examples rather than only abstractions.
  - **Show:**
    - support episode start and end;
    - query episode start and final submitted grid;
    - true hidden query output shown only in a separate evaluator-only pane.
  - **Key message:** Support fitting and hidden-query solving are related but distinct.
  - **Best use:** This can be a small figure in the main text or an appendix figure referenced from the section.

- [ ] **Figure: variable ARC data to fixed JAX buffers**
  - **Purpose:** Make the JAX-specific implementation choice easy to understand.
  - **Show:**
    - several differently sized raw ARC grids;
    - padding to a shared canvas;
    - boolean masks marking valid cells;
    - stacked tensors for batch sampling.
  - **Key message:** Static shapes are what make compilation and vectorization practical without losing true grid boundaries.
  - **Best use:** Only include this if space allows or if reviewers are likely to ask how variable-size ARC tasks fit accelerator-native execution.

## Optional appendix assets

- [ ] **Table: dataset and subset coverage**
  - **Purpose:** Document the benchmark surface without bloating the environment section.
  - **Recommended columns:** Dataset | Role in paper | Example subsets | Why included.
  - **Rows:** MiniARC, ConceptARC, ARC-AGI-1, and any canonical subset families used in the benchmark.

- [ ] **Appendix figure: wrapper catalogue**
  - **Purpose:** Document available action and observation wrappers without spending too much main-paper space on software plumbing.
  - **Note:** Keep this out of the main section unless reviewer feedback suggests confusion.

## Imported JMLR assets and suitability

These assets have been copied into `paper/neurips-ed/Figures/` for local use and reference.

### Strong candidate for current main-text use

- [x] `Figures/viz_complete_task.pdf` — **use now**. This is the best existing match for the environment section because it makes the support/query split legible immediately.

### Useful, but better as appendix or inset material

- [x] `Figures/viz_task_pair.pdf` — clean single input/output example, but too small in scope to carry the section by itself.
- [x] `Figures/viz_rl_step.pdf` — helpful for debugging and action semantics, but too policy-step-specific for the main environment section.
- [x] `Figures/viz_single_grid.pdf` — fine as a utility visualization reference, not a strong paper figure.
- [x] `Figures/arc_example_952a094c.pdf` — good qualitative task example with a clear train/test split; can support appendix or task-gallery material.
- [x] `Figures/arc_example_de1cd16c.pdf` — useful as a richer qualitative example, especially for varying grid sizes and multiple training pairs.
- [x] `Figures/arc_example_eb5a1d5d.pdf` — useful as another qualitative example, especially for nested geometric structure.

### Copied for reference, but not recommended as the hero figure in the main section

- [x] `Figures/jaxarc_system_architecture.pdf` — informative, but too implementation-heavy and software-diagram-like for the current NeurIPS environment section. Better as appendix/reference unless simplified.

## What not to lead with in this section

- [ ] Do **not** open this section with a speed plot.
- [ ] Do **not** make the first visual a systems diagram full of code modules and config names.
- [ ] Do **not** spend a main-text figure on Hydra or launcher plumbing.
- [ ] Do **not** use a wrapper catalogue as the hero visual.

## Suggested ordering in the paper

1. **Task anatomy / support-query figure**
2. **Pipeline figure or design-choice table**
3. **Optional interface schematic**
4. Keep practicality plots for the results section

## Chosen architecture figure for the current draft

- [x] `Figures/jaxarc_benchmark_flow.svg` — new main-text architecture / flow figure.
  - **Recommended role:** the architecture companion to `Figures/viz_complete_task.pdf`.
  - **Why this version is better:** it emphasizes information flow, support/query branching, wrapper locality, and evaluator-only hidden-target scoring instead of presenting JaxARC as a generic software dependency diagram.
  - **Recommended placement:** after the `System organization` or `Task representation and hidden-query semantics` discussion, depending on final page pressure.
  - **Caption seed:** a benchmark-flow caption is drafted in `Figures/jaxarc_benchmark_flow_notes.md`.

## Best asset to make first

If only one new asset gets made soon, make this one first:

- [ ] **Support/query task anatomy figure with evaluator-only true query target**

At the moment, `Figures/viz_complete_task.pdf` is the right temporary stand-in. It is already strong enough for the current draft, but a custom version with an explicit evaluator-only true-target pane would tell the story even better.

That is the single clearest way to communicate what is novel and important about JaxARC for this paper.
