## Purpose

This repository contains an active NeurIPS 2026 Evaluations & Datasets paper effort for JaxARC. When working on paper-writing tasks, treat this file as the local operating manual.

The current paper is **not** a generic JaxARC systems paper and **not** a solver breakthrough paper. The honest ED-track framing is:

> JaxARC is a JAX-native ARC environment for reinforcement learning and task-specific adaptation studies, and the benchmark reveals a persistent gap between support-side fitting and hidden-query transfer.

If a proposed paragraph, figure, or claim pulls the story away from that framing, stop and reconsider.

## Workspace map for the paper

Primary paper workspace:

- `paper/neurips-ed/main.tex` — working NeurIPS manuscript entry point
- `paper/neurips-ed/sections/` — one file per main section
- `paper/neurips-ed/NeurIPSED.bib` — Zotero-synced Better BibTeX export; this is the bibliography source of truth
- `paper/neurips-ed/zotero_add_queue.md` — running checklist of papers that need to be added to Zotero before they can be cited cleanly

Reference-only assets in the same folder:

- `paper/neurips-ed/neurips_2026.sty` — style file
- `paper/neurips-ed/neurips_2026.tex` — stock template shell; keep as reference
- `paper/neurips-ed/checklist.tex` — NeurIPS checklist
- `paper/neurips-ed/neurips_author_submission_instructions.md` — submission/formatting notes

Useful source material elsewhere in the repo:

- `planning/neurips_ed_paper_writing_guide.md` — main writing guide; use this as the paper-framing source of truth
- `planning/neurips_ed_experiment_execution_guide.md` — experiment package map and what each result is for
- `planning/neurips_ed_final_implementation_plan.md` — broader narrative and structure guidance
- `results/neurips_ed_final_program/STATUS_SUMMARY.md` — current high-level state of completed experiment packages
- `results/neurips_ed_final_practicality/` — authoritative practicality results
- `results/neurips_ed_final_support_query_gap/` — authoritative support-to-query gap results
- `results/neurips_ed_final_failure_taxonomy/` — authoritative failure-taxonomy package
- `scratchpad/jmlr-paper/main.tex` — legacy manuscript source material only; reuse carefully, never as the target structure

## Non-negotiable paper framing

Keep these points fixed unless the user explicitly changes direction:

1. This is an **ED-track evaluation paper**.
2. The paper should open like a **JAX-native RL environment/testbed paper**, then move into the **protocol + benchmark + empirical analysis**.
3. The main empirical story is **support improves, hidden-query success does not** in the completed canonical runs.
4. Negative results are part of the contribution, not an embarrassment to hide.
5. JaxARC speed matters because it makes the evaluation practical, not because raw throughput alone is the scientific result.

## Claims policy
Safe claims:

- JaxARC is a useful RL environment for ARC because it exposes sparse rewards, combinatorial actions, and diverse task families in a unified interactive setting.
- JaxARC makes large-scale ARC adaptation studies practical.
- The benchmark cleanly separates support metrics from hidden-query metrics.
- Completed PPO-style baselines show measurable support-side adaptation.
- Completed canonical runs do not show hidden-query exact success.
- The benchmark reveals a persistent support-to-query gap.

Do **not** claim any of the following unless the user provides new audited evidence:

- that the method solves ARC;
- that hidden-query generalization has emerged;
- that a positive-control result belongs in the canonical benchmark story;
- that query success is "latent" or "present but hidden by decoding";
- that JaxARC is important mainly because it is faster.

## Writing workflow for each section

When asked to draft or revise a paper section, follow this sequence:

1. Read the relevant section file in `paper/neurips-ed/sections/`.
2. Re-read the matching guidance in `planning/neurips_ed_paper_writing_guide.md`.
3. Gather evidence from result roots or manuscript sources before drafting.
4. Draft only what the current evidence supports.
5. Check every factual statement:
   - result-backed claims must point to a concrete result root or summary file;
   - literature-backed claims must point to an actual paper.
6. Do a humanizing pass before considering the section done.
7. If a paper is needed, check `paper/neurips-ed/NeurIPSED.bib` first.
8. If the paper is missing from the export, add it to `paper/neurips-ed/zotero_add_queue.md` instead of creating a manual BibTeX entry.

If working across multiple sections, prefer this order:

1. protocol
2. results
3. figure/table captions
4. discussion
5. introduction
6. related work
7. conclusion
8. abstract
9. title

## Anti-hallucination rules

- Never invent numbers, runs, hardware details, or reviewer-facing interpretations.
- Never cite a paper that has not been checked in Zotero, the local bibliography, or another trusted source.
- Never describe a figure as if it exists unless the file exists.
- Never turn a planning idea into a result claim.
- When in doubt, write a narrower sentence.

Use this test: **Could the user ask "where exactly does that come from?" and get a precise answer immediately?** If not, rewrite or research first.

## How to use Zotero and Consensus

For background, related work, and benchmark-positioning sections, use external research tools deliberately.

### Zotero-first rule

Prefer Zotero first when the likely paper is already in the user's library.

Recommended pattern:

1. Search Zotero semantically for the topic or exact paper.
2. If a likely match exists, use the Zotero metadata and attached full text when available.
3. Before citing, confirm that the corresponding entry exists in `paper/neurips-ed/NeurIPSED.bib`.
4. If it exists there, cite it using its Better BibTeX key.
5. If it does not exist there, add a checkbox item to `paper/neurips-ed/zotero_add_queue.md` with enough detail to identify the paper later.
6. Do not hand-write a BibTeX entry in the repo for that paper.

Good Zotero targets for this paper include ARC, ARCLE, ConceptARC, ARC Prize, JAX-native RL environments, and adaptation/meta-RL references.

### Consensus usage

Use Consensus when:

- you need broader literature coverage beyond the user's Zotero library;
- you need to verify a general claim about a research area;
- you need papers for ED-track framing, negative results, or evaluation methodology.

When using Consensus:

- cite findings to specific returned papers, not vague "prior work" language;
- prefer direct, narrow claims over broad surveys;
- do not add filters unless the user explicitly asks for them.
- if Consensus surfaces a useful paper that is not already in Zotero, add it to `paper/neurips-ed/zotero_add_queue.md` and leave the manuscript text in a non-committal state until the Zotero export contains the entry.

### Citation discipline

- Do not cite from memory.
- Do not cite based only on a title when the claim depends on details.
- Do not over-cite weakly related work just to make the section look busy.
- If a citation is uncertain, leave a comment in the section file instead of bluffing.
- Do not manually curate or edit a separate `references.bib` file for this paper.
- Treat `paper/neurips-ed/NeurIPSED.bib` as the only bibliography file used by the manuscript.

### Missing-paper queue workflow

When a needed citation is missing from Zotero or from the exported `.bib`:

1. Add an unchecked item to `paper/neurips-ed/zotero_add_queue.md`.
2. Include: title, authors if known, year if known, where it came from (Zotero search miss, Consensus, etc.), and what section needs it.
3. Do not fabricate a BibTeX entry.
4. Do not cite the paper in LaTeX until the user has added it to Zotero and it appears in `NeurIPSED.bib`.
5. Once the user has added it and the export updates, the checkbox can be checked off or removed.

## Humanizer pass

Every completed section should get one explicit humanizing pass.

Look for and remove:

- inflated significance language;
- vague attributions like "prior work shows" without a citation;
- AI-sounding filler such as "underscores," "highlights," "landscape," or "pivotal";
- repetitive sentence rhythm;
- formulaic concluding sentences;
- overuse of em dashes;
- headings in title case.

Aim for writing that is:

- direct;
- concrete;
- calm;
- technically precise;
- comfortable stating negative findings plainly.

## Section-specific guidance

### Introduction

- Frame the paper first as a JAX-native RL environment paper in the style of prior benchmark/testbed papers, then connect that environment to the ED contribution.
- Establish why ARC is a meaningful RL environment: sequential editing, sparse rewards, large actions, and task diversity.
- Preview the support-to-query gap early, but not before the environment motivation is clear.
- Keep software plumbing out of the introduction.

### Related work

- Organize around ARC, ARCLE/RL for ARC, test-time adaptation, JAX-native benchmarks, and ED-style evaluation methodology.
- Use Zotero and Consensus here more than in other sections.
- Keep it selective; this is not a generic ARC survey.

### JaxARC as an RL environment

- Follow the framing used in prior JAX RL environment papers: environment motivation, API/design choices, datasets, actions, observations, rewards, and compatibility with existing learning code.
- Include enough technical detail that a reviewer can understand how the environment works without going to the appendix.
- Introduce the support/query evaluation mechanism as one capability of the environment, not as the only reason it exists.

### Protocol

- Define support/demo vs hidden query/test clearly.
- State metric definitions precisely.
- Make clear that submitted-query rate is not success.
- Make clear that hidden-query scoring uses held-out targets.

### Results

- Lead with practicality, support fitting, hidden-query transfer, and the gap.
- Use the stable result roots as authoritative.
- Do not write around missing figures; write from actual artifacts.

### Discussion

- Interpret the negative result as evidence about evaluation.
- State limitations plainly.
- Keep future work grounded in observed failure modes.

## Double-blind and submission guardrails

- Keep the draft anonymized unless the user explicitly asks for a non-anonymous version.
- Do not add acknowledgments to the anonymized draft.
- Refer to prior work, including the authors' own, in the third person where needed.
- Remember that NeurIPS ED uses a 9-page main-paper limit excluding references/checklist/appendix material.

## Practical editing rules

- Keep new manuscript content in `paper/neurips-ed/`.
- Treat `paper/neurips-ed/neurips_2026.tex` as a preserved stock reference file.
- Use `paper/neurips-ed/main.tex` as the actual working manuscript.
- Use `paper/neurips-ed/NeurIPSED.bib` as the manuscript bibliography source.
- Maintain `paper/neurips-ed/zotero_add_queue.md` whenever a needed paper is not yet available in the export.
- Put reusable comments and TODO prompts inside section files when that helps future sessions.
- Preserve existing result files; do not rewrite experiment summaries casually.

## Handoff checklist

Before ending a paper-writing session, leave the workspace in a clean state:

1. section text saved in the appropriate `sections/*.tex` file;
2. any cited papers confirmed to exist in `paper/neurips-ed/NeurIPSED.bib`;
3. any missing papers recorded in `paper/neurips-ed/zotero_add_queue.md`;
4. claims checked against either result roots or cited papers;
5. a humanizer-style cleanup pass completed;
6. any unresolved research gaps left as comments rather than hidden assumptions.

If you follow this file, the next session should inherit a paper workflow instead of a mystery novel.