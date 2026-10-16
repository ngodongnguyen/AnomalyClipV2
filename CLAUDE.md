# AnomalyCLIP — Claude Code Instructions

Claude Code and Codex are peer coding and research agents.

Either agent may:
- inspect the repository
- research related methods
- design experiments
- modify implementation code
- debug
- run tests
- analyze experimental results
- review work produced by the other agent

Do not assume that work created by Codex needs to be redone.
The current source code and Git working tree are the ground truth for implementation state.

## Shared project context

Before substantial research or implementation work, read the shared project context:

@research/PROJECT_STATE.md
@research/REPOSITORY_GUIDANCE.md
@research/WORKFLOW.md

When relevant to the task, also read:

@research/HYPOTHESES.md
@research/EXPERIMENTS.md
@research/DECISIONS.md

These files are the canonical shared research state used by both Claude Code and Codex.

## Before implementation

Before making substantial implementation changes:

1. Inspect the current source code.
2. Inspect `git status`.
3. Inspect relevant `git diff` output when there are existing changes.
4. Read the relevant hypothesis, experiment, and decision records.
5. Verify what is actually implemented before assuming the state described in previous notes.

Do not overwrite or redo valid work simply because it was produced by Codex.

## Research state maintenance

Keep the canonical research files current after meaningful work.

Update only the files that are relevant:

### `research/PROJECT_STATE.md`

Update when there is a meaningful change to:
- the current research objective
- the current method or modification
- confirmed observations
- the main unresolved problem
- the current research direction
- important conclusions that affect future work

Keep this file concise and focused on the current state of the project.

### `research/HYPOTHESES.md`

Update when:
- a new hypothesis is created
- an existing hypothesis is refined
- evidence supports or weakens a hypothesis
- a hypothesis is rejected
- evidence remains inconclusive

Do not mark a hypothesis as supported without experimental evidence.

### `research/EXPERIMENTS.md`

Update when an experiment is:
- proposed for actual execution
- implemented
- started
- completed
- failed
- modified substantially
- associated with meaningful results

Record actual configurations, datasets, metrics, status, and conclusions when available.

Never invent experiment results.

### `research/DECISIONS.md`

Update when an important methodological, experimental, or architectural decision is made that future agents should know about.

Record:
- what was decided
- why
- relevant alternatives when useful
- related hypothesis or experiment IDs when applicable

## Raw research notes

Save exploratory Claude analysis in:

`research/notes/claude/`

Raw notes may contain:
- literature analysis
- debugging observations
- speculative ideas
- alternative explanations
- preliminary reasoning
- reviewer-style criticism

Raw notes are not automatically accepted conclusions.

Promote information from raw notes into the canonical research files only when it becomes relevant and sufficiently supported.

## Evidence discipline

Clearly distinguish between:
- confirmed implementation facts
- experimental observations
- hypotheses
- interpretations
- speculative ideas

Do not present a hypothesis as a confirmed mechanism without evidence.

Do not hide negative results.

Do not optimize conclusions around only the datasets that improved.

## Shared-agent workflow

Claude Code and Codex may continue work started by one another.

When continuing previous work:
- inspect the actual current implementation
- read the shared project state
- inspect existing changes
- preserve valid prior work
- avoid duplicating completed analysis or implementation

Before finishing substantial research or implementation work, check whether any of the following should be updated:

- `research/PROJECT_STATE.md`
- `research/HYPOTHESES.md`
- `research/EXPERIMENTS.md`
- `research/DECISIONS.md`

Do not update canonical files for trivial edits, formatting changes, or speculative thoughts that have not matured into useful project knowledge.

## Git safety

Never commit or push unless the user explicitly requests it in the current turn.

Do not discard or overwrite existing uncommitted changes without understanding them first.

If the repository already contains modifications, inspect them before editing related files.