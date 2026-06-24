# Work Routing

How to decide who handles what.

## Routing Table

| Work Type | Route To | Examples |
|-----------|----------|----------|
| Architecture & design | Neo | System design, module boundaries, research direction, cross-cutting decisions |
| Core neural modules | Tank | Spiking neurons, concept cells, GNW workspace, preprocessing, feature binding, hierarchy |
| Learning & memory | Morpheus | Training loops, memory systems, predictive coding, Hebbian learning, continual learning |
| Experiments & integration | Trinity | Benchmarks, experiment scripts, ensemble, language module, demos, config, multi-modal pipelines |
| Testing & QA | Switch | Write tests, edge cases, regression detection, coverage analysis, performance validation |
| Code review | Neo | Review PRs, check quality, architectural consistency |
| Scope & priorities | Neo | What to build next, trade-offs, decisions |
| Session logging | Scribe | Automatic — never needs routing |
| Work monitoring | Ralph | Backlog scanning, keep-alive, continuous work loop |
| RAI review | Rai | Content safety, bias checks, credential detection, ethical review |

## Domain Routing (by file path)

| Path Pattern | Route To |
|-------------|----------|
| `bioarn/core/` | Tank |
| `bioarn/workspace/` | Tank |
| `bioarn/preprocessing/` | Tank |
| `bioarn/hierarchy/` | Tank |
| `bioarn/multimodal/` | Tank |
| `bioarn/sensorimotor/` | Tank |
| `bioarn/training/` | Morpheus |
| `bioarn/memory/` | Morpheus |
| `bioarn/predictive/` | Morpheus |
| `bioarn/reward/` | Morpheus |
| `bioarn/persistence/` | Morpheus |
| `bioarn/ensemble/` | Trinity |
| `bioarn/language/` | Trinity |
| `bioarn/generation/` | Trinity |
| `bioarn/config.py` | Trinity |
| `bioarn/system.py` | Neo |
| `experiments/` | Trinity |
| `configs/` | Trinity |
| `demo/` | Trinity |
| `tests/` | Switch |
| `docs/` | Neo |

## Issue Routing

| Label | Action | Who |
|-------|--------|-----|
| `squad` | Triage: analyze issue, assign `squad:{member}` label | Neo |
| `squad:neo` | Architecture & design tasks | Neo |
| `squad:tank` | Core neural module tasks | Tank |
| `squad:morpheus` | Learning & memory tasks | Morpheus |
| `squad:trinity` | Integration & experiment tasks | Trinity |
| `squad:switch` | Testing tasks | Switch |

### How Issue Assignment Works

1. When a GitHub issue gets the `squad` label, **Neo** triages it — analyzing content, assigning the right `squad:{member}` label, and commenting with triage notes.
2. When a `squad:{member}` label is applied, that member picks up the issue in their next session.
3. Members can reassign by removing their label and adding another member's label.
4. The `squad` label is the "inbox" — untriaged issues waiting for Neo's review.

## Rules

1. **Eager by default** — spawn all agents who could usefully start work, including anticipatory downstream work.
2. **Scribe always runs** after substantial work, always as `mode: "background"`. Never blocks.
3. **Quick facts → coordinator answers directly.** Don't spawn an agent for "what port does the server run on?"
4. **When two agents could handle it**, pick the one whose domain is the primary concern.
5. **"Team, ..." → fan-out.** Spawn all relevant agents in parallel as `mode: "background"`.
6. **Anticipate downstream work.** If a feature is being built, spawn Switch to write test cases from requirements simultaneously.
7. **Issue-labeled work** — when a `squad:{member}` label is applied to an issue, route to that member. Neo handles all `squad` (base label) triage.
