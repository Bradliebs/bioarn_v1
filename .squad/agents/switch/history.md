# Switch (Tester) — History

## Session 2026-06-24T23:50:23Z

**Mission:** Run full test suite and assess test health

- Spawned as part of initial team assessment
- Running comprehensive test health evaluation


## Session 2026-06-25T02:08:39Z

**Mission:** Post-crash test suite assessment

- 411/411 runnable tests PASS (427 collected, zero failures)
- 16 text generation tests too slow for CPU CI, need @pytest.mark.slow
- PyYAML dependency missing from venv despite being in pyproject.toml (blocker)
- Gradio missing from optional deps, test_demo.py fails at collection
- Recommended: regenerate venv, add gradio to [demo], mark slow tests, move benchmarks to experiments/


## Session 2026-06-25T08:42:45Z

**Mission:** Writing integration tests for hierarchy+ensemble

**Context:** Team-wide orchestration sprint to validate module integration and infrastructure.

**Assigned Critical Tasks (P0):**
- Regenerate venv with `pip install -e ".[dev]"` (restore PyYAML)
- Add gradio to optional deps in pyproject.toml
- Verify test_demo.py passes

**Assigned P1 Tasks:**
- Create 15+ integration tests for hierarchy module
- Create 15+ integration tests for ensemble module (experts, voting, boosting)
- Validate end-to-end flows with both modules active
- Mark slow text generation tests with @pytest.mark.slow (16 tests)
- Ensure all 411 baseline tests remain passing

**Session Dependencies:**
- Blocking P0 must be cleared before team can proceed with module integration
- Results from Neo, Trinity, Morpheus will be validated by new tests


## Session 2026-06-26T10:30:40Z

**Mission:** Recording documentation refresh and regression sweep

- Updated the README with architecture, metrics, quick start, CI badge, and module map
- Regression validation reached 439 passing tests
- Committed `bb5f032`
- Team context: Tank's demo fixes and Neo/Trinity benchmark gains are now reflected in contributor-facing docs


## Session 2026-06-27T11:00:00Z

**Mission:** Recording benchmark and continual-learning findings around Sprint E

- Logged the severe continual-learning forgetting findings and the later Sprint D retest showing only marginal CIFAR gains with worse MNIST retention
- Logged the paper-positioning decision that emphasizes efficiency, robustness, and biological plausibility over raw top-1 accuracy
- Team context: the Sprint E combined benchmark on `experiments/sprint_e_benchmark.py` remains in progress per manifest


## Session 2026-06-27T19:30:00Z

**Mission:** Recording completed Sprint E combined benchmark

- Benchmarked standard and convolutional CCC variants on CIFAR-10 and Split-CIFAR-10
- Standard variants tied at 13.0% CIFAR-10 accuracy, while `conv_locked` and `conv_all` cut mean forgetting to 20.7% by retaining most performance on the final task
- Precision rose with pool entropy instead of decreasing with familiarity
- Team context: Morpheus folded these findings into the paper draft and Neo formalized the follow-on benchmark suite
