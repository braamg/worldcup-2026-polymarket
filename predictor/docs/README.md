# Design notes

Working notes written during the project, kept as they were apart from removing private details.

- `QUANT_THEORY_AND_DESIGN.md`: the full reasoning behind the model, the simulator and the sizing.
- `PHASE_DECISIONS.md`: decisions and surprises, phase by phase.
- `MODEL_REVIEW.md`: a critical review of the model and the changes it led to.
- `DERIVATIVES_FRAMEWORK_SUMMARY.md`: covariance-aware portfolio sizing (`src/portfolio/`).
- `LATENCY_ARB_README.md`: the latency-arbitrage monitor (`src/latency_arb/`), paper-traded only.

These notes mention files that are not in this public repository: the pipeline and diagnostic
scripts (`scripts/`), the internal project brief (`CLAUDE.md`), the order-placement modules
(`src/betting/execution.py`, `src/latency_arb/execution.py`) and the Transfermarkt scraper.
Dollar amounts in the portfolio notes are illustrative backtest examples, not real positions.
