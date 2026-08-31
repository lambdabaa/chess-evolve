# chess-evolve

Teach an LLM to play chess via evolutionary prompt optimization, powered by [remote-factory](https://github.com/akashgit/remote-factory).

Each chess move runs a composed `Package` pipeline through factory's `WorkflowExecutor`:

```
Sequential(
    Parallel(analyst, tactician, positionalist),
    Loop(Sequential(selector, verifier), gate)
)
```

The outer loop evolves this pipeline using factory's MAP-Elites quality-diversity search:
- `KNOB_MUTATE` tunes verification style, iteration count, and critique approach
- `PROMPT_MUTATE` rewrites agent prompts via Opus (full replacement, not append)
- Contrastive reflection identifies which knobs and prompts drive performance
- Rank-weighted tournament selection biases toward stronger parents

## Factory APIs used

| Component | What it does |
|---|---|
| `Package`, `Sequential`, `Parallel`, `Loop` | Compose the chess pipeline from reusable subgraphs |
| `OptKnob` | Declare tunable parameters the outer loop can mutate |
| `WorkflowExecutor` | Execute the compiled DAG per chess move |
| `MAPElitesArchive` | Quality-diversity archive preserving diverse strategies |
| `OuterLoopReflector` | Contrastive analysis of winners vs losers |
| `apply_random_mutation` | Mutation operators (knob, prompt, param) |
| `CycleRecord` | Structured experiment exhaust for reflection |

## Quick start

```bash
# Install
uv sync

# Install Stockfish
brew install stockfish  # macOS
# or: apt install stockfish

# Run the evolution loop
chess-evolve run

# In another terminal, start the live UI
chess-evolve serve
# Open http://localhost:8422
```

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `CHESS_MODEL` | `opus` | Model for the `claude` CLI (move generation uses Vertex Haiku) |
| `CHESS_WORKSPACE` | `/tmp/chess-factory` | Working directory for game data |

## Architecture

```
chess_evolve/
  evolve.py    # Pipeline definition + evolution loop (~1900 lines)
  serve.py     # FastAPI server with SSE for live UI (~900 lines)
  cli.py       # CLI entry point
  static/
    index.html # Live dashboard (~1100 lines)
```

The chess-specific code handles:
- Board representation and move parsing (python-chess)
- Stockfish opponent (configurable ELO)
- Game evaluation (composite scoring: position quality + survival + wins)
- Phase detection (opening/middlegame/endgame) with phase-specific prompts

Everything else is factory: pipeline composition, execution, mutation, selection, reflection.
