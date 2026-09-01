"""All chess prompt templates and the dynamic prompt registry."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chess_evolve.pipeline import PipelineConfig

# Dynamic prompt registry -- Opus can add new entries at runtime
PROMPT_REGISTRY: dict[str, dict[str, str]] = {}

ANALYSIS_PROMPTS = {
    "concise": (
        "Analyze the chess position. "
        "Give exactly 3 bullet points: material count, biggest threat, best opportunity."
    ),
    "detailed": (
        "Analyze the chess position thoroughly: material balance, "
        "king safety (castled? pawn shield?), piece activity (well/poorly placed), "
        "pawn structure weaknesses, and immediate threats."
    ),
    "threat_focused": (
        "Analyze the chess position focusing on threats. "
        "What is the opponent threatening? What can you threaten? "
        "List every undefended piece on both sides."
    ),
}

TACTICAL_PROMPTS = {
    "broad": (
        "Scan the position for tactical opportunities: "
        "forks, pins, skewers, hanging pieces, checkmate threats. "
        "For each tactic found, name the move (in UCI notation) that exploits it."
    ),
    "mate_focused": (
        "Look for forcing sequences: checks, captures, and threats that limit "
        "the opponent's options. Prioritize checkmate patterns -- back rank mates, "
        "smothered mates, discovered checks. Name the moves in UCI notation."
    ),
    "material": (
        "Focus on material: can you capture an undefended piece? "
        "Can you win a trade (e.g. knight for rook)? Are any of your pieces "
        "hanging? List every capture that gains material with the UCI move."
    ),
}

POSITIONAL_PROMPTS = {
    "classical": (
        "Evaluate the position: pawn structure, central control, piece coordination, "
        "open files, and king safety. Suggest the 2-3 best positional moves in UCI notation."
    ),
    "dynamic": (
        "Focus on piece activity over structure. Which pieces are passive? "
        "How can you activate them? Suggest moves (in UCI) that improve your worst piece."
    ),
    "prophylactic": (
        "Think about what the opponent wants to do. What is their best plan? "
        "Suggest moves (in UCI) that prevent their plan while improving your position."
    ),
}

OPENING_HINTS = {
    "principled": (
        "\n\nGAME PHASE: OPENING. Follow opening principles strictly: "
        "1) Control the center with pawns (e4, d4 as White; e5, d5, c5, Nf6 as Black). "
        "2) Develop knights before bishops (Nf3, Nc3, Nf6, Nc6). "
        "3) Castle early (within first 8-10 moves). "
        "4) Do NOT play a3, a4, h3, h4, or move edge pawns in the opening. "
        "5) Do NOT move the queen before developing minor pieces. "
        "6) Do NOT move the same piece twice unless capturing or avoiding capture. "
        "Strong White openings: 1.e4 or 1.d4. Strong Black replies: vs 1.e4 play e5 or c5; "
        "vs 1.d4 play d5 or Nf6."
    ),
    "aggressive": (
        "\n\nGAME PHASE: OPENING. Play for initiative: "
        "1) Occupy the center with e4+d4 as White; challenge it as Black. "
        "2) Develop pieces toward the center and the opponent's king. "
        "3) Look for early tactical shots -- gambits, piece sacrifices for development lead. "
        "4) Castle quickly, then attack. Do NOT play passively. "
        "5) Avoid a3/h3 unless preventing a specific pin or threat."
    ),
    "solid": (
        "\n\nGAME PHASE: OPENING. Play solid and safe: "
        "1) Control the center, but do NOT overextend. "
        "2) Develop all minor pieces before creating threats. "
        "3) Castle as early as possible -- kingside preferred. "
        "4) Do NOT push pawns beyond the 4th rank in the opening. "
        "5) Avoid early queen moves and pawn weaknesses. Build a fortress first."
    ),
    "theory": (
        "\n\nGAME PHASE: OPENING. Play known strong openings from theory: "
        "As White: play the Italian Game (1.e4 e5 2.Nf3 Nc6 3.Bc4), the Ruy Lopez "
        "(1.e4 e5 2.Nf3 Nc6 3.Bb5), or the Queen's Gambit (1.d4 d5 2.c4). "
        "As Black vs 1.e4: play the Sicilian Defense (1...c5) or the French Defense (1...e6). "
        "As Black vs 1.d4: play the Queen's Gambit Declined (1...d5 2.c4 e6) or the King's Indian "
        "(1...Nf6 2.c4 g6 3...Bg7). "
        "Follow the main lines you know. Castle kingside. Complete development before attacking."
    ),
}

MIDDLEGAME_HINTS = {
    "safety_first": (
        "\n\nGAME PHASE: MIDDLEGAME. Before choosing a move: "
        "1) What did the opponent's last move THREATEN? Respond to it. "
        "2) Does your candidate move leave any piece undefended? If yes, pick a different move. "
        "3) Check for tactics: forks, pins, skewers, discovered attacks -- for BOTH sides. "
        "4) If no tactics exist, improve your worst-placed piece or create a threat. "
        "5) Keep your king safe -- do NOT open lines toward your own king. "
        "6) Look for pawn breaks to open the position when your pieces are active. "
        "7) Put rooks on open files and behind passed pawns."
    ),
    "attacking": (
        "\n\nGAME PHASE: MIDDLEGAME. Play for the attack: "
        "1) Look for forcing moves first: checks, captures, threats -- in that order. "
        "2) Target the opponent's king -- pile pieces toward it. "
        "3) Open files and diagonals pointing at the enemy king. "
        "4) Sacrifice material if it exposes the king or creates a mating attack. "
        "5) Only defend if you are under immediate threat of losing material."
    ),
    "positional": (
        "\n\nGAME PHASE: MIDDLEGAME. Play positionally: "
        "1) Improve your worst piece every move. "
        "2) Control key squares -- especially outposts "
        "(squares your opponent can't attack with pawns). "
        "3) Create and exploit pawn structure weaknesses (isolated, doubled, backward pawns). "
        "4) Trade bad pieces for good ones (e.g. your bad bishop for their good knight). "
        "5) Only attack when your position is ready -- accumulate small advantages first."
    ),
    "theory": (
        "\n\nGAME PHASE: MIDDLEGAME. Apply classical middlegame concepts: "
        "1) Nimzowitsch's blockade -- place a piece (ideally a knight) on the square in front "
        "of the opponent's passed or isolated pawn. "
        "2) Minority attack -- advance your queenside pawns (a4-b5) to create weaknesses in "
        "the opponent's pawn structure. "
        "3) Greek gift sacrifice (Bxh7+) -- consider it when: bishop on d3, queen can reach h5, "
        "knight can reach g5, and the opponent has castled kingside with a standard pawn shield. "
        "4) Alekhine's gun -- stack queen behind rooks on an open file. "
        "5) Tarrasch rule -- rooks belong on the 7th rank or behind passed pawns."
    ),
}

ENDGAME_HINTS = {
    "technical": (
        "\n\nGAME PHASE: ENDGAME. Critical rules: "
        "1) Activate your king -- move it toward the center and toward passed pawns. "
        "2) Passed pawns must be pushed. A passed pawn on the 6th/7th rank wins games. "
        "3) If ahead in material, trade pieces (not pawns) to simplify. "
        "4) If behind in material, avoid trades and create counterplay. "
        "5) Rook endgames: rooks belong BEHIND passed pawns (yours or the opponent's). "
        "6) Do NOT make pointless moves -- every tempo matters in the endgame."
    ),
    "aggressive": (
        "\n\nGAME PHASE: ENDGAME. Push for the win: "
        "1) Advance your king aggressively -- it's a fighting piece now. "
        "2) Create passed pawns on both sides of the board to stretch the opponent. "
        "3) Push passed pawns relentlessly -- promotion is the goal. "
        "4) Use zugzwang -- force the opponent into positions where any move loses."
    ),
    "defensive": (
        "\n\nGAME PHASE: ENDGAME. Hold the position: "
        "1) Keep your king in front of the opponent's passed pawns -- blockade them. "
        "2) Maintain the opposition when kings face each other. "
        "3) Create a fortress if possible -- a structure the opponent cannot break through. "
        "4) Trade pawns (not pieces) to reduce the opponent's winning chances. "
        "5) Activate your rook -- passive defense loses. Counterattack on the other side."
    ),
    "theory": (
        "\n\nGAME PHASE: ENDGAME. Apply known endgame theory: "
        "1) Lucena position -- if you have rook + pawn vs rook and your pawn is on the 7th rank "
        "with your king on the promotion square, build a bridge: rook to the 4th rank, "
        "then interpose on the 8th to block checks. "
        "2) Philidor position -- if defending rook vs rook+pawn, keep your rook on the 6th rank "
        "while the enemy pawn hasn't reached the 6th; once it does, go to the back rank and check. "
        "3) Opposite-color bishops -- these tend to be drawn even a pawn down; if behind, "
        "trade into this endgame. "
        "4) Knight vs bishop -- knights are better in closed positions, bishops in open ones. "
        "5) Rule of the square -- a king can catch a passed pawn if it's inside the square "
        "formed by the pawn and the promotion square."
    ),
}

SINGLE_DEEP_PROMPT = (
    "You are a chess expert. Analyze this position deeply. Consider: "
    "1) Tactical patterns (forks, pins, skewers, hanging pieces, mates) "
    "2) Positional factors (pawn structure, piece activity, king safety, central control) "
    "3) For your top 3 candidate moves, think ahead 2-3 moves. "
    "Conclude with your chosen move."
)

ENUMERATE_PROMPT = (
    "You are a chess expert. List the 5 best candidate moves in this position. "
    "For each, give a one-line evaluation. Format each line as: "
    "MOVE: <uci> SCORE: <1-10> REASON: <brief>. "
    "Then state which move you recommend."
)

BLUNDER_CHECK_PROMPTS = {
    "strict": (
        "A chess engine selected the move shown below. Check for blunders: "
        "1) Does this move hang a piece (leave it undefended and attacked)? "
        "2) Does it walk into a fork, pin, or skewer? "
        "3) Does it allow the opponent a forced checkmate? "
        "4) Does it lose material in a trade sequence? "
        "5) Does it weaken king safety? "
        "Be suspicious -- assume the move is a blunder unless you can prove it's safe. "
        "If the move is a blunder, suggest a better move from the legal moves list. "
        "If the move is fine, confirm it."
    ),
    "standard": (
        "A chess engine selected the move shown below. Check for blunders: "
        "1) Does this move hang a piece (leave it undefended and attacked)? "
        "2) Does it walk into a fork, pin, or skewer? "
        "3) Does it allow the opponent a forced checkmate? "
        "If the move is a blunder, suggest a better move from the legal moves list. "
        "If the move is fine, confirm it."
    ),
    "lenient": (
        "A chess engine selected the move shown below. Only flag it as a blunder "
        "if it immediately loses material (hangs a piece) or allows forced checkmate. "
        "Positional inaccuracies are fine. "
        "If the move is a blunder, suggest a better move from the legal moves list. "
        "If the move is fine, confirm it."
    ),
}

CRITIQUE_PROMPTS = {
    "devils_advocate": (
        "You are a chess critic. A move has been selected for this position. "
        "Your job is to argue AGAINST this move. Find the strongest objection: "
        "What does the opponent do after this move? Does it ignore a threat? "
        "Does it miss a better alternative? Does it create a long-term weakness? "
        "Be specific -- name the opponent's best reply and why it's dangerous. "
        "If you genuinely cannot find a problem, say 'no objection'."
    ),
    "sanity_check": (
        "You are a chess advisor doing a final sanity check. A move has been selected. "
        "Quickly verify: 1) Is the opponent's best reply dangerous? "
        "2) Are we missing an obvious better move? "
        "If the move looks reasonable, confirm it. Only object if something is clearly wrong."
    ),
}

GAME_PLAN_PROMPT = (
    "You are a chess strategist. Given the position, the move just played, "
    "and the previous plan (if any), write a 1-sentence strategic plan for the next few moves. "
    "Examples: 'Attack the weak f7 pawn with bishop and queen.' "
    "'Trade pieces to simplify into a won endgame.' "
    "'Develop remaining pieces and castle kingside.' "
    "Output ONLY the plan sentence, nothing else."
)

OPPONENT_MODEL_PROMPT = (
    "You are predicting the opponent's intentions. Given the position, "
    "what is the opponent most likely threatening? What is their plan? "
    "List the top 2 threats in order of danger. Be specific about which "
    "pieces and squares are involved."
)

_SELECTOR_FORMAT = (
    "\n\nCRITICAL: Your entire response must be exactly one UCI move "
    "(4 or 5 characters like e2e4 or e7e8q). No other text. No explanation. "
    "No punctuation. Just the move."
)

SELECTOR_PROMPTS = {
    "balanced": (
        "You are given tactical and positional analysis of a chess position. "
        "Weigh both equally. Choose the move that best balances tactics and position."
    ),
    "aggressive": (
        "You are given tactical and positional analysis of a chess position. "
        "Strongly prefer moves that create threats, win material, or attack the king. "
        "Only play defensively if you are losing material otherwise."
    ),
    "defensive": (
        "You are given tactical and positional analysis of a chess position. "
        "Strongly prefer safe, solid moves. Never sacrifice material. "
        "Prioritize king safety and avoiding blunders over attacking."
    ),
    "calculating": (
        "You are given tactical and positional analysis of a chess position. "
        "For your top 3 candidate moves, think ahead: your move, opponent's best reply, "
        "your follow-up. Pick the move with the best position after 3 moves."
    ),
}

PROMPT_TABLES: dict[str, dict[str, str]] = {
    "tactical_style": TACTICAL_PROMPTS,
    "positional_style": POSITIONAL_PROMPTS,
    "analysis_style": ANALYSIS_PROMPTS,
    "selector_style": SELECTOR_PROMPTS,
    "verify_style": BLUNDER_CHECK_PROMPTS,
    "critique_style": CRITIQUE_PROMPTS,
    "opening_hint": OPENING_HINTS,
    "middlegame_hint": MIDDLEGAME_HINTS,
    "endgame_hint": ENDGAME_HINTS,
}


def detect_phase(fen: str) -> str:
    """Detect game phase from piece count in FEN."""
    piece_chars = sum(1 for c in fen.split()[0] if c.isalpha() and c.lower() != 'k')
    if piece_chars > 26:
        return "opening"
    elif piece_chars > 10:
        return "middlegame"
    return "endgame"


def _get_phase_hint(cfg: PipelineConfig, phase: str) -> str:
    hints = {"opening": OPENING_HINTS, "middlegame": MIDDLEGAME_HINTS, "endgame": ENDGAME_HINTS}
    key = {"opening": cfg.opening_hint, "middlegame": cfg.middlegame_hint,
           "endgame": cfg.endgame_hint}[phase]
    table = hints[phase]
    if "+" in key:
        return " ".join(table.get(k, k) for k in key.split("+"))
    return table.get(key, key)


def _get_prompt(knob_name: str, value: str) -> str:
    """Look up a prompt by knob name and value. Falls back to dynamic registry."""
    table = PROMPT_TABLES.get(knob_name, {})
    if value in table:
        return table[value]
    if knob_name in PROMPT_REGISTRY and value in PROMPT_REGISTRY[knob_name]:
        return PROMPT_REGISTRY[knob_name][value]
    if "+" in value:
        reg = PROMPT_REGISTRY.get(knob_name, {})
        parts = [table.get(v, reg.get(v, "")) for v in value.split("+")]
        combined = " ".join(p for p in parts if p)
        if combined:
            return combined
    if value and value not in table:
        return value
    return table.get(list(table.keys())[0], "") if table else ""


def _register_prompt(knob_name: str, value: str, prompt_text: str) -> None:
    """Register a new prompt variant created by Opus."""
    from chess_evolve.config import LIVE_DIR
    from chess_evolve.pipeline import KNOB_SPACE

    PROMPT_REGISTRY.setdefault(knob_name, {})[value] = prompt_text
    for _, (name, choices) in enumerate(KNOB_SPACE):
        if name == knob_name and value not in choices:
            choices.append(value)
            break
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_DIR / "recording.jsonl", "a") as f:
        f.write(json.dumps({
            "t": time.monotonic(), "type": "prompt_invented",
            "knob": knob_name, "value": value, "prompt": prompt_text,
        }) + "\n")
