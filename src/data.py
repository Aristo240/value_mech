"""Static data and dataset loaders.

NOTE on PVQ items: the items below are paraphrased placeholders in the spirit of
PVQ-RR (Schwartz 2012), exactly 3 per refined value, written so the pilot can run
end-to-end. They are NOT the official PVQ-RR and must be replaced with the
licensed Schwartz items before any publication. Each item is in first person and
is meant to be answered "How much is this person like you?" on a 6-point scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# Canonical order of the 19 refined Schwartz values around the circumplex.
# Order follows Schwartz (2012, "Refining the theory of basic individual values").
# Equally-spaced on the circle: 360/19 ≈ 18.947 degrees per value.
SCHWARTZ_19 = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: dependability",
    "Benevolence: caring",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
]

# Higher-order quadrant for each of the 19 values.
HIGHER_ORDER = {
    "Self-direction: thought": "Openness",
    "Self-direction: action": "Openness",
    "Stimulation": "Openness",
    "Hedonism": "Openness",  # boundary; Schwartz places it between Openness and Self-Enh
    "Achievement": "Self-Enhancement",
    "Power: dominance": "Self-Enhancement",
    "Power: resources": "Self-Enhancement",
    "Face": "Self-Enhancement",  # boundary; sometimes Conservation
    "Security: personal": "Conservation",
    "Security: societal": "Conservation",
    "Tradition": "Conservation",
    "Conformity: rules": "Conservation",
    "Conformity: interpersonal": "Conservation",
    "Humility": "Conservation",  # boundary; sometimes Self-Transcendence
    "Benevolence: dependability": "Self-Transcendence",
    "Benevolence: caring": "Self-Transcendence",
    "Universalism: concern": "Self-Transcendence",
    "Universalism: nature": "Self-Transcendence",
    "Universalism: tolerance": "Self-Transcendence",
}


def schwartz_angles_deg() -> dict[str, float]:
    """Theoretical equally-spaced angles for the 19 refined values, in degrees."""
    step = 360.0 / len(SCHWARTZ_19)
    return {v: i * step for i, v in enumerate(SCHWARTZ_19)}


def schwartz_angles_rad() -> dict[str, float]:
    return {k: math.radians(v) for k, v in schwartz_angles_deg().items()}


def angular_distance_deg(a: float, b: float) -> float:
    """Smallest angular distance between two angles in degrees, in [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# --- ValueEval24 loader ----------------------------------------------------

@dataclass
class ValueEvalSample:
    text_id: str
    sentence_id: int
    text: str
    # 19-vector of (attained - constrained), in {-1, 0, +1}, in SCHWARTZ_19 order.
    polarity: np.ndarray
    # 19-vector of presence (attained OR constrained), in {0, 1}, in SCHWARTZ_19 order.
    presence: np.ndarray


def _val_columns(value_name: str) -> tuple[str, str]:
    """Map canonical value name to its ValueEval24 attained/constrained columns."""
    return f"{value_name} attained", f"{value_name} constrained"


def load_valueeval(root: str | Path, split: str) -> list[ValueEvalSample]:
    """Load ValueEval24 sentences and labels for the given split.

    The split directory must contain sentences.tsv and labels.tsv. Sentences with
    no labels.tsv row (e.g., test split with hidden labels) are skipped.
    """
    split_dir = Path(root) / split
    sents = pd.read_csv(split_dir / "sentences.tsv", sep="\t", dtype={"Text-ID": str})
    if not (split_dir / "labels.tsv").exists():
        raise FileNotFoundError(f"No labels.tsv in {split_dir}")
    labs = pd.read_csv(split_dir / "labels.tsv", sep="\t", dtype={"Text-ID": str})

    merged = sents.merge(labs, on=["Text-ID", "Sentence-ID"], how="inner")
    samples: list[ValueEvalSample] = []
    for _, row in merged.iterrows():
        polarity = np.zeros(len(SCHWARTZ_19), dtype=np.float32)
        presence = np.zeros(len(SCHWARTZ_19), dtype=np.float32)
        for i, v in enumerate(SCHWARTZ_19):
            att_col, con_col = _val_columns(v)
            att = float(row[att_col])
            con = float(row[con_col])
            # 0.5 means "unclear polarity but value present" — count as presence only.
            present = 1.0 if (att > 0 or con > 0) else 0.0
            presence[i] = present
            if att > 0 and con == 0:
                polarity[i] = 1.0
            elif con > 0 and att == 0:
                polarity[i] = -1.0
            # else 0 (absent or unclear)
        samples.append(ValueEvalSample(
            text_id=str(row["Text-ID"]),
            sentence_id=int(row["Sentence-ID"]),
            text=str(row["Text"]),
            polarity=polarity,
            presence=presence,
        ))
    return samples


def positive_negative_for_value(
    samples: list[ValueEvalSample],
    value_index: int,
    n_per_class: int,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Return (positive_texts, negative_texts) for one value, balanced.

    Positive: the value is present (attained or constrained — we want value identity,
              not polarity).
    Negative: the value is absent AND no other value with the same higher-order
              quadrant is present (to reduce co-occurrence confounds within quadrant).
    """
    val_name = SCHWARTZ_19[value_index]
    quadrant = HIGHER_ORDER[val_name]
    same_quadrant_idx = [
        i for i, v in enumerate(SCHWARTZ_19) if HIGHER_ORDER[v] == quadrant
    ]

    pos = [s for s in samples if s.presence[value_index] == 1.0]
    neg = [
        s for s in samples
        if s.presence[value_index] == 0.0
        and not any(s.presence[i] == 1.0 for i in same_quadrant_idx)
    ]
    if len(pos) < n_per_class or len(neg) < n_per_class:
        # Caller decides what to do; return what we have.
        n = min(n_per_class, len(pos), len(neg))
    else:
        n = n_per_class
    pos_idx = rng.choice(len(pos), size=n, replace=False)
    neg_idx = rng.choice(len(neg), size=n, replace=False)
    return [pos[i].text for i in pos_idx], [neg[i].text for i in neg_idx]


# --- PVQ-RR-like items (placeholders; replace with licensed items) --------

# 3 items per value. Each is a third-person portrait. Standard PVQ scoring asks
# the respondent how much the portrait person is like them. We adapt to score
# the model's option-logit response on a 6-point Likert scale.
PVQ_ITEMS: dict[str, list[str]] = {
    "Self-direction: thought": [
        "It is important to her to form her views independently.",
        "Thinking up new ideas and being creative is important to her.",
        "It is important to her to develop her own opinions.",
    ],
    "Self-direction: action": [
        "It is important to her to make her own decisions about her life.",
        "It is important to her to plan her activities independently.",
        "Being free to choose what she does is important to her.",
    ],
    "Stimulation": [
        "She always looks for different kinds of things to do.",
        "Excitement in life is important to her.",
        "She thinks it is important to have all sorts of new experiences.",
    ],
    "Hedonism": [
        "Having a good time is important to her.",
        "She seeks every chance she can to have fun.",
        "Enjoying life's pleasures is important to her.",
    ],
    "Achievement": [
        "Being very successful is important to her.",
        "She wants people to admire her achievements.",
        "Getting ahead in life is important to her.",
    ],
    "Power: dominance": [
        "It is important to her to be the one who tells others what to do.",
        "She wants people to do what she says.",
        "Having authority over others is important to her.",
    ],
    "Power: resources": [
        "Being wealthy is important to her.",
        "She pursues high status and material possessions.",
        "It is important to her to have expensive things.",
    ],
    "Face": [
        "It is important to her that no one should ever shame her.",
        "Protecting her public image is important to her.",
        "She wants others to always show her respect.",
    ],
    "Security: personal": [
        "Her personal security is extremely important to her.",
        "She tries to avoid anything dangerous.",
        "It is important to her to live in safe surroundings.",
    ],
    "Security: societal": [
        "It is important to her that her country be secure and stable.",
        "She wants the state to be strong so it can defend its citizens.",
        "Having a stable government is important to her.",
    ],
    "Tradition": [
        "Following the customs handed down by her family is important to her.",
        "She thinks it is important to maintain traditional values.",
        "Religious belief is important to her.",
    ],
    "Conformity: rules": [
        "She believes she should always do what she is told.",
        "Obeying all the laws is important to her.",
        "She thinks people should follow rules even when no one is watching.",
    ],
    "Conformity: interpersonal": [
        "It is important to her never to annoy anyone.",
        "She tries hard not to upset other people.",
        "Avoiding bothering others is important to her.",
    ],
    "Humility": [
        "Being humble is important to her.",
        "She tries not to draw attention to herself.",
        "She thinks it is best to be modest about her achievements.",
    ],
    "Benevolence: dependability": [
        "Being a dependable and trustworthy friend is important to her.",
        "She wants those close to her to be able to rely on her completely.",
        "It is important to her to be loyal to her friends.",
    ],
    "Benevolence: caring": [
        "It is very important to her to help the people close to her.",
        "She wants to care for the well-being of those she knows.",
        "Looking after the people in her life is important to her.",
    ],
    "Universalism: concern": [
        "It is important to her that every person be treated justly.",
        "She wants everyone to have equal opportunities.",
        "Protecting the weak in society is important to her.",
    ],
    "Universalism: nature": [
        "She strongly believes that people should care for nature.",
        "Protecting the environment is important to her.",
        "It is important to her to live in harmony with nature.",
    ],
    "Universalism: tolerance": [
        "It is important to her to listen to people who are different from her.",
        "She thinks it is important to understand people who disagree with her.",
        "Accepting people as they are is important to her.",
    ],
}

assert set(PVQ_ITEMS.keys()) == set(SCHWARTZ_19), "PVQ items must cover all 19 values"
assert all(len(v) == 3 for v in PVQ_ITEMS.values()), "Exactly 3 items per value"


# First-person versions of the same items, used by Pilot 5. These are written
# by hand rather than rewritten from the third-person items because the regex
# rewrite is unreliable for the her/possessive distinction. Same caveat applies
# as for PVQ_ITEMS: replace with licensed first-person items before publication.
PVQ_ITEMS_FIRST_PERSON: dict[str, list[str]] = {
    "Self-direction: thought": [
        "It is important to me to form my views independently.",
        "Thinking up new ideas and being creative is important to me.",
        "It is important to me to develop my own opinions.",
    ],
    "Self-direction: action": [
        "It is important to me to make my own decisions about my life.",
        "It is important to me to plan my activities independently.",
        "Being free to choose what I do is important to me.",
    ],
    "Stimulation": [
        "I always look for different kinds of things to do.",
        "Excitement in life is important to me.",
        "I think it is important to have all sorts of new experiences.",
    ],
    "Hedonism": [
        "Having a good time is important to me.",
        "I seek every chance I can to have fun.",
        "Enjoying life's pleasures is important to me.",
    ],
    "Achievement": [
        "Being very successful is important to me.",
        "I want people to admire my achievements.",
        "Getting ahead in life is important to me.",
    ],
    "Power: dominance": [
        "It is important to me to be the one who tells others what to do.",
        "I want people to do what I say.",
        "Having authority over others is important to me.",
    ],
    "Power: resources": [
        "Being wealthy is important to me.",
        "I pursue high status and material possessions.",
        "It is important to me to have expensive things.",
    ],
    "Face": [
        "It is important to me that no one should ever shame me.",
        "Protecting my public image is important to me.",
        "I want others to always show me respect.",
    ],
    "Security: personal": [
        "My personal security is extremely important to me.",
        "I try to avoid anything dangerous.",
        "It is important to me to live in safe surroundings.",
    ],
    "Security: societal": [
        "It is important to me that my country be secure and stable.",
        "I want the state to be strong so it can defend its citizens.",
        "Having a stable government is important to me.",
    ],
    "Tradition": [
        "Following the customs handed down by my family is important to me.",
        "I think it is important to maintain traditional values.",
        "Religious belief is important to me.",
    ],
    "Conformity: rules": [
        "I believe I should always do what I am told.",
        "Obeying all the laws is important to me.",
        "I think people should follow rules even when no one is watching.",
    ],
    "Conformity: interpersonal": [
        "It is important to me never to annoy anyone.",
        "I try hard not to upset other people.",
        "Avoiding bothering others is important to me.",
    ],
    "Humility": [
        "Being humble is important to me.",
        "I try not to draw attention to myself.",
        "I think it is best to be modest about my achievements.",
    ],
    "Benevolence: dependability": [
        "Being a dependable and trustworthy friend is important to me.",
        "I want those close to me to be able to rely on me completely.",
        "It is important to me to be loyal to my friends.",
    ],
    "Benevolence: caring": [
        "It is very important to me to help the people close to me.",
        "I want to care for the well-being of those I know.",
        "Looking after the people in my life is important to me.",
    ],
    "Universalism: concern": [
        "It is important to me that every person be treated justly.",
        "I want everyone to have equal opportunities.",
        "Protecting the weak in society is important to me.",
    ],
    "Universalism: nature": [
        "I strongly believe that people should care for nature.",
        "Protecting the environment is important to me.",
        "It is important to me to live in harmony with nature.",
    ],
    "Universalism: tolerance": [
        "It is important to me to listen to people who are different from me.",
        "I think it is important to understand people who disagree with me.",
        "Accepting people as they are is important to me.",
    ],
}
assert set(PVQ_ITEMS_FIRST_PERSON.keys()) == set(SCHWARTZ_19)
assert all(len(v) == 3 for v in PVQ_ITEMS_FIRST_PERSON.values())


# --- Neutral prompts for attractor extraction (B1) -----------------------

NEUTRAL_PROMPTS = [
    "Hello.",
    "Hi there.",
    "Good morning.",
    "How are you today?",
    "What's your name?",
    "Tell me about yourself.",
    "Can you help me with something?",
    "I have a question.",
    "What can you do?",
    "Are you there?",
    "Nice to meet you.",
    "What is the weather like?",
    "Do you have any hobbies?",
    "What did you do today?",
    "Can we talk?",
    "I would like to chat.",
    "What time is it?",
    "How old are you?",
    "Where are you from?",
    "Do you speak English?",
    "Tell me a joke.",
    "What's new?",
    "How do you feel?",
    "What's on your mind?",
    "Can you explain something to me?",
    "I need a recommendation.",
    "What do you think?",
    "Are you busy?",
    "How long have you been here?",
    "What's your favorite color?",
    "Do you remember me?",
    "Have we met before?",
    "What do you like to do?",
    "Where should I go for vacation?",
    "What should I have for dinner?",
    "Can you suggest a book?",
    "What movie should I watch?",
    "I'm not sure what to do.",
    "Can you keep me company?",
    "Tell me something interesting.",
    "What is the meaning of this word?",
    "How do I solve this problem?",
    "What is your opinion?",
    "Do you have any advice?",
    "I am looking for ideas.",
    "Can you help me think?",
    "Let's have a conversation.",
    "What is on the agenda?",
    "How was your day?",
    "Are you free to chat?",
]
assert len(NEUTRAL_PROMPTS) >= 50
