"""Exact tabular adapter for Gymnasium Blackjack.

Gymnasium's raw observation contains the player sum, dealer upcard, and
usable-ace flag. The wrapper also uses the reset boundary to distinguish a
two-card natural blackjack from a soft 21 reached later in an episode. The
resulting model contains 290 decision states and one terminal bust sink:

* hard totals 4 through 21: 18 totals by 10 dealer cards
* soft totals 12 through 21: 10 totals by 10 dealer cards
* natural blackjack: 1 hand type by 10 dealer cards
* one terminal sink for all bust observations

This context is necessary because the raw tuple for a natural blackjack is the
same as the tuple for any other soft 21.
"""

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache

import gymnasium as gym

CARD_WEIGHTS = (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
    (8, 1),
    (9, 1),
    (10, 4),
)
DEALER_CARDS = (*range(2, 11), 1)
N_DECISION_STATES = 290
TERMINAL_STATE = N_DECISION_STATES
N_STATES = N_DECISION_STATES + 1


def _dealer_index(dealer_card):
    if not 1 <= dealer_card <= 10:
        raise ValueError(f"Invalid Blackjack dealer card: {dealer_card!r}")
    return (dealer_card - 2) % 10


def _transform_observation(observation, *, natural=False):
    player_sum, dealer_card, usable_ace = observation
    player_sum = int(player_sum)
    usable_ace = bool(usable_ace)

    dealer_index = _dealer_index(int(dealer_card))
    if player_sum > 21:
        return TERMINAL_STATE

    if natural:
        if player_sum != 21 or not usable_ace:
            raise ValueError("A natural blackjack must be a soft total of 21")
        hand_index = 28
    elif usable_ace:
        if not 12 <= player_sum <= 21:
            raise ValueError(f"Invalid Blackjack soft total: {player_sum!r}")
        hand_index = player_sum + 6
    else:
        if not 4 <= player_sum <= 21:
            raise ValueError(f"Invalid Blackjack hard total: {player_sum!r}")
        hand_index = player_sum - 4

    return hand_index * 10 + dealer_index


def _add_card(player_sum, usable_ace, card):
    if usable_ace:
        player_sum += card
        if player_sum > 21:
            return player_sum - 10, False
        return player_sum, True

    player_sum += card
    if card == 1 and player_sum + 10 <= 21:
        return player_sum + 10, True
    return player_sum, False


@lru_cache(maxsize=None)
def _dealer_outcomes(player_sum, usable_ace):
    if player_sum >= 17:
        score = 0 if player_sum > 21 else player_sum
        return ((score, Fraction(1)),)

    probabilities = defaultdict(Fraction)
    for card, weight in CARD_WEIGHTS:
        next_sum, next_usable_ace = _add_card(player_sum, usable_ace, card)
        for score, probability in _dealer_outcomes(next_sum, next_usable_ace):
            probabilities[score] += Fraction(weight, 13) * probability
    return tuple(sorted(probabilities.items()))


@lru_cache(maxsize=None)
def _dealer_scores(dealer_card):
    initial_sum = 11 if dealer_card == 1 else dealer_card
    initial_usable_ace = dealer_card == 1
    probabilities = defaultdict(Fraction)
    for hidden_card, weight in CARD_WEIGHTS:
        dealer_sum, usable_ace = _add_card(initial_sum, initial_usable_ace, hidden_card)
        dealer_natural = {dealer_card, hidden_card} == {1, 10}
        for score, probability in _dealer_outcomes(dealer_sum, usable_ace):
            probabilities[(score, dealer_natural)] += Fraction(weight, 13) * probability
    return tuple(
        (score, dealer_natural, probability)
        for (score, dealer_natural), probability in sorted(probabilities.items())
    )


def _stick_transitions(
    state,
    player_sum,
    dealer_card,
    *,
    player_natural,
    natural_payout,
    sab,
):
    reward_probabilities = defaultdict(Fraction)
    for dealer_score, dealer_natural, probability in _dealer_scores(dealer_card):
        reward = float(player_sum > dealer_score) - float(player_sum < dealer_score)
        if sab and player_natural and not dealer_natural:
            reward = 1.0
        elif natural_payout and player_natural and reward == 1.0:
            reward = 1.5
        reward_probabilities[reward] += probability
    return [
        (float(probability), state, reward, True)
        for reward, probability in sorted(reward_probabilities.items())
    ]


def _hit_transitions(player_sum, dealer_card, usable_ace):
    outcomes = defaultdict(int)
    for card, weight in CARD_WEIGHTS:
        next_sum, next_usable_ace = _add_card(player_sum, usable_ace, card)
        if next_sum > 21:
            outcome = (TERMINAL_STATE, -1.0, True)
        else:
            next_state = _transform_observation(
                (next_sum, dealer_card, next_usable_ace)
            )
            outcome = (next_state, 0.0, False)
        outcomes[outcome] += weight
    return [
        (weight / 13, next_state, reward, terminal)
        for (next_state, reward, terminal), weight in sorted(outcomes.items())
    ]


def _build_transition_model(*, natural_payout, sab):
    model = {state: {action: [] for action in range(2)} for state in range(N_STATES)}
    for usable_ace, player_sums in (
        (False, range(4, 22)),
        (True, range(12, 22)),
    ):
        for player_sum in player_sums:
            for dealer_card in DEALER_CARDS:
                state = _transform_observation((player_sum, dealer_card, usable_ace))
                model[state][0] = _stick_transitions(
                    state,
                    player_sum,
                    dealer_card,
                    player_natural=False,
                    natural_payout=natural_payout,
                    sab=sab,
                )
                model[state][1] = _hit_transitions(player_sum, dealer_card, usable_ace)

    for dealer_card in DEALER_CARDS:
        state = _transform_observation((21, dealer_card, True), natural=True)
        model[state][0] = _stick_transitions(
            state,
            21,
            dealer_card,
            player_natural=True,
            natural_payout=natural_payout,
            sab=sab,
        )
        model[state][1] = _hit_transitions(21, dealer_card, True)

    terminal_transition = [(1.0, TERMINAL_STATE, 0.0, True)]
    model[TERMINAL_STATE] = {
        0: terminal_transition,
        1: terminal_transition.copy(),
    }
    return model


class BlackjackWrapper(gym.Wrapper):
    """Expose exact tabular observations and a model for Blackjack."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Discrete(N_STATES)
        self._player_natural = False
        base_env = env.unwrapped
        self._P = _build_transition_model(
            natural_payout=bool(getattr(base_env, "natural", False)),
            sab=bool(getattr(base_env, "sab", False)),
        )

    def reset(self, *, seed=None, options=None):
        """Reset Blackjack and return its context-aware discrete observation."""
        observation, info = self.env.reset(seed=seed, options=options)
        self._player_natural = bool(observation[0] == 21 and observation[2])
        return (
            _transform_observation(
                observation,
                natural=self._player_natural,
            ),
            info,
        )

    def step(self, action):
        """Step Blackjack and return its context-aware discrete observation."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if action == 1:
            self._player_natural = False
        return (
            _transform_observation(
                observation,
                natural=self._player_natural,
            ),
            reward,
            terminated,
            truncated,
            info,
        )

    @property
    def P(self):
        """Return the exact transition and reward model."""
        return self._P

    @property
    def transform_obs(self):
        """Return the context-aware observation conversion function."""
        return _transform_observation
