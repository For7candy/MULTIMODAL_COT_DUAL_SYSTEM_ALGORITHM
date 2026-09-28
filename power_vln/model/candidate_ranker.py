"""Deterministic candidate ranking used to audit the model decision."""

from typing import Optional, Tuple

from power_vln.schemas import NavigationReasoning


def rank_reachable_candidates(
    reasoning: NavigationReasoning,
) -> Tuple[str, ...]:
    reachable = [
        evaluation
        for evaluation in reasoning.evaluations
        if evaluation.reachable
    ]
    reachable.sort(key=lambda item: (-item.score, item.candidate_id))
    return tuple(item.candidate_id for item in reachable)


def expected_candidate(reasoning: NavigationReasoning) -> Optional[str]:
    ranked = rank_reachable_candidates(reasoning)
    return ranked[0] if ranked else None
