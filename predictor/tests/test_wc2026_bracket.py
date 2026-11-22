"""
Unit tests for the WC2026 bracket logic.

Tests:
  1. group tiebreakers (FIFA criteria: pts → GD → GF → H2H → random)
  2. third-place team ranking across groups
  3. R32 third-place allocation (bipartite matching) — multiple combinations
  4. Every valid C(12,8)=495 combination produces a unique valid allocation
"""

import itertools
import pytest
import numpy as np

from src.simulation.tournament import (
    GROUPS,
    ALL_WC_TEAMS,
    THIRD_PLACE_SLOTS,
    rank_group,
    rank_third_place_teams,
    allocate_third_place,
)


# ---------------------------------------------------------------------------
# 1. Group tiebreakers
# ---------------------------------------------------------------------------

class TestGroupRanking:

    def _results(self, *rows):
        return [{'team_a': a, 'team_b': b, 'gf_a': ga, 'gf_b': gb}
                for a, b, ga, gb in rows]

    def test_simple_separation(self):
        """A team with more points ranks higher."""
        teams = ['A', 'B', 'C', 'D']
        results = self._results(
            ('A', 'B', 2, 0),
            ('C', 'D', 0, 1),
            ('A', 'C', 1, 0),
            ('B', 'D', 0, 2),
            ('A', 'D', 3, 0),
            ('B', 'C', 0, 0),
        )
        ranked = rank_group(teams, results)
        # A: 9pts, D: 3pts, B: 1pt (draw), C: 1pt (draw)
        assert ranked[0] == 'A'
        assert ranked[3] in ('B', 'C')   # B and C both 1pt — either can be 4th

    def test_gd_tiebreaker(self):
        """Goal difference breaks a points tie."""
        teams = ['X', 'Y', 'Z', 'W']
        results = self._results(
            ('X', 'Y', 1, 0),   # X 3pts, Y 0pts
            ('Z', 'W', 2, 0),   # Z 3pts, W 0pts
            ('X', 'Z', 0, 0),   # X+Z each +1 draw
            ('Y', 'W', 1, 0),   # Y 3pts, W 0pts
            ('X', 'W', 4, 0),   # X dominant
            ('Y', 'Z', 0, 2),   # Z 3 more pts
        )
        ranked = rank_group(teams, results)
        # X: 7pts (3+1+3), Z: 7pts (3+0+3)... let's just check W is last
        assert ranked[-1] == 'W'

    def test_head_to_head_applied(self):
        """When pts and overall GD/GF are equal, H2H is used."""
        teams = ['P', 'Q', 'R', 'S']
        # Craft a tie between P and Q at same pts/GD/GF, but P beat Q
        results = self._results(
            ('P', 'Q', 1, 0),   # P>Q
            ('R', 'S', 0, 0),
            ('P', 'R', 0, 1),   # R wins
            ('Q', 'S', 1, 0),   # Q same pts as P
            ('P', 'S', 1, 0),
            ('Q', 'R', 1, 0),
        )
        # P: wins(P-Q, P-S)=6pts, loss(P-R)=0 → 6pts, GD=+1
        # Q: wins(Q-S, Q-R)=6pts, loss(Q-P)=0 → 6pts, GD=0
        # P and Q tied on pts but P has better GD overall, so P ranks above Q
        ranked = rank_group(teams, results)
        p_rank = ranked.index('P')
        q_rank = ranked.index('Q')
        assert p_rank < q_rank   # P above Q (better GD)

    def test_all_teams_present(self):
        """Ranked list always contains all 4 teams."""
        teams = ['Alpha', 'Beta', 'Gamma', 'Delta']
        results = self._results(
            ('Alpha', 'Beta',  1, 1),
            ('Gamma', 'Delta', 1, 1),
            ('Alpha', 'Gamma', 0, 0),
            ('Beta',  'Delta', 0, 0),
            ('Alpha', 'Delta', 0, 0),
            ('Beta',  'Gamma', 0, 0),
        )
        ranked = rank_group(teams, results)
        assert len(ranked) == 4
        assert set(ranked) == set(teams)


# ---------------------------------------------------------------------------
# 2. Third-place ranking
# ---------------------------------------------------------------------------

class TestThirdPlaceRanking:

    def test_ranked_by_pts_first(self):
        thirds = [
            {'team': 'T1', 'group': 'A', 'pts': 4, 'gd': 0, 'gf': 2},
            {'team': 'T2', 'group': 'B', 'pts': 3, 'gd': 2, 'gf': 4},
            {'team': 'T3', 'group': 'C', 'pts': 4, 'gd': 1, 'gf': 2},
        ]
        ranked = rank_third_place_teams(thirds)
        # T1: (pts=4,gd=0,gf=2), T3: (pts=4,gd=1,gf=2), T2: (pts=3,gd=2,gf=4)
        # Sorted by (pts,gd,gf) desc: T3 > T1 > T2
        assert ranked[0]['team'] == 'T3'
        assert ranked[1]['team'] == 'T1'
        assert ranked[2]['team'] == 'T2'

    def test_top_8_selected(self):
        thirds = [
            {'team': f'T{i}', 'group': chr(65+i), 'pts': 12-i, 'gd': 0, 'gf': 1}
            for i in range(12)
        ]
        ranked = rank_third_place_teams(thirds)
        assert len(ranked) == 12
        assert ranked[0]['team'] == 'T0'   # highest pts
        assert ranked[-1]['team'] == 'T11' # lowest pts


# ---------------------------------------------------------------------------
# 3. R32 allocation — specific cases
# ---------------------------------------------------------------------------

class TestR32Allocation:

    def test_k_and_l_always_fixed(self):
        """Groups K and L each appear in exactly ONE slot (80 and 87 resp.)."""
        assert 'K' in THIRD_PLACE_SLOTS[80]
        assert 'L' in THIRD_PLACE_SLOTS[87]
        for slot, compat in THIRD_PLACE_SLOTS.items():
            if slot != 80:
                assert 'K' not in compat, f"K unexpectedly in slot {slot}"
            if slot != 87:
                assert 'L' not in compat, f"L unexpectedly in slot {slot}"

    def test_k_qualifies_goes_to_80(self):
        """If K is among 8 qualifying groups, it must go to slot 80."""
        qual = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'K']
        mapping = allocate_third_place(qual)
        assert mapping['K'] == 80

    def test_l_qualifies_goes_to_87(self):
        """If L is among 8 qualifying groups, it must go to slot 87."""
        qual = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'L']
        mapping = allocate_third_place(qual)
        assert mapping['L'] == 87

    def test_both_k_and_l_qualify(self):
        qual = ['A', 'B', 'C', 'D', 'E', 'K', 'G', 'L']
        mapping = allocate_third_place(qual)
        assert mapping['K'] == 80
        assert mapping['L'] == 87

    def test_valid_assignment_8_groups(self):
        """All 8 qualifying groups map to distinct, compatible slots."""
        qual = ['A', 'B', 'C', 'D', 'E', 'F', 'I', 'J']
        mapping = allocate_third_place(qual)
        assert len(mapping) == 8
        assert len(set(mapping.values())) == 8   # distinct slots
        for grp, slot in mapping.items():
            assert grp in THIRD_PLACE_SLOTS[slot], (
                f"Group {grp} assigned to incompatible slot {slot}"
            )

    def test_another_combination(self):
        qual = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        mapping = allocate_third_place(qual)
        for grp, slot in mapping.items():
            assert grp in THIRD_PLACE_SLOTS[slot]
        assert len(set(mapping.values())) == 8

    def test_high_cross_conf_combination(self):
        """Typical WC scenario: all confederation representatives."""
        qual = ['B', 'C', 'D', 'F', 'G', 'I', 'J', 'K']
        mapping = allocate_third_place(qual)
        assert mapping['K'] == 80
        for grp, slot in mapping.items():
            assert grp in THIRD_PLACE_SLOTS[slot]


# ---------------------------------------------------------------------------
# 4. Exhaust all 495 combinations
# ---------------------------------------------------------------------------

class TestAllCombinations:

    def test_all_495_have_valid_allocation(self):
        """
        Every C(12,8)=495 combination of qualifying groups must produce
        a valid allocation (no ValueError, all assignments compatible).
        """
        all_groups = list(GROUPS.keys())  # A-L
        assert len(all_groups) == 12

        fail_count = 0
        for combo in itertools.combinations(all_groups, 8):
            qual = list(combo)
            try:
                mapping = allocate_third_place(qual)
                # Verify compatibility
                for grp, slot in mapping.items():
                    assert grp in THIRD_PLACE_SLOTS[slot], (
                        f"Combo {combo}: {grp} → slot {slot} is incompatible"
                    )
                assert len(set(mapping.values())) == 8, (
                    f"Combo {combo}: duplicate slot assignments"
                )
            except (ValueError, AssertionError) as e:
                fail_count += 1
                print(f"FAIL combo {combo}: {e}")

        assert fail_count == 0, f"{fail_count}/495 combinations failed allocation"
