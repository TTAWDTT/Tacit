from __future__ import annotations

import unittest
from itertools import combinations

from experiments.emergent_ood_v0_4.split import build_split
from research.emergent_ood_scaling import scaling_report


class EmergentOODScalingTests(unittest.TestCase):
    def test_default_fixture_scaling_matches_exact_expected_counts(self):
        report = scaling_report(dimensions=4, value_count=4, candidate_count=4)
        self.assertEqual((report["universe_size"], report["training_size"], report["held_out_size"]),
                         (256, 192, 64))
        self.assertEqual(report["one_way_zero_error_payload_lower_bound_bits"], 8)
        self.assertEqual(report["held_out_support_fixed_width_zero_error_payload_lower_bound_bits"], 6)
        self.assertEqual(report["held_out_support_rank_code_payload_bytes"], 1)
        self.assertEqual(report["full_universe_rank_code_payload_bytes"], 1)
        self.assertEqual(report["held_out_rank_code_vs_full_universe_ideal_bit_saving"], 2)
        self.assertEqual(report["held_out_rank_code_vs_full_universe_serialized_byte_saving"], 0)
        self.assertEqual([
            (row["partial_assignments"], row["held_out_completions_per_assignment"],
             row["training_completions_per_assignment"])
            for row in report["proper_partial_orders"]
        ], [(16, 16, 48), (96, 4, 12), (256, 1, 3)])
        self.assertEqual(report["balanced_no_message_accuracy"], 0.25)
        self.assertEqual(report["balanced_no_message_accuracy_exact"], {"numerator": 1, "denominator": 4})
        self.assertEqual(report["held_out_fraction_exact"], {"numerator": 1, "denominator": 4})

    def test_exact_partial_assignment_counts_hold_for_multiple_dimensions_and_values(self):
        for dimensions in range(2, 6):
            for value_count in range(2, 5):
                attributes = tuple(f"axis-{axis}" for axis in range(dimensions))
                values = tuple(
                    tuple(f"a{axis}-v{value}" for value in range(value_count))
                    for axis in range(dimensions)
                )
                split = build_split(seed=7, attributes=attributes, values=values)
                report = scaling_report(
                    dimensions=dimensions, value_count=value_count, candidate_count=2,
                )
                meaning_by_id = {row["meaning_id"]: row for row in split["meanings"]}
                train_ids = set(split["train_meaning_ids"])
                held_out_ids = set(split["held_out_meaning_ids"])
                self.assertEqual(len(train_ids), report["training_size"])
                self.assertEqual(len(held_out_ids), report["held_out_size"])
                for summary in report["proper_partial_orders"]:
                    order = summary["order"]
                    for axes in combinations(range(dimensions), order):
                        supports: dict[str, dict[tuple[str, ...], int]] = {"train": {}, "held_out": {}}
                        for stage, ids in (("train", train_ids), ("held_out", held_out_ids)):
                            for meaning_id in ids:
                                meaning = meaning_by_id[meaning_id]
                                key = tuple(meaning["values"][axis] for axis in axes)
                                supports[stage][key] = supports[stage].get(key, 0) + 1
                        self.assertEqual(len(supports["train"]), value_count ** order)
                        self.assertEqual(len(supports["held_out"]), value_count ** order)
                        self.assertEqual(
                            set(supports["train"].values()),
                            {summary["training_completions_per_assignment"]},
                        )
                        self.assertEqual(
                            set(supports["held_out"].values()),
                            {summary["held_out_completions_per_assignment"]},
                        )

    def test_invalid_scaling_parameters_are_rejected(self):
        for kwargs in (
            {"dimensions": 1, "value_count": 4},
            {"dimensions": 4, "value_count": True},
            {"dimensions": 4, "value_count": 4, "candidate_count": 1},
            {"dimensions": 2, "value_count": 2, "candidate_count": 5},
        ):
            with self.assertRaises(ValueError):
                scaling_report(**kwargs)

    def test_bit_savings_and_rounded_byte_savings_are_reported_separately(self):
        reports = [
            scaling_report(dimensions=dimensions, value_count=value_count)
            for dimensions in range(2, 7)
            for value_count in range(2, 6)
        ]
        for report in reports:
            full_bits = report["one_way_zero_error_payload_lower_bound_bits"]
            heldout_bits = report["held_out_support_fixed_width_zero_error_payload_lower_bound_bits"]
            full_bytes = report["full_universe_rank_code_payload_bytes"]
            heldout_bytes = report["held_out_support_rank_code_payload_bytes"]
            self.assertEqual(
                report["held_out_rank_code_vs_full_universe_ideal_bit_saving"],
                full_bits - heldout_bits,
            )
            self.assertEqual(
                report["held_out_rank_code_vs_full_universe_serialized_byte_saving"],
                full_bytes - heldout_bytes,
            )
            self.assertGreaterEqual(full_bytes, heldout_bytes)
        self.assertTrue(any(
            report["held_out_rank_code_vs_full_universe_serialized_byte_saving"] > 0
            for report in reports
        ))


if __name__ == "__main__":
    unittest.main()
