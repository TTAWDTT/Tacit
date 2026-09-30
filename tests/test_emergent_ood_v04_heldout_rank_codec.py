from __future__ import annotations

import unittest

from experiments.emergent_ood_v0_4.heldout_rank_codec import (
    CODEC_ID, HeldOutRankCodec,
)
from experiments.emergent_ood_v0_4.split import build_split


class HeldOutRankCodecTests(unittest.TestCase):
    def test_default_support_is_bijectively_encoded_at_six_bits(self) -> None:
        split = build_split(seed=17)
        codec = HeldOutRankCodec(split)
        held_out = [row for row in split["meanings"] if row["split"] == "held_out"]
        payloads = [codec.encode(dict(zip(split["attributes"], row["values"]))) for row in held_out]

        self.assertEqual(CODEC_ID, "tlu.heldout-modular-rank.v1")
        self.assertEqual(codec.support_size, 64)
        self.assertEqual(codec.bit_width, 6)
        self.assertEqual(codec.payload_bytes, 1)
        self.assertEqual(len(set(payloads)), 64)
        for row, payload in zip(held_out, payloads):
            self.assertEqual(
                codec.decode(payload), dict(zip(split["attributes"], row["values"]))
            )

    def test_encoder_rejects_training_meanings_and_unknown_fields(self) -> None:
        split = build_split(seed=4)
        codec = HeldOutRankCodec(split)
        train_row = next(row for row in split["meanings"] if row["split"] == "train")
        train_meaning = dict(zip(split["attributes"], train_row["values"]))
        with self.assertRaisesRegex(ValueError, "outside"):
            codec.encode(train_meaning)
        with self.assertRaisesRegex(ValueError, "exactly"):
            codec.encode({**train_meaning, "extra": "value"})
        unknown = dict(train_meaning)
        unknown[split["attributes"][0]] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown value"):
            codec.encode(unknown)

    def test_non_power_of_two_support_rejects_unused_codewords_and_padding(self) -> None:
        split = build_split(
            seed=9, attributes=("a", "b"),
            values=(("a0", "a1", "a2"), ("b0", "b1", "b2")),
        )
        codec = HeldOutRankCodec(split)
        self.assertEqual(codec.support_size, 3)
        self.assertEqual(codec.bit_width, 2)
        self.assertEqual(codec.payload_bytes, 1)
        with self.assertRaisesRegex(ValueError, "unused"):
            codec.decode(bytes([3 << codec.padding_bits]))
        with self.assertRaisesRegex(ValueError, "padding"):
            codec.decode(b"\x01")

    def test_payload_shape_is_strict(self) -> None:
        codec = HeldOutRankCodec(build_split(seed=1))
        for payload in (bytearray(b"\x00"), b"", b"\x00\x00"):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "exactly 1 bytes"):
                codec.decode(payload)

    def test_rank_order_tracks_each_seed_and_axis_permutation(self) -> None:
        for seed in (0, 1, 5, 17, 100):
            split = build_split(seed=seed)
            codec = HeldOutRankCodec(split)
            held_out = [row for row in split["meanings"] if row["split"] == "held_out"]
            codes = {
                codec.encode(dict(zip(split["attributes"], row["values"])))
                for row in held_out
            }
            self.assertEqual(len(codes), codec.support_size)
            self.assertEqual(
                {tuple(codec.decode(code)[a] for a in split["attributes"]) for code in codes},
                {tuple(row["values"]) for row in held_out},
            )

    def test_small_dimension_and_cardinality_grid_exhausts_support(self) -> None:
        for dimensions in range(2, 5):
            for value_count in range(2, 5):
                attributes = tuple(f"axis_{axis}" for axis in range(dimensions))
                values = tuple(
                    tuple(f"axis_{axis}_value_{value}" for value in range(value_count))
                    for axis in range(dimensions)
                )
                split = build_split(seed=dimensions * 10 + value_count,
                                    attributes=attributes, values=values)
                codec = HeldOutRankCodec(split)
                meanings = [
                    dict(zip(attributes, row["values"]))
                    for row in split["meanings"]
                    if row["split"] == "held_out"
                ]
                payloads = [codec.encode(meaning) for meaning in meanings]

                with self.subTest(dimensions=dimensions, value_count=value_count):
                    self.assertEqual(codec.support_size, value_count ** (dimensions - 1))
                    self.assertEqual(codec.bit_width, (codec.support_size - 1).bit_length())
                    self.assertEqual(len(set(payloads)), codec.support_size)
                    self.assertEqual([codec.decode(payload) for payload in payloads], meanings)


if __name__ == "__main__":
    unittest.main()
