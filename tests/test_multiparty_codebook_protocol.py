import json
import shutil
import uuid
import unittest

from tacit import ChatCompletion, DialogueProtocolCard
from experiments.multiparty_sum_v0_1.codebook_protocol import (
    build_protocol_card,
    run_codebook_episode,
)
from experiments.multiparty_sum_v0_1.codebook_runner import run_codebook_bundle_batch
from experiments.multiparty_sum_v0_1.generate_tasks import ROOT, generate_dataset


def _width(partition):
    return (len(partition) - 1).bit_length()


class FakeCodebookSender:
    def __init__(self, role, value, card, raw_output=None):
        self.role = role
        self.value = value
        self.card = card
        self.raw_output = raw_output
        self.calls = []

    def complete(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        if self.raw_output is not None:
            output = self.raw_output
        else:
            index = int(self.role[1:]) - 1
            partition = self.card["sender_partitions"][index]
            message_id = next(i for i, block in enumerate(partition) if self.value in block)
            output = format(message_id, f"0{_width(partition)}b")
        return ChatCompletion(output, f"fake-{self.role}")


class FakeCodebookReceiver:
    def __init__(self, card):
        self.card = card
        self.calls = []

    def complete(self, messages):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        by_speaker = {row["sender"]: row["message"] for row in payload["visible_transcript"]}
        possibilities = [set(range(4))]
        for index, partition in enumerate(self.card["sender_partitions"], start=1):
            width = _width(partition)
            if width == 0:
                possibilities.append(set(range(4)))
                continue
            codeword = by_speaker.get(f"S{index}")
            if codeword is None or len(codeword) != width or set(codeword) - {"0", "1"}:
                possibilities.append(set(range(4)))
                continue
            message_id = int(codeword, 2)
            possibilities.append(set(partition[message_id]) if message_id < len(partition) else set(range(4)))
        coefficients = [1]
        for values in possibilities[1:]:
            updated = [0] * (len(coefficients) + max(values))
            for subtotal, count in enumerate(coefficients):
                for value in values:
                    updated[subtotal + value] += count
            coefficients = updated
        answer = coefficients.index(max(coefficients))
        return ChatCompletion(str(answer), "fake-R")


class MultipartyCodebookProtocolTests(unittest.TestCase):
    def test_all_cards_round_trip_and_encode_every_private_value(self):
        for agent_count in range(2, 5):
            for budget in range(1, 2 * agent_count + 1):
                with self.subTest(agent_count=agent_count, budget=budget):
                    try:
                        card = build_protocol_card(agent_count, budget)
                    except ValueError as exc:
                        self.assertEqual(
                            str(exc),
                            "the zero-bit point has no codebook senders; use the no_message condition",
                        )
                        continue
                    runtime_card = DialogueProtocolCard.from_json(
                        json.dumps(card["dialogue_protocol_card"], separators=(",", ":"))
                    )
                    self.assertEqual(runtime_card.to_dict(), card["dialogue_protocol_card"])
                    self.assertEqual(card["actual_fixed_width_payload_bits"], sum(card["sender_widths_bits"]))
                    self.assertLessEqual(card["actual_fixed_width_payload_bits"], budget)
                    self.assertEqual(
                        card["card_json_utf8_bytes"],
                        len(runtime_card.to_json_bytes()),
                    )
                    self.assertEqual(
                        card["system_instruction_utf8_bytes_per_episode"],
                        sum(card["role_instruction_utf8_bytes"].values()),
                    )
                    for partition in card["sender_partitions"]:
                        word_to_cell = {}
                        for message_id, cell in enumerate(partition):
                            word = format(message_id, f"0{_width(partition)}b") if _width(partition) else ""
                            self.assertNotIn(word, word_to_cell)
                            word_to_cell[word] = set(cell)
                        self.assertEqual(set().union(*word_to_cell.values()), set(range(4)))

    def test_full_information_card_uses_real_sdk_and_preserves_role_privacy(self):
        card = build_protocol_card(2, 4)
        values = [1, 3]
        senders = {
            role: FakeCodebookSender(role, values[int(role[1:]) - 1], card)
            for role in card["active_senders"]
        }
        receiver = FakeCodebookReceiver(card)
        result = run_codebook_episode(
            values,
            card_spec=card,
            sender_clients=senders,
            receiver_client=receiver,
        )
        self.assertTrue(result["exact_success"])
        self.assertTrue(result["receiver_matches_codebook_oracle"])
        self.assertTrue(result["all_delivered_sender_messages_match_partition"])
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["wire_bytes"], sum(row["application_bytes"] for row in result["sender_messages"]))
        for role, sender in senders.items():
            index = int(role[1:]) - 1
            self.assertEqual(sender.calls[0]["private_context"],
                             f"Public metadata: sender_count=2. Your private integer is {values[index]}.")
            self.assertEqual(sender.calls[0]["visible_transcript"], [])
        self.assertEqual(receiver.calls[0]["private_context"],
                         "Public metadata: sender_count=2. Assumed input prior: independent_uniform_integer_0_to_3.")
        self.assertEqual(len(receiver.calls[0]["visible_transcript"]), 2)

    def test_malformed_sender_is_separate_from_receiver_and_end_task_scores(self):
        card = build_protocol_card(2, 4)
        values = [0, 2]
        senders = {
            "S1": FakeCodebookSender("S1", 0, card, raw_output="not-bits"),
            "S2": FakeCodebookSender("S2", 2, card),
        }
        result = run_codebook_episode(
            values,
            card_spec=card,
            sender_clients=senders,
            receiver_client=FakeCodebookReceiver(card),
        )
        self.assertEqual(result["sender_messages"][0]["status"], "missing_or_malformed")
        self.assertFalse(result["all_delivered_sender_messages_match_partition"])
        self.assertTrue(result["receiver_matches_codebook_oracle"])

    def test_invalid_card_and_call_cap_reject_before_dispatch(self):
        card = build_protocol_card(2, 4)
        senders = {role: FakeCodebookSender(role, 0, card) for role in card["active_senders"]}
        receiver = FakeCodebookReceiver(card)
        altered = dict(card)
        altered["protocol_id"] = "caller-controlled"
        with self.assertRaisesRegex(ValueError, "canonical exhaustive frontier"):
            run_codebook_episode([0, 0], card_spec=altered, sender_clients=senders, receiver_client=receiver)
        with self.assertRaisesRegex(ValueError, "exceeding request cap"):
            run_codebook_episode([0, 0], card_spec=card, sender_clients=senders,
                                 receiver_client=receiver, request_cap=2)
        self.assertTrue(all(sender.calls == [] for sender in senders.values()))
        self.assertEqual(receiver.calls, [])

    def test_frozen_bundle_batch_uses_card_and_checks_total_call_cap(self):
        root = ROOT / ".cache" / f"test-codebook-runner-{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        try:
            bundle = root / "bundle"
            generate_dataset(bundle, task_key=bytes(range(32)), task_seed=7,
                             agent_counts=(2,), episodes_per_count=2)
            card = build_protocol_card(2, 4)
            sender_clients = {
                role: FakeCodebookSender(role, 0, card)
                for role in card["active_senders"]
            }
            receiver = FakeCodebookReceiver(card)
            with self.assertRaisesRegex(ValueError, "exceeding request cap"):
                run_codebook_bundle_batch(
                    bundle, agent_count=2, episode_indices=(0, 1),
                    payload_budget_bits=4, sender_clients=sender_clients,
                    receiver_client=receiver, request_cap=5,
                )
            self.assertTrue(all(not client.calls for client in sender_clients.values()))
            self.assertFalse(receiver.calls)

            report = run_codebook_bundle_batch(
                bundle, agent_count=2, episode_indices=(0,), payload_budget_bits=4,
                sender_clients=sender_clients, receiver_client=receiver, request_cap=3,
            )
            self.assertEqual(report["planned_model_calls"], 3)
            self.assertEqual(report["actual_model_calls"], 3)
            self.assertEqual(len(report["bundle_manifest_sha256"]), 64)
            self.assertTrue(report["episodes"][0]["episode_id"].startswith("sum-m02-"))
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
