import json
import tempfile
import unittest
from pathlib import Path

from training_data import (
    DatasetFormatError,
    build_prompt_completion_examples,
    read_records,
)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.assertions = (tokenize, add_generation_prompt)
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_multiturn_conversation_expands_each_assistant_turn(self):
        records = [
            {
                "messages": [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "One?"},
                    {"role": "assistant", "content": "First."},
                    {"role": "user", "content": "Two?"},
                    {"role": "assistant", "content": "Second."},
                ]
            }
        ]
        examples, stats = build_prompt_completion_examples(records, self.tokenizer)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["completion"], "First.<eos>")
        self.assertIn("<assistant>First.", examples[1]["prompt"])
        self.assertEqual(stats.output_examples, 2)

    def test_role_aliases_and_content_blocks(self):
        records = [
            {
                "messages": [
                    {"role": "human", "content": [{"type": "text", "text": "Hi"}]},
                    {"role": "gpt", "content": [{"type": "output_text", "text": "Hello"}]},
                ]
            }
        ]
        examples, _ = build_prompt_completion_examples(records, self.tokenizer)
        self.assertEqual(examples[0]["completion"], "Hello<eos>")
        self.assertTrue(examples[0]["prompt"].startswith("<user>Hi"))

    def test_prompt_completion_and_plain_text(self):
        records = [
            {"prompt": "Question: ", "completion": "Answer"},
            {"text": "Standalone"},
        ]
        examples, stats = build_prompt_completion_examples(records, self.tokenizer)
        self.assertEqual(examples[0], {"prompt": "Question: ", "completion": "Answer<eos>"})
        self.assertEqual(examples[1], {"prompt": "", "completion": "Standalone<eos>"})
        self.assertEqual(stats.prompt_completion_records, 1)
        self.assertEqual(stats.plain_text_records, 1)

    def test_exact_duplicates_are_removed(self):
        records = [{"text": "Same"}, {"text": "Same"}]
        examples, stats = build_prompt_completion_examples(records, self.tokenizer)
        self.assertEqual(len(examples), 1)
        self.assertEqual(stats.duplicates_removed, 1)

    def test_invalid_record_raises_in_strict_mode(self):
        with self.assertRaises(DatasetFormatError):
            build_prompt_completion_examples([{"messages": []}], self.tokenizer)

    def test_invalid_record_can_be_skipped(self):
        examples, stats = build_prompt_completion_examples(
            [{"messages": []}, {"text": "Valid"}], self.tokenizer, strict=False
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(stats.skipped_records, 1)

    def test_read_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                json.dumps({"text": "A"}) + "\n" + json.dumps({"text": "B"}) + "\n",
                encoding="utf-8",
            )
            records = read_records(path)
        self.assertEqual([record["text"] for record in records], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
