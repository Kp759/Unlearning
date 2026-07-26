import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import rwku_data as DATA  # noqa: E402
import rwku_representation as REP  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    eos_token = "<eos>"

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        assert add_generation_prompt
        return f"<user>{messages[0]['content']}</user><assistant>"

    @staticmethod
    def _encode(text, add_special_tokens):
        # Ignore whitespace so " A", " B", ... have distinct first tokens.
        values = [
            3 + (ord(character) % 120)
            for character in str(text)
            if not character.isspace()
        ]
        return ([1] if add_special_tokens else []) + values

    def __call__(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, list):
            return {
                "input_ids": [
                    self._encode(value, add_special_tokens) for value in text
                ]
            }
        return {"input_ids": self._encode(text, add_special_tokens)}


class TinyAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden):
        return self.o_proj(torch.tanh(self.q_proj(hidden) + self.v_proj(hidden)))


class TinyMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(torch.tanh(hidden))


class TinyLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = TinyAttention(hidden_size)
        self.mlp = TinyMLP(hidden_size)

    def forward(self, hidden):
        hidden = hidden + self.self_attn(hidden)
        return hidden + self.mlp(hidden)


class TinyBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_size, layer_count):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TinyLayer(hidden_size) for _ in range(layer_count)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyLlamaCausalLM(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=12, layer_count=2):
        super().__init__()
        torch.manual_seed(11)
        self.model = TinyBackbone(vocab_size, hidden_size, layer_count)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(
            tie_word_embeddings=False,
            use_cache=False,
            model_type="llama",
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        **kwargs,
    ):
        decoded = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = self.lm_head(decoded.last_hidden_state)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(
                (decoded.last_hidden_state,) if output_hidden_states else None
            ),
        )


class TinyOptimizer:
    def __init__(self, parameters, lr, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-self.lr)


def qa_row(index, *, level="2"):
    answers = ("alpha", "bravo", "charlie", "delta")
    return {
        "query": f"What fact {index} is associated with Stephen King?",
        "answer": answers[index],
        "level": level,
        "type": "simple question",
        "subject": "Stephen King",
    }


def loose_config(**overrides):
    values = {
        "steps": 4,
        "learning_rate": 1e-2,
        "rank": 2,
        "alpha": 2.0,
        "last_n_layers": 1,
        "max_length": 96,
        "retain_top_k": 8,
        "positive_max_rows": 1,
        "positive_tokens_per_row": 24,
        "candidate_scales": (1.0, 0.0),
        "max_retain_kl": 100.0,
        "min_retain_answer_probability_ratio": 1e-6,
        "max_retain_answer_probability_ratio": 1e6,
        "min_retain_top1_agreement": 0.0,
        "min_retain_hidden_cosine": 0.0,
        "max_retain_hidden_relative_l2": 100.0,
        "gate_retain_limit": 1,
        "selection_calibration_limit": 4,
        "selection_generation_limit": 0,
    }
    values.update(overrides)
    return REP.RepresentationConfig(**values)


class LoRAWrapperTests(unittest.TestCase):
    def test_injection_freezes_base_and_fp32_merge_matches_live_adapter(self):
        model = TinyLlamaCausalLM()
        config = loose_config()
        embeddings_before = model.get_input_embeddings().weight.detach().clone()
        head_before = model.get_output_embeddings().weight.detach().clone()
        first_layer_before = (
            model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        )
        selected_before = (
            model.model.layers[1].self_attn.q_proj.weight.detach().clone()
        )

        handles = REP.inject_lora_adapters(model, config)
        self.assertEqual(len(handles), 4)
        self.assertEqual({handle.layer_index for handle in handles}, {1})
        self.assertEqual(
            {handle.module_name for handle in handles},
            {"q_proj", "v_proj", "o_proj", "down_proj"},
        )
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in REP.adapter_parameters(handles))
        )
        self.assertFalse(model.get_input_embeddings().weight.requires_grad)
        self.assertFalse(model.get_output_embeddings().weight.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in REP.adapter_parameters(handles)))

        with torch.no_grad():
            for handle in handles:
                handle.wrapper.lora_B.fill_(0.03)
        input_ids = torch.tensor([[1, 7, 9, 11]], dtype=torch.long)
        live = model(input_ids, output_hidden_states=True).logits.detach()
        merge = REP.remove_lora_adapters(handles, merge_scale=1.0)
        merged = model(input_ids, output_hidden_states=True).logits.detach()

        self.assertTrue(torch.allclose(live, merged, atol=1e-6, rtol=1e-5))
        self.assertEqual(merge["merge_compute_dtype"], "float32")
        self.assertEqual(merge["changed_module_count"], 4)
        self.assertFalse(
            any(isinstance(module, REP.LoRALinear) for module in model.modules())
        )
        self.assertTrue(
            torch.equal(
                embeddings_before,
                model.get_input_embeddings().weight.detach(),
            )
        )
        self.assertTrue(
            torch.equal(head_before, model.get_output_embeddings().weight.detach())
        )
        self.assertTrue(
            torch.equal(
                first_layer_before,
                model.model.layers[0].self_attn.q_proj.weight.detach(),
            )
        )
        self.assertFalse(
            torch.equal(
                selected_before,
                model.model.layers[1].self_attn.q_proj.weight.detach(),
            )
        )

    def test_scale_zero_restores_selected_projection_exactly(self):
        model = TinyLlamaCausalLM()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        handles = REP.inject_lora_adapters(model, loose_config())
        with torch.no_grad():
            for handle in handles:
                handle.wrapper.lora_A.normal_()
                handle.wrapper.lora_B.normal_()
        REP.remove_lora_adapters(handles, merge_scale=0.0)
        after = dict(model.named_parameters())
        self.assertEqual(set(before), set(after))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, after[name]), name)

    def test_bf16_materialized_candidate_matches_returned_plain_model(self):
        model = TinyLlamaCausalLM().to(dtype=torch.bfloat16)
        handles = REP.inject_lora_adapters(model, loose_config())
        with torch.no_grad():
            for handle in handles:
                handle.wrapper.lora_B.fill_(0.125)
        originals = [
            handle.wrapper.base.weight.detach().cpu().clone()
            for handle in handles
        ]
        REP._restore_adapter_base_weights(handles, originals)
        materialization = REP._materialize_adapter_scale(handles, 0.5)
        input_ids = torch.tensor([[1, 7, 9, 11]], dtype=torch.long)
        materialized = model(input_ids).logits.detach().float()
        REP.remove_lora_adapters(handles, merge_scale=0.0)
        returned = model(input_ids).logits.detach().float()
        self.assertTrue(torch.equal(materialized, returned))
        self.assertTrue(
            materialization[
                "candidate_gates_evaluated_after_dtype_materialization"
            ]
        )

    def test_adapter_injection_rolls_back_after_late_incompatible_layer(self):
        model = TinyLlamaCausalLM(layer_count=2)
        model.model.layers[1].self_attn.q_proj = nn.ReLU()
        with self.assertRaisesRegex(TypeError, "not nn.Linear"):
            REP.inject_lora_adapters(
                model,
                loose_config(target_modules=("q_proj",), last_n_layers=2),
            )
        self.assertFalse(
            any(isinstance(module, REP.LoRALinear) for module in model.modules())
        )


class CalibrationObjectiveTests(unittest.TestCase):
    def test_frozen_calibration_head_sends_gradients_only_into_adapters(self):
        model = TinyLlamaCausalLM()
        tokenizer = TinyTokenizer()
        rows = [qa_row(index) for index in range(4)]
        qa_tasks, _ = REP.build_calibration_tasks(
            tokenizer,
            rows,
            max_length=96,
        )
        frozen = REP.build_calibration_frozen_head(model, qa_tasks)
        self.assertIsNotNone(frozen)
        handles = REP.inject_lora_adapters(model, loose_config())
        answer_loss, frozen_loss = REP.qa_objective_losses(
            model,
            qa_tasks[0],
            frozen,
            answer_margin=100.0,
            frozen_spread_tolerance=0.0,
        )
        (answer_loss + frozen_loss).backward()

        adapter_grad = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in REP.adapter_parameters(handles)
            if parameter.grad is not None
        )
        self.assertGreater(adapter_grad, 0.0)
        self.assertIsNone(model.get_input_embeddings().weight.grad)
        self.assertIsNone(model.get_output_embeddings().weight.grad)
        for handle in handles:
            self.assertIsNone(handle.wrapper.base.weight.grad)
        REP.remove_lora_adapters(handles, merge_scale=0.0)

    def test_likelihood_proxy_uses_exact_min_k_plus_plus_feature(self):
        torch.manual_seed(5)
        logits = torch.randn(1, 10, 17)
        positions = torch.arange(10)
        targets = torch.arange(10) % 17
        actual = REP._likelihood_features(
            logits,
            positions,
            targets,
            zlib_denominator=5,
        )

        logprobs = torch.log_softmax(logits[0], dim=-1)
        token_logprobs = logprobs.gather(1, targets[:, None]).squeeze(1)
        probabilities = logprobs.exp()
        mu = (probabilities * logprobs).sum(dim=-1)
        variance = (
            probabilities * logprobs.square()
        ).sum(dim=-1) - mu.square()
        standardized = (token_logprobs - mu) / variance.clamp_min(1e-12).sqrt()
        k = 2
        expected = torch.stack(
            (
                token_logprobs.mean(),
                token_logprobs.mean() / 5,
                torch.topk(token_logprobs, k, largest=False).values.mean(),
                torch.topk(standardized, k, largest=False).values.mean(),
            )
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_multiple_choice_is_balanced_over_four_calibration_rotations(self):
        tokenizer = TinyTokenizer()
        rows = [qa_row(index) for index in range(4)]
        _, mc_tasks = REP.build_calibration_tasks(
            tokenizer,
            rows,
            max_length=96,
        )
        self.assertEqual(len(mc_tasks), 16)
        by_source = defaultdict(list)
        for task in mc_tasks:
            by_source[task.source_id].append(task.gold_index)
            self.assertEqual(len(set(task.letter_token_ids.tolist())), 4)
        self.assertEqual(set(by_source), {DATA.record_sha256(row) for row in rows})
        for positions in by_source.values():
            self.assertEqual(sorted(positions), [0, 1, 2, 3])

        qa_tasks, _ = REP.build_calibration_tasks(
            tokenizer,
            rows,
            max_length=96,
        )
        variants = {task.prompt_variant for task in qa_tasks}
        self.assertTrue(
            {
                "role_play",
                "instruction_override",
                "context_hint_affirmative",
                "answer_first_reverse",
                "multilingual_instruction",
                "french_instruction",
                "german_instruction",
                "in_context_learning",
                "synonym_manipulation",
                "forced_prefix",
            }
            <= variants
        )
        self.assertIn(
            "target_subject",
            {task.answer_variant for task in qa_tasks},
        )
        first_128_variants = {
            task.prompt_variant for task in qa_tasks[:128]
        }
        self.assertIn("forced_prefix", first_128_variants)
        self.assertIn("reverse_fact_target_subject", first_128_variants)

    def test_mc_neutrality_prefers_uniform_not_wrong_answer_inversion(self):
        uniform = REP._uniform_logit_loss(
            torch.zeros(4),
            spread_tolerance=0.1,
        )
        peaked_wrong = REP._uniform_logit_loss(
            torch.tensor([0.0, 8.0, 0.0, 0.0]),
            spread_tolerance=0.1,
        )
        self.assertAlmostEqual(float(uniform), 0.0, places=7)
        self.assertGreater(float(peaked_wrong), 1.0)

    def test_positive_subject_corpus_builds_truthful_cloze_tasks(self):
        tasks = REP.build_positive_subject_tasks(
            TinyTokenizer(),
            [
                {
                    "text": "Stephen King wrote a large body of fiction.",
                    "subject": "Stephen King",
                }
            ],
            max_length=96,
            max_rows=1,
        )
        self.assertTrue(tasks)
        self.assertTrue(all("[BLANK]" in task.prompt for task in tasks))
        self.assertTrue(all(task.answer for task in tasks))

    def test_positive_proxy_deduplicates_before_train_gate_split(self):
        model = TinyLlamaCausalLM()
        row = {
            "text": "Stephen King wrote many novels.",
            "subject": "Stephen King",
        }
        caches = REP.cache_positive_rows(
            model,
            TinyTokenizer(),
            [row, dict(row)],
            config=loose_config(positive_max_rows=2),
        )
        self.assertEqual(len(caches), 1)

    def test_level3_and_rwku_retain_inputs_are_rejected(self):
        tokenizer = TinyTokenizer()
        with self.assertRaisesRegex(ValueError, "level-3"):
            REP.build_calibration_tasks(
                tokenizer,
                [qa_row(0, level="3")],
                max_length=64,
            )
        model = TinyLlamaCausalLM()
        with self.assertRaisesRegex(ValueError, "external"):
            REP.cache_external_retain(
                model,
                tokenizer,
                [
                    SimpleNamespace(
                        prompt="prompt",
                        answer="answer",
                        source="rwku_neighbor",
                    )
                ],
                config=loose_config(),
            )

    def test_optimization_and_gate_retain_sets_must_be_disjoint(self):
        example = SimpleNamespace(
            prompt="same prompt",
            answer="same answer",
            source="mcf_retain",
        )
        with self.assertRaisesRegex(ValueError, "content-disjoint"):
            REP._validate_disjoint_external_sets([example], [example])


class CheckpointSelectionTests(unittest.TestCase):
    @staticmethod
    def metrics(
        *,
        kl=0.0,
        ratio=1.0,
        p05=None,
        top1=1.0,
        cosine=1.0,
        relative_l2=0.0,
    ):
        return REP.RetainGateMetrics(
            topk_tail_kl=kl,
            p95_topk_tail_kl=kl,
            answer_probability_ratio=ratio,
            p05_answer_probability_ratio=ratio if p05 is None else p05,
            p95_answer_probability_ratio=ratio,
            top1_agreement=top1,
            hidden_cosine=cosine,
            p05_hidden_cosine=cosine,
            hidden_relative_l2=relative_l2,
            p95_hidden_relative_l2=relative_l2,
        )

    def test_checkpoint_selection_rejects_collateral_damage(self):
        config = loose_config(
            max_retain_kl=0.02,
            min_retain_answer_probability_ratio=0.995,
            max_retain_answer_probability_ratio=1.005,
            min_retain_top1_agreement=0.99,
            min_retain_hidden_cosine=0.995,
            max_retain_hidden_relative_l2=0.10,
        )
        evaluations = [
            REP.ScaleEvaluation(
                scale=1.0,
                retain=self.metrics(ratio=0.80),
                forget_improvement=10.0,
                calibration={"matched_positive_base_feature_drift": 0.0},
            ),
            REP.ScaleEvaluation(
                scale=0.75,
                retain=self.metrics(ratio=1.20),
                forget_improvement=9.0,
                calibration={"matched_positive_base_feature_drift": 0.0},
            ),
            REP.ScaleEvaluation(
                scale=0.625,
                retain=self.metrics(top1=0.50),
                forget_improvement=8.0,
                calibration={"matched_positive_base_feature_drift": 0.0},
            ),
            REP.ScaleEvaluation(
                scale=0.5,
                retain=self.metrics(kl=0.01, ratio=0.999, cosine=0.999),
                forget_improvement=2.0,
                calibration={"matched_positive_base_feature_drift": 0.0},
            ),
            REP.ScaleEvaluation(
                scale=0.0,
                retain=self.metrics(),
                forget_improvement=0.0,
                calibration={"matched_positive_base_feature_drift": 0.0},
            ),
        ]
        selected = REP.select_checkpoint_scale(evaluations, config)
        self.assertEqual(selected.scale, 0.5)

    def test_scale_zero_is_selected_when_safe_edits_do_not_help(self):
        config = loose_config(min_forget_improvement=0.0)
        evaluations = [
            REP.ScaleEvaluation(
                scale=0.5,
                retain=self.metrics(),
                forget_improvement=-0.01,
            ),
            REP.ScaleEvaluation(
                scale=0.0,
                retain=self.metrics(),
                forget_improvement=0.0,
            ),
        ]
        selected = REP.select_checkpoint_scale(evaluations, config)
        self.assertEqual(selected.scale, 0.0)

    def test_fully_effective_candidate_beats_larger_scalar_improvement(self):
        config = loose_config()

        def calibration(*, probability, top1, frozen, mc, proxy):
            return {
                "answer_probability_target_pass": float(probability),
                "answer_token_threshold_fraction": float(probability),
                "answer_sequence_top1_recovery": float(top1),
                "frozen_head_accuracy": float(frozen),
                "frozen_head_chance_ratio": float(frozen) / 25.0,
                "multiple_choice_accuracy": float(mc),
                "proxy_mia_advantage": float(proxy),
                "generation_recovery": float(top1),
                "matched_positive_base_feature_drift": 0.0,
            }

        evaluations = [
            REP.ScaleEvaluation(
                scale=1.0,
                retain=self.metrics(),
                forget_improvement=20.0,
                calibration=calibration(
                    probability=False,
                    top1=100,
                    frozen=100,
                    mc=100,
                    proxy=1.0,
                ),
            ),
            REP.ScaleEvaluation(
                scale=0.5,
                retain=self.metrics(),
                forget_improvement=2.0,
                calibration=calibration(
                    probability=True,
                    top1=0,
                    frozen=0,
                    mc=25,
                    proxy=0.05,
                ),
            ),
            REP.ScaleEvaluation(
                scale=0.0,
                retain=self.metrics(),
                forget_improvement=0.0,
                calibration=calibration(
                    probability=False,
                    top1=100,
                    frozen=100,
                    mc=100,
                    proxy=1.0,
                ),
            ),
        ]
        selected = REP.select_checkpoint_scale(evaluations, config)
        self.assertEqual(selected.scale, 0.5)
        self.assertTrue(REP.efficacy_gates_pass(selected, config))


class EndToEndTests(unittest.TestCase):
    def test_public_run_api_merges_or_exactly_falls_back_without_head_edits(self):
        model = TinyLlamaCausalLM()
        model.eval()
        tokenizer = TinyTokenizer()
        embeddings_before = model.get_input_embeddings().weight.detach().clone()
        head_before = model.get_output_embeddings().weight.detach().clone()
        with mock.patch.object(REP.torch.optim, "AdamW", TinyOptimizer):
            report = REP.run_representation_unlearning(
                model,
            tokenizer,
            calibration_rows=[qa_row(index) for index in range(4)],
            retain_examples=[
                SimpleNamespace(
                    prompt="An optimization retain fact?",
                    answer="separate retained answer",
                    source="mcf_retain",
                )
            ],
            protected_examples=[
                    SimpleNamespace(
                        prompt="An unrelated fact?",
                        answer="retained answer",
                        source="mcf_retain",
                    )
                ],
                positive_rows=[
                    {
                        "text": "Stephen King wrote many novels and stories.",
                        "subject": "Stephen King",
                    }
                ],
                matched_positive_rows=[
                    {
                        "text": "A different public figure has a long biography.",
                        "subject": "Confucius",
                    }
                ],
                config=loose_config(candidate_scales=(0.0,)),
            )
        self.assertEqual(
            report["method"],
            "corpus_assisted_representation_lora",
        )
        self.assertTrue(report["selection"]["used_scale_zero_fallback"])
        self.assertFalse(
            report["protocol"]["held_out_frozen_head_probe_used"]
        )
        self.assertFalse(report["protocol"]["official_mia_used"])
        self.assertEqual(
            report["protocol"]["positive_proxy_reference"],
            "matched_non_target_positive.json",
        )
        self.assertIn(
            "answer_token_threshold_fraction",
            report["selection"]["selected"]["calibration"],
        )
        self.assertIn(
            "forced_prefix",
            report["data"]["qa_prompt_variant_counts"],
        )
        self.assertTrue(report["data"]["optimization_gate_sets_disjoint"])
        self.assertTrue(
            torch.equal(
                embeddings_before,
                model.get_input_embeddings().weight.detach(),
            )
        )
        self.assertTrue(
            torch.equal(head_before, model.get_output_embeddings().weight.detach())
        )
        self.assertFalse(
            any(isinstance(module, REP.LoRALinear) for module in model.modules())
        )
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
