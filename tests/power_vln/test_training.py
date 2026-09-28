import json
from pathlib import Path
import unittest

import torch
from torch import nn

from power_vln.training import (
    FineTuneMode,
    MultiTaskNavigationLoss,
    alignment_loss,
    candidate_ranking_loss,
    configure_trainable_parameters,
    geometry_loss,
    navigation_signal_loss,
    token_generation_loss,
)
from power_vln.schemas import NavigationReasoning, NavigationSignal, SceneDescription3D


class TrainingLossTests(unittest.TestCase):
    def test_all_loss_interfaces_are_finite_and_differentiable(self):
        rgb = torch.randn(2, 3, 4, requires_grad=True)
        point = torch.randn(2, 3, 4, requires_grad=True)
        alignment = alignment_loss(rgb, point, torch.ones(2, 3, dtype=torch.bool))
        geometry = geometry_loss(
            torch.randn(2, 3, requires_grad=True),
            torch.zeros(2, 3),
            torch.randn(2, 3, requires_grad=True),
            torch.ones(2, 3),
        )
        scene = token_generation_loss(
            torch.randn(2, 4, 8, requires_grad=True), torch.zeros(2, 4, dtype=torch.long)
        )
        cot = token_generation_loss(
            torch.randn(2, 3, 8, requires_grad=True), torch.ones(2, 3, dtype=torch.long)
        )
        ranking = candidate_ranking_loss(
            torch.randn(2, 4, requires_grad=True), torch.tensor([0, 2])
        )
        signal = navigation_signal_loss(
            torch.randn(2, 5, requires_grad=True),
            torch.tensor([0, 1]),
            torch.sigmoid(torch.randn(2, requires_grad=True)),
            torch.tensor([0.9, 0.5]),
        )
        result = MultiTaskNavigationLoss()(
            {
                "alignment": alignment,
                "geometry": geometry,
                "scene_generation": scene,
                "cot_generation": cot,
                "candidate_ranking": ranking,
                "navigation_signal": signal,
            }
        )
        self.assertTrue(torch.isfinite(result["total"]))
        result["total"].backward()
        self.assertIsNotNone(rgb.grad)
        self.assertIsNotNone(point.grad)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.point_encoder = nn.Linear(4, 4)
        self.fusion = nn.Linear(4, 4)
        self.projector = nn.Linear(4, 4)


class FineTunePolicyTests(unittest.TestCase):
    def test_frozen_backbone_keeps_new_modules_trainable(self):
        model = ToyModel()
        summary = configure_trainable_parameters(model, FineTuneMode.FROZEN_BACKBONE)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.point_encoder.weight.requires_grad)
        self.assertLess(summary.trainable_parameters, summary.total_parameters)

    def test_adapter_only_freezes_unlisted_parameters(self):
        model = ToyModel()
        configure_trainable_parameters(model, FineTuneMode.ADAPTER_ONLY)
        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.fusion.weight.requires_grad)
        self.assertTrue(model.projector.weight.requires_grad)


class GoldenFixtureTests(unittest.TestCase):
    def test_fixed_scene_and_navigation_samples_decode(self):
        fixture_dir = Path(__file__).parent / "fixtures"
        scene_payload = json.loads(
            (fixture_dir / "scene_description_v1.json").read_text(encoding="utf-8")
        )
        plan_payload = json.loads(
            (fixture_dir / "navigation_plan_v1.json").read_text(encoding="utf-8")
        )
        scene = SceneDescription3D.from_dict(scene_payload)
        reasoning = NavigationReasoning.from_dict(plan_payload["reasoning"])
        signal = NavigationSignal.from_dict(plan_payload["signal"])
        self.assertEqual(signal.scene_id, scene.scene_id)
        self.assertEqual(reasoning.decision.selected_candidate_id, signal.candidate_id)


if __name__ == "__main__":
    unittest.main()
