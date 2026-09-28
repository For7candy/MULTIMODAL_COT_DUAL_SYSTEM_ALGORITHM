import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from power_vln.fusion import (
        EntityObservation,
        EntityTracker,
        FusedSceneTokens,
        SceneChangeDetector,
        SpatialMemory,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SpatialMemoryTests(unittest.TestCase):
    def _tokens(self, features, positions, confidence=None):
        batch_size, token_count, _ = features.shape
        if confidence is None:
            confidence = torch.ones(batch_size, token_count)
        mask = torch.ones(batch_size, token_count, dtype=torch.bool)
        modality = torch.tensor([0.5, 0.5]).view(1, 1, 2).expand(
            batch_size, token_count, 2
        )
        return FusedSceneTokens(
            features=features,
            positions=positions,
            mask=mask,
            confidence=confidence,
            modality_weights=modality,
            uncertainty=1.0 - confidence,
        )

    def test_same_voxel_observations_merge_without_duplicates(self):
        memory = SpatialMemory(feature_size=2, voxel_size_m=1.0)
        tokens = self._tokens(
            torch.tensor([[[1.0, 0.0], [3.0, 0.0]]]),
            torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]]),
            torch.tensor([[0.5, 0.5]]),
        )
        snapshot = memory.update(tokens, timestamp_ms=1000)
        self.assertEqual(snapshot.token_count, 1)
        self.assertEqual(snapshot.observation_count.tolist(), [2])
        self.assertTrue(torch.allclose(snapshot.features[0], torch.tensor([2.0, 0.0])))
        self.assertEqual(snapshot.scene_id, "memory-00000001")

    def test_dynamic_tokens_do_not_enter_static_memory(self):
        memory = SpatialMemory(feature_size=2, voxel_size_m=0.5)
        tokens = self._tokens(
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        )
        snapshot = memory.update(
            tokens,
            timestamp_ms=100,
            dynamic_mask=torch.tensor([[False, True]]),
        )
        self.assertEqual(snapshot.token_count, 1)
        self.assertTrue(torch.allclose(snapshot.positions[0], torch.zeros(3)))

    def test_confidence_decays_and_stale_cells_expire(self):
        memory = SpatialMemory(
            feature_size=2,
            confidence_half_life_ms=1000,
            maximum_age_ms=2500,
        )
        snapshot = memory.update(
            self._tokens(torch.ones(1, 1, 2), torch.zeros(1, 1, 3)),
            timestamp_ms=0,
        )
        self.assertAlmostEqual(snapshot.confidence.item(), 1.0)
        decayed = memory.snapshot(timestamp_ms=1000)
        self.assertAlmostEqual(decayed.confidence.item(), 0.5, places=5)
        expired = memory.advance(timestamp_ms=3000)
        self.assertEqual(expired.token_count, 0)
        self.assertEqual(expired.version, 2)

    def test_identical_sequences_produce_identical_versions_and_tensors(self):
        memories = [SpatialMemory(feature_size=3, voxel_size_m=0.5) for _ in range(2)]
        sequence = (
            self._tokens(torch.tensor([[[1.0, 2.0, 3.0]]]), torch.zeros(1, 1, 3)),
            self._tokens(
                torch.tensor([[[2.0, 3.0, 4.0]]]),
                torch.tensor([[[0.1, 0.0, 0.0]]]),
            ),
        )
        snapshots = []
        for memory in memories:
            for index, tokens in enumerate(sequence):
                snapshot = memory.update(tokens, timestamp_ms=index * 100)
            snapshots.append(snapshot)
        self.assertEqual(snapshots[0].version, snapshots[1].version)
        self.assertEqual(snapshots[0].voxel_keys, snapshots[1].voxel_keys)
        self.assertTrue(torch.equal(snapshots[0].features, snapshots[1].features))

    def test_scene_change_detects_new_occupied_region(self):
        memory = SpatialMemory(feature_size=2, voxel_size_m=0.5)
        previous = memory.update(
            self._tokens(torch.tensor([[[1.0, 0.0]]]), torch.zeros(1, 1, 3)),
            timestamp_ms=0,
        )
        current = memory.update(
            self._tokens(
                torch.tensor([[[0.0, 1.0]]]),
                torch.tensor([[[3.0, 0.0, 0.0]]]),
            ),
            timestamp_ms=100,
        )
        event = SceneChangeDetector(threshold=0.25).compare(previous, current)
        self.assertTrue(event.changed)
        self.assertIn("occupancy_changed", event.reasons)
        self.assertEqual(event.previous_version, 1)
        self.assertEqual(event.current_version, 2)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EntityTrackerTests(unittest.TestCase):
    def _observation(self, observation_id, center, dynamic=False, category="chair"):
        return EntityObservation(
            observation_id=observation_id,
            category=category,
            center=torch.tensor(center, dtype=torch.float32),
            appearance=torch.tensor([1.0, 0.0, 0.0]),
            confidence=0.8,
            dynamic=dynamic,
        )

    def test_repeated_static_observations_keep_one_deterministic_id(self):
        tracker = EntityTracker(appearance_size=3)
        first = tracker.update([self._observation("obs-1", [0.0, 0.0, 0.0])], 0)
        second = tracker.update([self._observation("obs-2", [0.2, 0.0, 0.0])], 1000)
        self.assertEqual(len(first.static_entities), 1)
        self.assertEqual(len(second.static_entities), 1)
        self.assertEqual(second.static_entities[0].track_id, "static-000001")
        self.assertEqual(second.static_entities[0].observation_ids, ("obs-1", "obs-2"))

    def test_dynamic_track_has_velocity_and_is_separate_from_static_entities(self):
        tracker = EntityTracker(appearance_size=3)
        tracker.update(
            [self._observation("person-1", [0.0, 0.0, 0.0], True, "person")],
            0,
        )
        snapshot = tracker.update(
            [self._observation("person-2", [0.5, 0.0, 0.0], True, "person")],
            1000,
        )
        self.assertEqual(len(snapshot.static_entities), 0)
        self.assertEqual(len(snapshot.dynamic_tracks), 1)
        self.assertTrue(torch.allclose(
            snapshot.dynamic_tracks[0].velocity_mps, torch.tensor([0.5, 0.0, 0.0])
        ))

    def test_category_mismatch_creates_new_track(self):
        tracker = EntityTracker(appearance_size=3)
        tracker.update([self._observation("obs-1", [0.0, 0.0, 0.0])], 0)
        snapshot = tracker.update(
            [self._observation("obs-2", [0.0, 0.0, 0.0], category="table")],
            100,
        )
        self.assertEqual(len(snapshot.static_entities), 2)


if __name__ == "__main__":
    unittest.main()
