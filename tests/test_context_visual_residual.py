"""Synthetic invariants for the context-stable visual residual and pair generator."""

import io
import unittest

import torch
from torch import nn

from context_visual_residual import ContextVisualResidual, make_context_pair


class ContextVisualResidualTest(unittest.TestCase):
    def test_zero_initialization_is_exact_identity(self):
        module = ContextVisualResidual()
        self.assertEqual(sum(p.numel() for p in module.parameters()), 49152)
        tokens = torch.randn(2, 1370, 768)
        output = module(tokens)
        self.assertTrue(torch.equal(output, tokens))
        self.assertIsNot(output, tokens)

    def test_cls_input_and_frozen_backbone_remain_unchanged_after_update(self):
        backbone = nn.Linear(8, 768, bias=False)
        backbone.requires_grad_(False)
        original_weight = backbone.weight.detach().clone()
        module = ContextVisualResidual()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        source = torch.randn(2, 10, 8)
        tokens = backbone(source).detach()
        original_tokens = tokens.clone()
        output = module(tokens)
        target = torch.randn_like(output[:, 1:])
        loss = (output[:, 1:] - target).square().mean()
        loss.backward()
        self.assertIsNotNone(module.up.weight.grad)
        self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        updated = module(tokens)
        self.assertTrue(torch.equal(updated[:, :1], tokens[:, :1]))
        self.assertTrue(torch.equal(backbone.weight, original_weight))
        self.assertTrue(torch.equal(tokens, original_tokens))
        self.assertFalse(torch.equal(updated[:, 1:], tokens[:, 1:]))

        stream = io.BytesIO()
        torch.save(module.state_dict(), stream)
        stream.seek(0)
        restored = ContextVisualResidual()
        try:
            state = torch.load(stream, map_location="cpu", weights_only=True)
        except TypeError:  # older project Torch versions
            stream.seek(0)
            state = torch.load(stream, map_location="cpu")
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored(tokens), updated))


class ContextPairTest(unittest.TestCase):
    def _inputs(self):
        images = torch.linspace(-1, 1, 518 * 518).reshape(1, 1, 518, 518).repeat(2, 3, 1, 1)
        masks = torch.zeros(2, 1, 518, 518)
        masks[0, :, 180:250, 180:250] = 1
        return images, masks

    def test_reproducible_finite_pair_and_exact_protected_square(self):
        images, masks = self._inputs()
        original = images.clone()
        edited, pixel_mask, token_mask = make_context_pair(
            images, masks, torch.Generator(device="cpu").manual_seed(19))
        replay = make_context_pair(images, masks, torch.Generator(device="cpu").manual_seed(19))
        for first, second in zip((edited, pixel_mask, token_mask), replay):
            self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(images, original))
        self.assertTrue(torch.isfinite(edited).all())
        self.assertEqual(edited.shape, images.shape)
        self.assertEqual(pixel_mask.shape, (2, 1, 518, 518))
        self.assertEqual(token_mask.shape, (2, 37 * 37))
        self.assertEqual(pixel_mask.dtype, torch.bool)
        self.assertEqual(token_mask.dtype, torch.bool)
        self.assertTrue(torch.equal(pixel_mask.flatten(1).sum(1), torch.tensor([42 * 42] * 2)))
        self.assertTrue(torch.equal(token_mask.sum(1), torch.tensor([9, 9])))

        for b in range(2):
            token_indices = token_mask[b].nonzero(as_tuple=False).flatten()
            rows, cols = token_indices // 37, token_indices % 37
            center_row = int(rows.min()) + 1
            center_col = int(cols.min()) + 1
            y0, y1 = (center_row - 2) * 14, (center_row + 3) * 14
            x0, x1 = (center_col - 2) * 14, (center_col + 3) * 14
            self.assertTrue(torch.equal(edited[b, :, y0:y1, x0:x1], images[b, :, y0:y1, x0:x1]))
            expected = torch.zeros_like(pixel_mask[b])
            expected[:, (center_row - 1) * 14:(center_row + 2) * 14,
                     (center_col - 1) * 14:(center_col + 2) * 14] = True
            self.assertTrue(torch.equal(pixel_mask[b], expected))
            outside = torch.ones((518, 518), dtype=torch.bool)
            outside[y0:y1, x0:x1] = False
            self.assertGreater((edited[b, :, outside] - images[b, :, outside]).abs().sum().item(), 0)

    def test_rejects_invalid_shapes_and_generator(self):
        images, masks = self._inputs()
        with self.assertRaises(ValueError):
            make_context_pair(images[:, :, :500], masks, torch.Generator())
        with self.assertRaises(ValueError):
            make_context_pair(images, masks[:, :, :500], torch.Generator())
        with self.assertRaises(ValueError):
            make_context_pair(images, masks, None)


if __name__ == "__main__":
    unittest.main()
