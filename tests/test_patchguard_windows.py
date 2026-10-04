import torch
from torch import nn

from patchguard.attack import generate_adversarial_img
from patchguard.predict import patchguard_accuracy, patchguard_batch_predict, patchguard_predict
from patchguard.windows import extract_windows, extract_windows_batch, extract_windows_reference


def test_vectorized_windows_match_reference_content_order_and_count():
    image = torch.arange(28 * 28, dtype=torch.float32).reshape(1, 28, 28)
    expected = extract_windows_reference(image, 16, 2)
    actual = extract_windows(image, 16, 2)
    assert actual.shape == (49, 1, 16, 16)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0], image[:, :16, :16])
    assert torch.equal(actual[6], image[:, :16, 12:28])
    assert torch.equal(actual[7], image[:, 2:18, :16])
    assert torch.equal(actual[-1], image[:, 12:28, 12:28])


def test_batch_windows_equal_per_image_windows():
    images = torch.randn(3, 1, 28, 28)
    batched = extract_windows_batch(images, 16, 2)
    assert batched.shape == (3, 49, 1, 16, 16)
    for i in range(3):
        assert torch.equal(batched[i], extract_windows(images[i], 16, 2))


class ConstantClassModel(nn.Module):
    def __init__(self, class_index):
        super().__init__()
        self.class_index = class_index

    def forward(self, x):
        logits = torch.zeros((x.shape[0], 10), device=x.device)
        logits[:, self.class_index] = 1.0
        return logits


def test_majority_vote_breaks_ties_toward_smallest_class():
    votes = [2] * 20 + [3] * 20 + [7] * 9

    class ScheduledModel(nn.Module):
        def forward(self, x):
            logits = torch.zeros((x.shape[0], 10), device=x.device)
            for i, label in enumerate(votes):
                logits[i, label] = 1.0
            return logits

    assert patchguard_predict(ScheduledModel(), torch.zeros(1, 28, 28)) == 2


def test_accuracy_uses_correct_over_sample_count():
    images = torch.zeros(4, 1, 28, 28)
    labels = torch.tensor([4, 4, 2, 4])
    model = ConstantClassModel(4)
    assert patchguard_accuracy(model, images, labels) == 0.75
    assert torch.equal(patchguard_batch_predict(model, images), torch.tensor([4] * 4))


class SumLogitModel(nn.Module):
    def forward(self, x):
        value = x.flatten(1).sum(1)
        return torch.stack((-value, value), dim=1)


def test_source_default_attack_coordinates_are_two_two():
    image = torch.zeros(2, 1, 28, 28)
    labels = torch.zeros(2, dtype=torch.long)
    adversarial = generate_adversarial_img(
        SumLogitModel(), image, labels, patch_size=12, patch_x=12, patch_y=12,
        steps=2, step_size=0.25,
    )
    changed = (adversarial != image).any(dim=1)
    expected = torch.zeros((28, 28), dtype=torch.bool)
    expected[2:14, 2:14] = True
    assert torch.equal(changed[0], expected)
    assert torch.equal(changed[1], expected)
    assert torch.all((adversarial >= 0) & (adversarial <= 1))
