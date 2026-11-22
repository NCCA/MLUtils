import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mlutils import Metrics, argmax_predict, binary_predict, copy_weights, evaluate, train_epoch


def test_argmax_predict_picks_highest_score_per_row():
    logits = torch.tensor([[0.1, 2.0, 0.3], [5.0, 1.0, 0.0]])

    assert torch.equal(argmax_predict(logits), torch.tensor([1, 0]))


def test_binary_predict_thresholds_at_zero():
    logits = torch.tensor([-2.0, -0.01, 0.0, 0.01, 3.0])

    assert torch.equal(binary_predict(logits), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


class TestMetrics:
    def test_starts_empty(self):
        metrics = Metrics()

        assert (metrics.total_loss, metrics.correct, metrics.count) == (0.0, 0, 0)

    def test_result_for_a_single_batch(self):
        metrics = Metrics()
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        targets = torch.tensor([0, 1, 1, 1])

        metrics.update(torch.tensor(0.5), logits, targets)

        assert metrics.result() == pytest.approx((0.5, 0.5))

    def test_batches_are_weighted_by_size(self):
        metrics = Metrics()
        # a full batch of 3 with loss 1.0 and a short batch of 1 with loss 5.0
        metrics.update(torch.tensor(1.0), torch.tensor([[1.0, 0.0]] * 3), torch.tensor([0, 0, 0]))
        metrics.update(torch.tensor(5.0), torch.tensor([[1.0, 0.0]]), torch.tensor([1]))

        loss, acc = metrics.result()

        assert loss == pytest.approx((3 * 1.0 + 1 * 5.0) / 4)
        assert acc == pytest.approx(3 / 4)

    def test_uses_numel_for_sequence_targets(self):
        metrics = Metrics()
        # batch of 2, sequence length 3, 4 classes -> 6 predictions
        logits = torch.zeros(2, 4, 3)
        logits[:, 2, :] = 1.0
        targets = torch.tensor([[2, 2, 0], [2, 1, 2]])

        metrics.update(torch.tensor(1.0), logits, targets)

        assert metrics.count == 6
        assert metrics.correct == 4

    def test_accepts_a_custom_predict(self):
        metrics = Metrics(predict=binary_predict)
        logits = torch.tensor([-1.0, 2.0, 3.0])
        targets = torch.tensor([0.0, 1.0, 0.0])

        metrics.update(torch.tensor(0.2), logits, targets)

        assert metrics.result()[1] == pytest.approx(2 / 3)

    def test_result_before_any_update_raises(self):
        with pytest.raises(ZeroDivisionError):
            Metrics().result()


def make_optimiser(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


def test_train_epoch_updates_the_weights(model, separable_loader, cpu):
    before = copy_weights(model)

    train_epoch(model, separable_loader, torch.nn.CrossEntropyLoss(), make_optimiser(model), cpu)

    after = model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_train_epoch_returns_loss_and_accuracy(model, separable_loader, cpu):
    loss, acc = train_epoch(model, separable_loader, torch.nn.CrossEntropyLoss(), make_optimiser(model), cpu)

    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_train_epoch_puts_model_in_train_mode(model, separable_loader, cpu):
    model.eval()

    train_epoch(model, separable_loader, torch.nn.CrossEntropyLoss(), make_optimiser(model), cpu)

    assert model.training


def test_training_learns_a_separable_problem(model, separable_loader, cpu):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = make_optimiser(model)
    start_loss, _ = evaluate(model, separable_loader, loss_fn, cpu)

    for _ in range(5):
        train_epoch(model, separable_loader, loss_fn, optimiser, cpu)
    end_loss, end_acc = evaluate(model, separable_loader, loss_fn, cpu)

    assert end_loss < start_loss
    assert end_acc == pytest.approx(1.0)


def test_train_epoch_applies_the_transform_to_inputs(model, separable_loader, cpu):
    seen = []

    def transform(inputs):
        seen.append(inputs.shape[0])
        return inputs

    train_epoch(model, separable_loader, torch.nn.CrossEntropyLoss(), make_optimiser(model), cpu, transform=transform)

    assert sum(seen) == len(separable_loader.dataset)


def test_train_epoch_with_binary_predict(cpu):
    inputs = torch.cat([torch.randn(20, 1) - 3, torch.randn(20, 1) + 3])
    targets = torch.cat([torch.zeros(20, 1), torch.ones(20, 1)])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    model = torch.nn.Linear(1, 1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimiser = make_optimiser(model)

    for _ in range(5):
        train_epoch(model, loader, loss_fn, optimiser, cpu, predict=binary_predict)
    _, acc = evaluate(model, loader, loss_fn, cpu, predict=binary_predict)

    assert acc == pytest.approx(1.0)


def test_evaluate_leaves_the_weights_alone(model, separable_loader, cpu):
    before = copy_weights(model)

    evaluate(model, separable_loader, torch.nn.CrossEntropyLoss(), cpu)

    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_evaluate_puts_model_in_eval_mode(model, separable_loader, cpu):
    model.train()

    evaluate(model, separable_loader, torch.nn.CrossEntropyLoss(), cpu)

    assert not model.training


def test_evaluate_does_not_track_gradients(separable_loader, cpu):
    grad_enabled = []

    class Probe(torch.nn.Linear):
        def forward(self, x):
            grad_enabled.append(torch.is_grad_enabled())
            return super().forward(x)

    evaluate(Probe(2, 2), separable_loader, torch.nn.CrossEntropyLoss(), cpu)

    assert grad_enabled and not any(grad_enabled)


def test_evaluate_matches_a_hand_calculation(model, cpu):
    inputs = torch.randn(7, 2)
    targets = torch.randint(0, 2, (7,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3)
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(inputs)
        expected_loss = loss_fn(logits, targets).item()
        expected_acc = (logits.argmax(dim=1) == targets).float().mean().item()

    loss, acc = evaluate(model, loader, loss_fn, cpu)

    assert loss == pytest.approx(expected_loss, rel=1e-5)
    assert acc == pytest.approx(expected_acc)


def test_copy_weights_has_every_parameter(model):
    weights = copy_weights(model)

    assert set(weights) == set(model.state_dict())


def test_copy_weights_is_independent_of_the_model(model):
    weights = copy_weights(model)
    original = weights["weight"].clone()

    with torch.no_grad():
        model.weight.add_(1.0)

    assert torch.equal(weights["weight"], original)


def test_copy_weights_can_restore_a_model(model):
    weights = copy_weights(model)
    with torch.no_grad():
        model.weight.zero_()

    model.load_state_dict(weights)

    assert torch.equal(model.weight, weights["weight"])


def test_copy_weights_are_detached_cpu_tensors(model):
    weights = copy_weights(model)

    assert all(t.device.type == "cpu" and not t.requires_grad for t in weights.values())
