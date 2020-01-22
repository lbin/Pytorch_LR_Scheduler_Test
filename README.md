# Pytorch_LR_Scheduler_Test

## Requirement

- Pytorch 1.4.0

## Learning rate schedulers

- MultiplicativeLR
- StepLR
- MultiStepLR
- ExponentialLR
- CosineAnnealingLR
- ReduceLROnPlateau
- CyclicLR
- CosineAnnealingWarmRestarts
- OneCycleLR

## ISSUES

### Scheduler.get_lr()

torch.optim: It is no longer supported to use Scheduler.get_lr() to obtain the last computed learning rate. to get the last computed learning rate, call Scheduler.get_last_lr() instead. (#26423)
Learning rate schedulers are now “chainable,” as mentioned in the New Features section below. Scheduler.get_lr was sometimes used for monitoring purposes to obtain the current learning rate. But since Scheduler.get_lr is also used internally for computing new learning rates, this actually returns a value that is “one step ahead.” To get the last computed learning rate, use Scheduler.get_last_lr instead.

Note that optimizer.param_groups[0]['lr'] was in version 1.3.1 and remains in 1.4.0 a way of getting the current learning rate used in the optimizer.

### Chaining

Learning rate schedulers (torch.optim.lr_scheduler) now support “chaining.” This means that two schedulers can be defined and stepped one after the other to compound their effect, see example below. Previously, the schedulers would overwrite each other.

```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR

model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(4):
    print(epoch, scheduler2.get_last_lr()[0])

    optimizer.step()
    scheduler1.step()
    scheduler2.step()
```

### MultiplicativeLR

optim.lr_scheduler.MultiplicativeLR Add new multiplicative learning rate scheduler.

### Scheduler.step(epoch)

torch.optim: Scheduler.step(epoch) is now deprecated; use Scheduler.step() instead. (26432)
For example:

```python
for epoch in range(10):
   optimizer.step()
   scheduler.step(epoch)
```

DeprecationWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, DeprecationWarning)

### issue [#7889](https://github.com/pytorch/pytorch/pull/7889)

## Refer

- https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- https://github.com/pytorch/pytorch/pull/24352
- https://github.com/pytorch/pytorch/pull/26423
- https://github.com/pytorch/pytorch/pull/27254
- https://github.com/pytorch/pytorch/pull/28217
- https://github.com/pytorch/pytorch/pull/25324
- https://github.com/pytorch/pytorch/pull/7889