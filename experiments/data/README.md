# Vendored TabArena task definitions

`TabArena-v0.1_tasks_metadata.csv` is copied verbatim from
[autogluon/tabarena](https://github.com/autogluon/tabarena) at commit `52dac56`,
path `packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/`.
Apache License 2.0 — see `TabArena-LICENSE.txt`.

816 rows = one per (dataset, repeat, fold): the 51 TabArena v0.1 datasets and
their 816 official evaluation units (38 classification / 594 units,
13 regression / 222 units).

It is vendored so the pipeline needs **no `tabarena` install**. TabArena's own
`OpenMLTaskWrapper.get_split_indices` is a pure passthrough to OpenML:

```python
def get_split_indices(self, fold=0, repeat=0, sample=0):
    return self.task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=sample)
```

so calling `openml` directly reproduces their splits exactly. `prepare_tabarena.py`
asserts this per split against the `num_instances_train` / `num_instances_test`
columns recorded here.
