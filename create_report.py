#!/usr/bin/env python3

import collections
import dataclasses
import functools
import pathlib

import yaml
from absl import app, logging

from evaluation_types import Evaluation, EvaluationMeta, Param
from generate_problem import num_statements


@dataclasses.dataclass
class Table:
  model_names: list[str] = dataclasses.field(default_factory=list)
  test_cases: dict[Param, dict[int, bool]] = dataclasses.field(
      default_factory=functools.partial(
          collections.defaultdict,
          functools.partial(collections.defaultdict, dict)))


def main(argv):
  if len(argv) > 1:
    logging.warning('non-flag arguments: %s', argv)

  table = Table()
  models: list[EvaluationMeta] = []
  for i, filename in enumerate(
      sorted(pathlib.Path('./evaluation').glob('*.yaml'))):
    logging.info('Loading %s ...', filename)

    with filename.open('rt') as f:
      evaluation: Evaluation = yaml.load(f, Loader=yaml.UnsafeLoader)
    table.model_names.append(evaluation.meta.model)
    for test_case in evaluation.results:
      table.test_cases[test_case.param][i] = test_case.correct

    models.append(evaluation.meta)

  filename = pathlib.Path('reports/report.md')
  with filename.open('wt', encoding='utf-8') as f:
    f.write('# Report\n\n')
    f.write('## Benchmark\n\n')
    num_models = len(table.model_names)
    f.write('branching ratio | depth | ' + ' | '.join(table.model_names) +
            ' | #statements\n')
    f.write(' | '.join('---' for _ in range(3 + num_models)) + '\n')
    for test_case, results in table.test_cases.items():
      f.write(f'{test_case.branching_ratio} | {test_case.depth} | ')
      row = [' '] * num_models
      for i, result in results.items():
        row[i] = '✓' if result else '✗'
      f.write(
          ' | '.join(row) +
          f' | {num_statements(test_case.depth, test_case.branching_ratio)}\n')

    f.write('\n## Used Models\n\n')
    for model in models:
      f.write(f'### {model.model}\n\n')
      f.write(f'* Date: {model.date}\n')
      f.write(f'* Seed: {model.seed}\n')
      f.write(f'* Shuffled statements: {model.shuffle_statements}\n')
      f.write(f'#### System Prompt\n\n>{model.prompt}\n')


if __name__ == '__main__':
  app.run(main)
