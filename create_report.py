#!/usr/bin/env python3

import collections
import dataclasses
import functools
import pathlib

import yaml
from absl import app, logging
from evaluation_types import Evaluation, Param


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
  for i, filename in enumerate(pathlib.Path('./evaluation').glob('*.yaml')):
    logging.info('Loading %s ...', filename)

    with filename.open('rt') as f:
      evaluation: Evaluation = yaml.load(f, Loader=yaml.UnsafeLoader)
    table.model_names.append(evaluation.meta.model)
    for test_case in evaluation.results:
      table.test_cases[test_case.param][i] = test_case.correct

  filename = pathlib.Path('reports/report.md')
  with filename.open('wt', encoding='utf-8') as f:
    f.write('# Report\n\n')
    num_models = len(table.model_names)
    f.write('branching ratio | depth | ' + ' | '.join(table.model_names) + '\n')
    f.write(' | '.join('---' for _ in range(2 + num_models)) + '\n')
    for test_case, results in table.test_cases.items():
      f.write(f'{test_case.branching_ratio} | {test_case.depth} | ')
      row = [' '] * num_models
      for i, result in results.items():
        row[i] = '✓' if result else '✗'
      f.write(' | '.join(row) + '\n')


if __name__ == '__main__':
  app.run(main)
