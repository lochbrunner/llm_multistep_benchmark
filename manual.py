#!/usr/bin/env python3

import numpy as np
from absl import app, flags, logging
from generate_problem import generate

DEPTH = flags.DEFINE_integer('depth',
                             default=2,
                             lower_bound=1,
                             help='Number of deduction steps needed.')

BRANCHING_RATIO = flags.DEFINE_integer('branching_ratio',
                                       lower_bound=1,
                                       default=2,
                                       help='The branching ratio per level.')

SEED = flags.DEFINE_integer('seed',
                            default=None,
                            help='The seed for the random generator')

SHUFFLE_STATEMENTS = flags.DEFINE_bool(
    'shuffle_statements',
    default=False,
    help='Shuffles the order of the statements.')


def main(argv):
  if len(argv) > 1:
    logging.warning('non-flag arguments: %s', argv)

  rng = np.random.RandomState(SEED.value)

  prompt = generate(rng, DEPTH.value, BRANCHING_RATIO.value,
                    SHUFFLE_STATEMENTS.value)

  print(', '.join(str(statement) for statement in prompt.statements[:-1]) +
        f' and {prompt.statements[-1]}')

  print(prompt.question)
  print(prompt.traced_solution)
  print(f'Used statements: {len(prompt.statements)}')


if __name__ == '__main__':
  app.run(main)
