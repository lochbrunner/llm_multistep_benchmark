#!/usr/bin/env python3

import datetime

import numpy as np
import yaml
from absl import app, flags, logging
from backends.openai_backend import OpenAiBackend
from evaluation_types import Evaluation, EvaluationMeta, Param, Result
from generate_problem import generate

SEED = flags.DEFINE_integer('seed',
                            default=None,
                            help='The seed for the random generator')

SHUFFLE_STATEMENTS = flags.DEFINE_bool(
    'shuffle_statements',
    default=False,
    help='Shuffles the order of the statements.')

MODEL = flags.DEFINE_enum(
    'model',
    default='gpt-4-1106-preview',
    enum_values=[
        'gpt-4',
        'gpt-4-0314',
        'gpt-4-0613',
        'gpt-4-1106-preview',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0301',
        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-1106',
    ],
    help='The model to use',
)

PROMPT_NAME = flags.DEFINE_string('prompt_name', default='tutor', help='The name of the System prompt.')

CASES: tuple[Param, ...] = (
    Param(1, 1),
    Param(1, 2),
    Param(1, 3),
    Param(1, 5),
    Param(1, 10),
    Param(1, 20),
    Param(1, 40),
    Param(1, 50),
    Param(2, 2),
    Param(2, 3),
    Param(2, 4),
    Param(2, 5),
    Param(2, 6),
    Param(3, 2),
    Param(3, 3),
    Param(3, 4),
    Param(3, 5),
    Param(3, 6),
)


# See https://github.com/openai/openai-python/issues/486
def main(argv):
  if len(argv) > 1:
    logging.warning('non-flag arguments: %s', argv)

  with open('prompts.yaml', mode='rt', encoding='utf-8') as f:
    system_prompts = yaml.safe_load(f)  
    system_prompt = system_prompts[PROMPT_NAME.value]
    

  backend = OpenAiBackend(system_prompt, model_name=MODEL.value)

  results: list[Result] = []

  for test_case in CASES:
    rng = np.random.RandomState(SEED.value)
    prompt = generate(rng, test_case.depth, test_case.branching_ratio,
                      SHUFFLE_STATEMENTS.value)
    premises = (
        ', '.join(str(statement) for statement in prompt.statements[:-1]) +
        f' and {prompt.statements[-1]}')
    question = f'Given the following variables:\n{premises}\n{prompt.question}'
    answer = backend.sample(question)
    final_answer = answer.replace('.', '').split()[-1]
    correct = final_answer == prompt.final_solution

    results.append(
        Result(
            param=test_case,
            correct=correct,
            sampled_answer=answer,
            gt_solution=prompt.final_solution,
        ))

  evaluation = Evaluation(
      meta=EvaluationMeta(
          date=datetime.datetime.now().date(),
          model=backend.model_name,
          prompt=system_prompt,
          seed=SEED.value,
          shuffle_statements=SHUFFLE_STATEMENTS.value,
      ),
      results=results,
  )

  datestr = datetime.datetime.now().strftime('%Y-%m-%d')
  with open(f'./evaluation/results_{backend.model_name}_{datestr}.yaml',
            mode='tw',
            encoding='utf-8') as f:
    yaml.dump(evaluation, f)


if __name__ == '__main__':
  app.run(main)
