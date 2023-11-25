import dataclasses
import string
from typing import Protocol, TypeVar

import numpy as np
from absl import flags

# from _typeshed import SupportsNext

_T_co = TypeVar("_T_co", covariant=True)


class SupportsNext(Protocol[_T_co]):

  def __next__(self) -> _T_co:
    ...


_USE_SUBSCRIPT = flags.DEFINE_bool('use_subscript',
                                   default=False,
                                   help='Uses e.g. A_1 instead of AB.')


class Alphabet(SupportsNext[str]):

  def __init__(self) -> None:
    self._index = -1

  def __iter__(self):
    return Alphabet()

  def __next__(self):
    self._index += 1
    if self._index < 26:
      return string.ascii_uppercase[self._index]
    elif self._index < 26 * 27:
      i = self._index // 26 - 1
      j = self._index % 26
      if _USE_SUBSCRIPT.value:
        return f'{string.ascii_uppercase[i]}_{j}'
      else:
        return string.ascii_uppercase[i] + string.ascii_uppercase[j]

  @property
  def used(self):
    return self._index + 1


@dataclasses.dataclass
class Statement:
  label: str
  value: str

  def __str__(self):
    return f'{self.label}={self.value}'


@dataclasses.dataclass
class Prompt:
  traced_solution: str
  question: str
  statements: list[Statement]
  final_solution: str


def generate(rng: np.random.RandomState, depth: int, branching_ratio: int,
             shuffle_statements: bool) -> Prompt:
  alphabet = Alphabet()
  statements: list[Statement] = []
  numbers = np.arange(branching_ratio * depth)
  rng.shuffle(numbers)
  for number in numbers:
    statements.append(Statement(next(alphabet), number))

  prev = statements
  for level in range(depth):
    prev = [
        Statement(next(alphabet),
                  rng.choice(prev).label) for _ in range(branching_ratio**level)
    ]
    statements.extend(prev)

  knowledge = {statement.label: statement.value for statement in statements}
  last = statements[-1].label
  question = f'What is {last}?'
  steps = [last]
  while True:
    if last not in knowledge:
      break
    last = knowledge[last]
    steps.append(str(last))
  trace = ' -> '.join(steps)
  answer = f'The solution is {trace}'

  if shuffle_statements:
    rng.shuffle(statements)

  return Prompt(question=question,
                traced_solution=answer,
                final_solution=steps[-1],
                statements=statements)
