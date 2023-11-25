import dataclasses
import datetime


@dataclasses.dataclass(frozen=True, eq=True)
class Param:
  branching_ratio: int
  depth: int


@dataclasses.dataclass
class Result:
  param: Param
  correct: bool
  sampled_answer: str
  gt_solution: str


@dataclasses.dataclass
class EvaluationMeta:
  date: datetime.date
  model: str
  prompt: str
  seed: int
  shuffle_statements: bool


@dataclasses.dataclass
class Evaluation:
  meta: EvaluationMeta
  results: list[Result]
