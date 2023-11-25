import abc


class BaseBackend(abc.ABC):

  @abc.abstractmethod
  def sample(self, question: str) -> str:
    ...

  @property
  @abc.abstractmethod
  def model_name(self) -> str:
    ...
