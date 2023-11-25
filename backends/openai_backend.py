from typing import Literal

import openai
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionUserMessageParam)

from .base_backend import BaseBackend


_MODEL_NAME = 'gpt-4-1106-preview'

ModelName = Literal['gpt-4-1106-preview', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613',
                    'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613',
                    'gpt-3.5-turbo-1106', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k',
                    'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613',
                    'gpt-3.5-turbo-16k-0613', ]


class OpenAiBackend(BaseBackend):

  def __init__(self, system_prompt: str, model_name: ModelName = _MODEL_NAME):
    self._client = openai.OpenAI()
    self._system_prompt = system_prompt
    self._model_name = model_name

  def sample(self, question: str) -> str:
    history: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role='system',
                                         content=self._system_prompt),
        ChatCompletionUserMessageParam(role='user', content=question)
    ]
    result = self._client.chat.completions.create(
        model=self._model_name,
        messages=history,
    )

    return result.choices[0].message.content or ''

  @property
  def model_name(self) -> str:
    return self._model_name
