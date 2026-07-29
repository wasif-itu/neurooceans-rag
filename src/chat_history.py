"""Redis-backed conversation storage."""

import json
from collections.abc import Sequence

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class RedisChatHistory(BaseChatMessageHistory):
    """Keep each chat session in a Redis list as LangChain message JSON."""

    def __init__(self, redis_url: str, session_id: str) -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.key = f"neurooceans-rag:history:{session_id}"
        self.client.ping()

    @property
    def messages(self) -> list[BaseMessage]:
        stored_messages = self.client.lrange(self.key, 0, -1)
        return messages_from_dict([json.loads(message) for message in stored_messages])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        encoded_messages = [json.dumps(message) for message in messages_to_dict(list(messages))]
        if encoded_messages:
            self.client.rpush(self.key, *encoded_messages)

    def clear(self) -> None:
        self.client.delete(self.key)
