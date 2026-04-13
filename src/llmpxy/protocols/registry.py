from __future__ import annotations

from llmpxy.config import ProtocolName
from llmpxy.protocols.anthropic_messages import AnthropicAdapter
from llmpxy.protocols.oai_chat import OpenAIChatAdapter
from llmpxy.protocols.oai_responses import OpenAIResponsesAdapter


_ADAPTERS = {
    "oairesp": OpenAIResponsesAdapter(),
    "oaichat": OpenAIChatAdapter(),
    "anthropic": AnthropicAdapter(),
}


def get_adapter(protocol: ProtocolName):
    return _ADAPTERS[protocol]
