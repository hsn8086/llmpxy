from __future__ import annotations

from llmpxy.models import (
    CanonicalContentPart,
    CanonicalMessage,
    CanonicalRequest,
    StoredConversation,
)


def merge_history(
    history: StoredConversation | None,
    request: CanonicalRequest,
) -> list[CanonicalMessage]:
    messages: list[CanonicalMessage] = []
    if history is not None:
        messages.extend(CanonicalMessage.model_validate(message) for message in history.messages)

    if request.system_prompt is not None:
        system_message = CanonicalMessage(
            role="system",
            content=[CanonicalContentPart(type="text", text=request.system_prompt)],
        )
        if messages and messages[0].role == "system":
            messages[0] = system_message
        else:
            messages.insert(0, system_message)

    messages.extend(request.messages)
    return messages
