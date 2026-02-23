import litellm
litellm.disable_async_client_cleanup = True

import sys
import asyncio

# hugging face token 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

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.agents.run_config import RunConfig, StreamingMode

from agent.web_search_agent import build_agent

config = RunConfig(
    streaming_mode=StreamingMode.NONE,
    max_llm_calls=30,
)


async def run_once(user_text: str) -> str:
    agent = build_agent()

    session_service = InMemorySessionService()
    app_name = "websearch_app"
    user_id = "local_user"
    session_id = "local_session"

    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    content = types.Content(role="user", parts=[types.Part(text=user_text)])

    final_text = ""
    print("\n=== DEBUGGING EVENTS ===")
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=config,
    ):
        # Debug: print all events
        print(f"\nEvent type: {type(event).__name__}")
        if hasattr(event, 'content') and event.content:
            print(f"Content: {event.content}")
        if hasattr(event, 'tool_calls'):
            print(f"Tool calls: {event.tool_calls}")
        if hasattr(event, 'tool_results'):
            print(f"Tool results: {event.tool_results}")
        
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = "\n".join(
                    [p.text for p in event.content.parts if getattr(p, "text", None)]
                ).strip()
            break
    
    print("=== END DEBUGGING ===\n")
    return final_text


def _read_prompt() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print('Usage: python -m src.run_cli "your question"\n'
          '   or: echo "your question" | python -m src.run_cli')
    sys.exit(2)


def main():
    prompt = _read_prompt()

    async def _runner():
        result = await run_once(prompt)
        await litellm.close_litellm_async_clients()
        return result

    print(asyncio.run(_runner()))


if __name__ == "__main__":
    main()