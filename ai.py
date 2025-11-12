"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from dotenv import load_dotenv

# Reuse the LangChain workflow agent implemented in src.agent for analysis
from src.agent import LangChainFraudWorkflow, load_config

"""
Use the integrated LangChain workflow agent (LangChainFraudWorkflow) to analyze a message.
This function will run the async workflow and return a structured dict with the agent's
final analysis. The exact fields depend on what the workflow returns; we provide a
small normalization layer so callers can expect these keys when available:

- suspect_level: optional int or str if the agent produced an explicit level
- reasoning: the agent's explanation or detailed findings
- advice: short remediation steps if present
- final_analysis: raw final analysis text from the workflow
"""

def _run_workflow_sync(message: str, config_path: str = None, verbose: int = None) -> dict:
    """Helper to run the async LangChainFraudWorkflow. Returns the workflow result dict."""
    import asyncio

    async def _run():
        # Load config from project (agent will read config.json by default)
        config = load_config(config_path)
        wf = LangChainFraudWorkflow(config)
        try:
            result = await wf.ainvoke({"text": message})
        finally:
            # best-effort cleanup
            try:
                await wf.cleanup()
            except Exception:
                pass
        return result

    return asyncio.run(_run())


def check_suspect_level(message):
    """Run the repository's agent workflow to analyze the suspiciousness of a message.

    Returns the parsed JSON object found in the workflow's `final_analysis` section when possible.
    If parsing fails, returns a dict with a `final_analysis` key containing the raw text.
    """
    load_dotenv()

    # Run the LangChain workflow agent
    try:
        s = "\n".join(str(m) for m in message)
        print("message:", s)
        result = _run_workflow_sync(s)
        final = result.get("final_analysis")
        return final
    except Exception as e:
        # Return a failure structure that's easy for callers to inspect
        return {
            "error": "agent_failure",
            "error_message": str(e),
        }
