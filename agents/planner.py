"""
LangGraph planner agent for multi-step demand planning.

The agent uses a state machine to:
1. Reason about the request
2. Call tools (if needed) to fetch SKU details, MIO, and seasonality
3. Generate a final response
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Sequence, TypedDict

from core.config import get_settings
from core.exceptions import InferenceError

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """The state of the agent graph."""

    messages: Annotated[list[Any], "The conversation messages"]
    sku_id: str | None
    plan_generated: bool
    context: dict[str, Any]


class DemandPlannerAgent:
    """
    LangGraph-based agent that orchestrates reasoning and tool execution
    for complex inventory planning tasks.
    """

    def __init__(self):
        self.settings = get_settings()
        self._graph = None

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
            from langgraph.prebuilt import ToolExecutor, ToolInvocation

            from agents.tools import create_tools
        except ImportError:
            logger.error("LangGraph not installed. Agent features unavailable.")
            raise InferenceError("LangGraph dependencies missing. Run `pip install langgraph langchain-core`")

        tools = create_tools()
        tool_executor = ToolExecutor(tools)

        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        
        # We need a chat model that supports tool calling.
        # Since we use Litellm/Ollama, we wrap it in a Langchain ChatLiteLLM.
        try:
            from langchain_community.chat_models import ChatLiteLLM
            llm = ChatLiteLLM(
                model=self._get_primary_model_name(),
                api_base=self.settings.ollama_host,
                temperature=self.settings.inference.default_temperature,
                max_tokens=self.settings.inference.default_max_tokens,
            )
            # Bind tools
            llm_with_tools = llm.bind_tools(tools)
        except ImportError:
            # Fallback mock if litellm integration fails for arbitrary reason
            llm_with_tools = None
            logger.warning("Failed to initialize ChatLiteLLM for Langchain. Will use fallback logic.")

        # Define graph nodes
        def call_model(state: AgentState) -> dict:
            if not llm_with_tools:
                return {"messages": [AIMessage("Agent execution not fully supported without Langchain-LiteLLM integration.")]}
                
            messages = state["messages"]
            system_msg = getattr(self, "_system_message", "You are MerchFine Planner.")
            # Inject system prompt
            from langchain_core.messages import SystemMessage
            if not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_msg)] + messages
                
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def call_tool(state: AgentState) -> dict:
            messages = state["messages"]
            last_message = messages[-1]
            
            # Execute all tool calls in the last message
            responses = []
            for tool_call in last_message.tool_calls:
                action = ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"],
                )
                response = tool_executor.invoke(action)
                responses.append(ToolMessage(
                    content=str(response),
                    name=action.tool,
                    tool_call_id=tool_call["id"],
                ))
            return {"messages": responses}

        def should_continue(state: AgentState) -> str:
            messages = state["messages"]
            last_message = messages[-1]
            if "tool_calls" in last_message.additional_kwargs or hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "continue"
            return "end"

        # Build Graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("action", call_tool)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile()

    def _get_primary_model_name(self) -> str:
        pk, ps = self.settings.models.get_primary_model()
        return ps.ollama_name

    async def run(self, query: str) -> dict[str, Any]:
        """Run the planner agent on a query."""
        if not self._graph:
            self._graph = self._build_graph()

        from langchain_core.messages import HumanMessage
        
        self._system_message = (
            "You are the MerchFine Lead Planner. Use your tools to look up SKU details, "
            "calculate MIO, and check seasonality. Provide a comprehensive plan."
        )

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "sku_id": None,
            "plan_generated": False,
            "context": {},
        }

        try:
            # Run the graph
            result_state = await self._graph.ainvoke(initial_state)
            final_message = result_state["messages"][-1]
            
            return {
                "answer": final_message.content,
                "tool_calls_made": len([m for m in result_state["messages"] if m.type == "tool"]),
                "status": "success",
            }
        except Exception as e:
            logger.error("Agent execution failed: %s", e)
            raise InferenceError(f"Agent failed: {e}")
