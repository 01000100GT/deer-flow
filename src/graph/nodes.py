# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents import create_agent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine

from uuid import uuid4

logger = logging.getLogger(__name__)


@tool
def handoff_to_planner(
    task_title: Annotated[str, "The title of the task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background investigation node is running.")
    configurable = Configuration.from_runnable_config(config)
    query = state["messages"][-1].content
    background_investigation_results = None
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        searched_content = LoggedTavilySearch(
            max_results=configurable.max_search_results
        ).invoke(query)
        if isinstance(searched_content, list):
            background_investigation_results = [
                f"## {elem['title']}\n\n{elem['content']}" for elem in searched_content
            ]
            return {
                "background_investigation_results": "\n\n".join(
                    background_investigation_results
                )
            }
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
    else:
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)
    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii=False
        )
    }


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan")
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    messages = apply_prompt_template("planner", state, configurable)

    if (
        plan_iterations == 0
        and state.get("enable_background_investigation")
        and state.get("background_investigation_results")
    ):
        messages += [
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    + "\n"
                ),
            }
        ]

    # 修改：不再需要为basic planner特殊处理结构化输出
    llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(goto="reporter")

    full_response = ""
    # 修改：统一处理llm响应，先获取原始文本
    if AGENT_LLM_MAP["planner"] == "basic":
        response = llm.invoke(messages)
        full_response = response.content
    else:
        response = llm.stream(messages)
        for chunk in response:
            full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    try:
        curr_plan = json.loads(repair_json_output(full_response))

        # 为LLM生成的计划添加缺失的必需字段
        from src.prompts.planner_model import StepType

        if "steps" in curr_plan:
            for step in curr_plan["steps"]:
                # 如果缺少step_type字段，设置默认值
                if "step_type" not in step:
                    step["step_type"] = StepType.RESEARCH.value
                # 如果缺少need_search字段，设置默认值
                if "need_search" not in step:
                    step["need_search"] = True

    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")
    if curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        new_plan = Plan.model_validate(curr_plan)
        return Command(
            update={
                "current_plan": new_plan,
            },
            goto="reporter",
        )
    return Command(
        update={
            "current_plan": full_response,
        },
        goto="human_feedback",
    )


def human_feedback_node(
    state,
) -> Command[
    Literal["planner", "research_team", "reporter", "manual_plan_editor", "__end__"]
]:
    logger.info("--- HUMAN_FEEDBACK [START] ---")
    current_plan = state.get("current_plan", "")
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    # 注释: 详细打印进入节点时current_plan的状态
    logger.info(f"--- HUMAN_FEEDBACK: current_plan on entry: {type(current_plan)} ---")
    if isinstance(current_plan, str):
        logger.info(current_plan[:1000])
    else:
        # 导入Plan以在日志记录中使用
        from src.prompts.planner_model import Plan

        if isinstance(current_plan, Plan):
            logger.info(current_plan.model_dump_json(indent=2))
        else:
            logger.info(current_plan)

    if not auto_accepted_plan:
        # # 注释: 这里是关键，我们必须调用interrupt()来暂停执行并等待前端反馈。
        # feedback = interrupt("Please Review the Plan.")
        # feedback_str = str(feedback)
        # 注释：获取状态中最新的消息，中断将附加到此消息上
        last_message = state.get("messages", [{}])[-1]
        last_message_id = last_message.id if hasattr(last_message, "id") else None

        # 注释：这里是关键，我们必须调用interrupt()来暂停执行并等待前端反馈。
        # 注释：通过指定ns（命名空间），确保中断附加到最新的消息上。
        feedback = interrupt("Please Review the Plan.")
        feedback_str = str(feedback)
        # 注释: 详细打印收到的中断反馈
        logger.info(
            f"--- HUMAN_FEEDBACK: Received feedback content: {feedback_str[:500]}"
        )
        logger.info(f"检查 feedback_str 的内容是: {feedback_str} ---")
        # 检查是否为系统自动编辑
        if feedback_str.strip().upper().startswith("[EDIT_PLAN]"):
            logger.info(
                "--- HUMAN_FEEDBACK [DECISION]: Detected [EDIT_PLAN], going to planner. ---"
            )
            return Command(
                update={
                    "messages": [HumanMessage(content=feedback_str, name="feedback")]
                },
                goto="planner",
            )
        # 检查是否为手工编辑提交
        elif "[MANUAL_EDIT]" in feedback_str:
            logger.info(
                "--- HUMAN_FEEDBACK [DECISION]: Detected [MANUAL_EDIT], going to manual_plan_editor. ---"
            )
            return Command(
                update={
                    "messages": [HumanMessage(content=feedback_str, name="feedback")]
                },
                goto="manual_plan_editor",
            )
        # 检查是否接受计划
        elif feedback_str.strip().upper().startswith("[ACCEPTED]"):
            logger.info(
                "--- HUMAN_FEEDBACK [DECISION]: Detected [ACCEPTED], proceeding. ---"
            )
            logger.info("Plan is accepted by user.")
        # 其他不支持的反馈类型
        else:
            # 允许空反馈，直接重新显示计划和按钮
            if feedback is None or feedback_str == "":
                logger.info(
                    "--- HUMAN_FEEDBACK [DECISION]: Feedback is empty, returning to human_feedback. ---"
                )
                return Command(goto="human_feedback")
            raise TypeError(f"Interrupt value of {feedback} is not supported.")

    # 如果计划被接受，则继续执行后续节点
    plan_iterations = state.get("plan_iterations", 0)
    goto = "research_team"
    try:
        # 注释：导入Plan和StepType用于类型检查和数据处理。
        from src.prompts.planner_model import Plan, StepType

        new_plan_data = {}
        # 注释：在处理之前，检查current_plan的类型。这是为了修复当计划已被解析为Plan对象时发生崩溃的错误。
        if isinstance(current_plan, str):
            # 注释：如果current_plan是一个字符串（例如，来自planner的初始原始输出），则修复并解析它。
            repaired_plan_str = repair_json_output(current_plan)
            new_plan_data = json.loads(repaired_plan_str)
        elif isinstance(current_plan, Plan):
            # 注释：如果current_plan已经是一个Plan对象（例如，在手动编辑之后），我们直接将其转换为字典以进行进一步处理。
            new_plan_data = current_plan.model_dump()
        else:
            # 注释：如果current_plan是任何其他意外类型，则引发类型错误。
            raise TypeError(f"Unsupported type for current_plan: {type(current_plan)}")

        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = new_plan_data

        # 为计划添加缺失的必需字段
        if "steps" in new_plan:
            for step in new_plan["steps"]:
                # 如果缺少step_type字段，设置默认值
                if "step_type" not in step:
                    step["step_type"] = StepType.RESEARCH.value
                # 如果缺少need_search字段，设置默认值
                if "need_search" not in step:
                    step["need_search"] = True

        if new_plan.get("has_enough_context"):
            goto = "reporter"
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Planner response is not a valid JSON or has wrong type: {e}")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")

    logger.info(f"--- HUMAN_FEEDBACK [END]: Returning with goto: {goto} ---")
    return Command(
        update={
            "current_plan": Plan.model_validate(new_plan),
            "plan_iterations": plan_iterations,
            "locale": new_plan["locale"],
        },
        goto=goto,
    )


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)
    messages = apply_prompt_template("coordinator", state)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    logger.debug(f"Current state messages: {state['messages']}")

    # 检查响应是否包含工具调用
    if response.tool_calls:
        # 如果模型调用了工具，我们不希望包含任何文本内容。
        # 因此，在框架将其添加到状态之前，我们强制清空文本。
        response.content = ""

    goto = "__end__"
    locale = state.get("locale", "en-US")  # Default locale if not specified

    if len(response.tool_calls) > 0:
        goto = "planner"
        if state.get("enable_background_investigation"):
            # if the search_before_planning is True, add the web search tool to the planner agent
            goto = "background_investigator"
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_locale := tool_call.get("args", {}).get("locale"):
                    locale = tool_locale
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
    else:
        logger.warning(
            "Coordinator response contains no tool calls. Terminating workflow execution."
        )
        logger.debug(f"Coordinator response: {response}")

    return Command(
        update={"locale": locale, "resources": configurable.resources},
        goto=goto,
    )


def reporter_node(state: State, config: RunnableConfig):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    configurable = Configuration.from_runnable_config(config)
    current_plan = state.get("current_plan")
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    invoke_messages = apply_prompt_template("reporter", input_, configurable)
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    invoke_messages.append(
        HumanMessage(
            content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
            name="system",
        )
    )

    for observation in observations:
        invoke_messages.append(
            HumanMessage(
                content=f"Below are some observations for the research task:\n\n{observation}",
                name="observation",
            )
        )
    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    response_content = response.content
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}


def research_team_node(state: State):
    """Research team node that collaborates on tasks."""
    logger.info("Research team is collaborating on tasks.")
    pass


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent."""
    current_plan = state.get("current_plan")
    observations = state.get("observations", [])

    # Find the first unexecuted step
    current_step = None
    completed_steps = []
    for step in current_plan.steps:
        if not step.execution_res:
            current_step = step
            break
        else:
            completed_steps.append(step)

    if not current_step:
        logger.warning("No unexecuted step found")
        return Command(goto="research_team")

    logger.info(f"Executing step: {current_step.title}, agent: {agent_name}")

    # Format completed steps information
    completed_steps_info = ""
    if completed_steps:
        completed_steps_info = "# Existing Research Findings\n\n"
        for i, step in enumerate(completed_steps):
            completed_steps_info += f"## Existing Finding {i + 1}: {step.title}\n\n"
            completed_steps_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

    # Prepare the input for the agent with completed steps info
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"{completed_steps_info}# Current Task\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
            )
        ]
    }

    # Add citation reminder for researcher agent
    if agent_name == "researcher":
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )

    # Invoke the agent
    default_recursion_limit = 25
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)

        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    logger.info(f"Agent input: {agent_input}")
    result = await agent.ainvoke(
        input=agent_input, config={"recursion_limit": recursion_limit}
    )

    # Process the result
    response_content = result["messages"][-1].content
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Update the step with the execution result
    current_step.execution_res = response_content
    logger.info(f"Step '{current_step.title}' execution completed by {agent_name}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": observations + [response_content],
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step.

    This function handles the common logic for both researcher_node and coder_node:
    1. Configures MCP servers and tools based on agent type
    2. Creates an agent with the appropriate tools or uses the default agent
    3. Executes the agent on the current step

    Args:
        state: The current state
        config: The runnable config
        agent_type: The type of agent ("researcher" or "coder")
        default_tools: The default tools to add to the agent

    Returns:
        Command to update state and go to research_team
    """
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = {}
    enabled_tools = {}

    # Extract MCP server configuration for this agent type
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                mcp_servers[server_name] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_name

    # Create and execute agent with MCP tools if available
    if mcp_servers:
        async with MultiServerMCPClient(mcp_servers) as client:
            loaded_tools = default_tools[:]
            for tool in client.get_tools():
                if tool.name in enabled_tools:
                    tool.description = (
                        f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                    )
                    loaded_tools.append(tool)
            agent = create_agent(agent_type, agent_type, loaded_tools, agent_type)
            return await _execute_agent_step(state, agent, agent_type)
    else:
        # Use default tools if no MCP servers are configured
        agent = create_agent(agent_type, agent_type, default_tools, agent_type)
        return await _execute_agent_step(state, agent, agent_type)


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")
    configurable = Configuration.from_runnable_config(config)
    tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        tools.insert(0, retriever_tool)
    logger.info(f"Researcher tools: {tools}")
    return await _setup_and_execute_agent_step(
        state,
        config,
        "researcher",
        tools,
    )


async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Coder node that do code analysis."""
    logger.info("Coder node is coding.")
    return await _setup_and_execute_agent_step(
        state,
        config,
        "coder",
        [python_repl_tool],
    )


def manual_plan_editor_node(
    state,
) -> Command[Literal["human_feedback", "research_team"]]:
    """处理手工编辑的计划"""
    logger.info("--- MANUAL_PLAN_EDITOR [START] ---")

    messages = state.get("messages", [])
    if not messages:
        logger.warning("--- MANUAL_PLAN_EDITOR [EXIT]: No messages found. ---")
        return Command(goto="human_feedback")

    last_message = messages[-1]
    if not (hasattr(last_message, "content") and last_message.content):
        logger.warning(
            "--- MANUAL_PLAN_EDITOR [EXIT]: Last message has no content. ---"
        )
        return Command(goto="human_feedback")

    content = last_message.content
    logger.info(f"--- MANUAL_PLAN_EDITOR: Parsing content: {content[:500]} ---")
    if "[MANUAL_EDIT]" not in content:
        logger.warning(
            "--- MANUAL_PLAN_EDITOR [EXIT]: [MANUAL_EDIT] not in content. ---"
        )
        return Command(goto="human_feedback")

    try:
        # 使用正则表达式从复杂字符串中提取核心的JSON负载
        import json
        import re
        from src.prompts.planner_model import Plan, StepType

        match = re.search(r"\[MANUAL_EDIT\]\s*({.*})", content, re.DOTALL)
        if not match:
            raise ValueError("Could not find [MANUAL_EDIT] JSON payload in the content")

        plan_json = match.group(1)
        edited_plan_data = json.loads(plan_json)

        # 为每个步骤添加缺失的必需字段，以确保验证通过
        if "steps" in edited_plan_data:
            for step in edited_plan_data["steps"]:
                if "step_type" not in step:
                    step["step_type"] = StepType.RESEARCH.value
                if "need_search" not in step:
                    step["need_search"] = True

        # 为计划本身添加缺失的必需字段
        if "locale" not in edited_plan_data:
            edited_plan_data["locale"] = state.get("locale", "zh-CN")
        if "has_enough_context" not in edited_plan_data:
            edited_plan_data["has_enough_context"] = False

        edited_plan = Plan.model_validate(edited_plan_data)

        # 将Pydantic模型转换为JSON字符串
        edited_plan_json = edited_plan.model_dump_json(indent=4, exclude_none=True)

        logger.info(
            f"--- MANUAL_PLAN_EDITOR: Plan parsed and validated successfully. Title: {edited_plan.title} ---"
        )

        # command_to_return = Command(
        #     update={
        #         "current_plan": edited_plan,
        #         "messages": [
        #             AIMessage(
        #                 content=edited_plan_json,
        #                 name="manual_plan_editor",
        #                 response_metadata={"finish_reason": "stop"},
        #             )
        #         ],
        #     },
        #     goto="human_feedback",
        # )
        new_message_id = f"manual_plan_editor:{uuid4()}"

        command_to_return = Command(
            update={
                "current_plan": edited_plan,
                "messages": [
                    AIMessage(
                        id=new_message_id,  # 注释：为消息明确分配ID
                        content=edited_plan_json,
                        name="manual_plan_editor",
                        response_metadata={"finish_reason": "stop"},
                    )
                ],
            },
            goto="human_feedback",
        )

        logger.info(
            f"--- MANUAL_PLAN_EDITOR [END]: Returning command to update plan and goto human_feedback. ---"
        )
        return command_to_return
    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"Failed to parse manually edited plan: {e}")
        # 如果解析失败，返回到人工反馈节点，让用户可以重试
        logger.warning("--- MANUAL_PLAN_EDITOR [EXIT]: Error parsing plan. ---")
        return Command(goto="human_feedback")
