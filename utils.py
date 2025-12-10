# Create a reusable script that prints colorized conversations from your results file.
from textwrap import fill, indent
import json
import argparse
import os
import sys

ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "user": "\033[92m",       # green
    "assistant": "\033[96m",  # cyan
    "tool": "\033[95m",       # magenta
    "meta": "\033[90m",       # gray
    "error": "\033[91m",      # red
}

def _supports_color(force_color: bool) -> bool:
    if force_color:
        return True
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        # On Windows 10+ most terminals support ANSI, but detect via env when possible
        return True
    return True

def _color(text: str, code: str, enable: bool) -> str:
    return f"{ANSI[code]}{text}{ANSI['reset']}" if enable else text

def _wrap(text: str, width: int) -> str:
    if width <= 0:
        return text
    # Preserve code blocks and lists minimally by line
    lines = []
    for line in text.splitlines():
        if line.strip().startswith(("```", "> ", "- ", "* ")):
            lines.append(line)
        else:
            lines.append(fill(line, width=width))
    return "\n".join(lines)

def _compact_args(arg_str: str, max_len: int = 200) -> str:
    """
    Turn a function.arguments JSON string into a compact key=value, comma-separated
    representation, truncating long values.
    """
    if not arg_str:
        return ""
    try:
        args = json.loads(arg_str)
    except Exception:
        s = arg_str.replace("\n", " ")
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    parts = []
    def trunc(v):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= 80 else s[:79] + "…"

    if isinstance(args, dict):
        for k, v in args.items():
            parts.append(f"{k}={trunc(v)}")
        out = ", ".join(parts)
    else:
        out = trunc(args)

    return out if len(out) <= max_len else out[: max_len - 1] + "…"


def print_conversation(
    file_path: str,
    task_id: int,
    trial: int,
    width: int = 100,
    color: bool = True,
):
    if not os.path.exists(file_path):
        print(_color(f"[error] File not found: {file_path}", "error", color))
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(_color(f"[error] Failed to parse JSON: {e}", "error", color))
        return

    # Find the entry with matching task_id and trial
    match = None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("task_id") == task_id and entry.get("trial") == trial:
                match = entry
                break
    else:
        print(_color("[error] Expected a list at JSON top level.", "error", color))
        return

    if match is None:
        print(_color(f"[error] No record found for task_id={task_id}, trial={trial}.", "error", color))
        return

    traj = match.get("traj", [])
    if not traj:
        print(_color("[error] No 'traj' found in the selected record.", "error", color))
        return

    # Pretty header
    header = f"Conversation — task_id={task_id}, trial={trial}"
    print(_color(header, "bold", color))
    print(_color("─" * len(header), "meta", color))

    # Instruction (if present)
    instr = None
    try:
        instr = match["info"]["task"]["instruction"]
    except Exception:
        instr = None
    if instr:
        print(_color("[instruction]", "bold", color))
        print(indent(_wrap(instr, width), "  "))
        print(_color("─" * len(header), "meta", color))

    # Build tool_call_id -> (func_name, compact_args)
    id2call = {}
    for turn in traj:
        if turn.get("role") == "assistant":
            for tc in (turn.get("tool_calls") or []):
                func = (tc or {}).get("function") or {}
                fname = func.get("name", "")
                fargs = func.get("arguments", "")
                comp = _compact_args(fargs)
                call_id = tc.get("id") or tc.get("tool_call_id")
                if call_id:
                    id2call[call_id] = (fname, comp)

    # Start from first user turn
    start_idx = 0
    for i, turn in enumerate(traj):
        if isinstance(turn, dict) and turn.get("role") == "user":
            start_idx = i
            break

    # Iterate and print
    for turn in traj[start_idx:]:
        role = turn.get("role", "assistant")
        raw_content = turn.get("content")
        tool_name = turn.get("name")                # for role == "tool"
        tool_call_id = turn.get("tool_call_id")     # for role == "tool"

        if role == "assistant":
            role_tag = _color("[assistant]", "assistant", color)
            if raw_content is None:
                # Assistant made tool call(s) only — print compact summary with args
                tool_calls = turn.get("tool_calls") or []
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        func = (tc or {}).get("function") or {}
                        fname = func.get("name", "<?>")
                        comp = _compact_args(func.get("arguments", ""))
                        parts.append(f"{fname}({comp})" if comp else f"{fname}()")
                    msg = "→ tool call(s): " + "; ".join(parts)
                else:
                    msg = ""
            else:
                msg = raw_content

        elif role == "user":
            role_tag = _color("[user]", "user", color)
            msg = raw_content or ""

        elif role == "tool":
            # Show tool name with compact args from the originating assistant call
            fname, comp = ("", "")
            if tool_call_id and tool_call_id in id2call:
                fname, comp = id2call[tool_call_id]
            display = tool_name or fname or ""
            if display and comp:
                role_tag = _color(f"[tool:{display}({comp})]", "tool", color)
            elif display:
                role_tag = _color(f"[tool:{display}]", "tool", color)
            else:
                role_tag = _color("[tool]", "tool", color)
            msg = raw_content or ""

        elif role == "system":
            role_tag = _color("[system]", "meta", color)
            msg = raw_content or ""

        else:
            role_tag = _color(f"[{role}]", "meta", color)
            msg = raw_content or ""

        # Wrap and indent content
        msg_wrapped = _wrap(msg, width)
        print(f"{role_tag} ")
        if msg_wrapped.strip():
            print(indent(msg_wrapped, "  "))
        else:
            print(indent(_color("(no content)", "dim", color), "  "))

def filter_tasks_by_tool(
    file_path: str,
    tool_name: str,
) -> list[int]:
    """
    Filter tasks by ground truth tool calls.
    
    Args:
        file_path: Path to results.json
        tool_name: Tool name to filter by (e.g., "exchange_delivered_order_items")
        exact_match: If True, only return tasks where ground truth contains ONLY this tool.
                     If False, return tasks where ground truth contains this tool (may have others).
    
    Returns:
        List of task_ids that match the filter criteria
        Successful ids on this result file 
    """
    if not os.path.exists(file_path):
        print(f"[error] File not found: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[error] Failed to parse JSON: {e}")
        return []
    
    if not isinstance(data, list):
        print("[error] Expected a list at JSON top level.")
        return []
    
    matching_task_ids = []
    successful_task_ids = []
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        task_id = entry.get("task_id")
        if task_id is None:
            continue
        
        # Extract ground truth tool calls
        try:
            gt_tools = entry["info"]["task"]["actions"]
        except (KeyError, TypeError):
            continue
        
        if not isinstance(gt_tools, list):
            continue
        
        # Check if this task matches our filter
        # At least one ground truth tool must be the target tool
        for tool_call in gt_tools:
            gt_tool_name = tool_call.get("name", "")
            if gt_tool_name == tool_name and task_id not in matching_task_ids:
                matching_task_ids.append(task_id)
                if entry.get("reward", 0.0) > 0.0:
                    successful_task_ids.append(task_id)

    return sorted(matching_task_ids), sorted(successful_task_ids)

def extract_conversation_text(
    file_path: str,
    task_id: int,
    trial: int,
    include_instruction: bool = False,
    include_system: bool = False,
    width: int = None,
) -> str:
    """
    Extract conversation as clean formatted text for feeding to LLM.
    
    Args:
        file_path: Path to results.json
        task_id: Task ID to extract
        trial: Trial number to extract
        include_instruction: Whether to include the task instruction at the top
        include_system: Whether to include system messages
        width: Line width for wrapping (None for no wrapping)
    
    Returns:
        Formatted conversation string, or empty string if not found
    """
    if not os.path.exists(file_path):
        return ""
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    
    # Find the entry
    match = None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("task_id") == task_id and entry.get("trial") == trial:
                match = entry
                break
    
    if match is None:
        return ""
    
    traj = match.get("traj", [])
    if not traj:
        return ""
    
    lines = []
    
    # Add instruction if requested
    if include_instruction:
        try:
            instr = match["info"]["task"]["instruction"]
            if instr:
                lines.append("[instruction]")
                if width:
                    lines.append(textwrap.fill(instr, width=width))
                else:
                    lines.append(instr)
                lines.append("---")
        except Exception:
            pass
    
    # Build tool_call_id -> (func_name, compact_args) mapping
    id2call = {}
    for turn in traj:
        if turn.get("role") == "assistant":
            for tc in (turn.get("tool_calls") or []):
                func = (tc or {}).get("function") or {}
                fname = func.get("name", "")
                fargs = func.get("arguments", "")
                comp = _compact_args(fargs)
                call_id = tc.get("id") or tc.get("tool_call_id")
                if call_id:
                    id2call[call_id] = (fname, comp)
    
    # Find first user turn
    start_idx = 0
    for i, turn in enumerate(traj):
        if isinstance(turn, dict) and turn.get("role") == "user":
            start_idx = i
            break
    
    # Process turns
    for turn in traj[start_idx:]:
        role = turn.get("role", "assistant")
        raw_content = turn.get("content")
        tool_name = turn.get("name")
        tool_call_id = turn.get("tool_call_id")
        
        if role == "assistant":
            if raw_content is None:
                # Tool call(s) only
                tool_calls = turn.get("tool_calls") or []
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        func = (tc or {}).get("function") or {}
                        fname = func.get("name", "<?>")
                        comp = _compact_args(func.get("arguments", ""))
                        parts.append(f"{fname}({comp})" if comp else f"{fname}()")
                    msg = "→ tool call(s): " + "; ".join(parts)
                else:
                    msg = ""
            else:
                msg = raw_content
            
            lines.append("[assistant]")
            if msg:
                if width:
                    lines.append(textwrap.fill(msg, width=width))
                else:
                    lines.append(msg)
        
        elif role == "user":
            msg = raw_content or ""
            lines.append("[user]")
            if msg:
                if width:
                    lines.append(textwrap.fill(msg, width=width))
                else:
                    lines.append(msg)
        
        elif role == "tool":
            fname, comp = ("", "")
            if tool_call_id and tool_call_id in id2call:
                fname, comp = id2call[tool_call_id]
            display = tool_name or fname or ""
            if display and comp:
                role_tag = f"[tool:{display}({comp})]"
            elif display:
                role_tag = f"[tool:{display}]"
            else:
                role_tag = "[tool]"
            
            msg = raw_content or ""
            lines.append(role_tag)
            if msg:
                if width:
                    lines.append(textwrap.fill(msg, width=width))
                else:
                    lines.append(msg)
        
        elif role == "system":
            if include_system:
                msg = raw_content or ""
                lines.append("[system]")
                if msg:
                    if width:
                        lines.append(textwrap.fill(msg, width=width))
                    else:
                        lines.append(msg)
    
    return "\n".join(lines)

def _build_argparser():
    p = argparse.ArgumentParser(description="Print colorized conversation from LLM trajectories JSON.")
    p.add_argument("--file", "-f", required=True, help="Path to the JSON result file.")
    p.add_argument("--task-id", "-t", type=int, required=True, help="Task ID to select.")
    p.add_argument("--trial", "-r", type=int, required=True, help="Trial index to select.")
    p.add_argument("--width", "-w", type=int, default=100, help="Soft-wrap width (<=0 to disable).")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color output.")
    return p

def main():
    ap = _build_argparser()
    args = ap.parse_args()
    use_color = _supports_color(force_color=not args.no_color)
    print_conversation(args.file, args.task_id, args.trial, width=args.width, color=use_color)

def filter_by_tool_call(file_paths, tool_name):
    """Filter the results file by successful tool call.
    Input:
    - file_path: list(str), list of paths to the JSON results file
    - tool_name: str, name of the tool to filter by
    
    Output:
    - write to tool_filtered_{original_filename}.json
    """
    filtered_data = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            continue

        
        for entry in data:
            if entry.get("reward") == 0.0:
                continue
            traj = entry.get("traj", [])
            for turn in traj:
                if turn.get("role") == "assistant":
                    tool_calls = turn.get("tool_calls") or []
                    for tc in tool_calls:
                        func = (tc or {}).get("function") or {}
                        fname = func.get("name", "")
                        if fname == tool_name:
                            filtered_data.append(entry)
                            break

        # output_file = f"{tool_name}_{os.path.basename(file_path)}"
    output_file = f"{tool_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"Filtered data written to {output_file}")
    

if __name__ == "__main__":
    main()
