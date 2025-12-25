# v1 
# policies_text += f"=== Policy {i} ===\n{policy}\n\n"
    
# MERGE_PROMPT = f"""You are given multiple agent policies, each focused on a specific customer service workflow (e.g., exchanges, returns, cancellations, modifications).

# Your task is to merge these into ONE unified, coherent policy that covers all workflows.

# {policies_text}

# Please create a single comprehensive policy that:
# 1. Combines all the guidelines and rules from the individual policies
# 2. Organizes them in a clear, logical structure
# 3. Removes redundancies while preserving all important details
# 4. Maintains specificity for each workflow type where needed
# 5. Identifies common patterns that apply across workflows

# Provide the complete merged policy."""

# v2     # Build the merge prompt
#     policies_text = ""
#     for i, wiki in enumerate(wikis, 1):
#         policies_text += f"=== Wiki {i} ===\n{wiki}\n\n"
    
#     MERGE_PROMPT = f"""You are given multiple agent wikis, each focused on a specific customer service workflow (e.g., exchanges, returns, cancellations, modifications).

# Your task is to merge these into ONE unified, coherent policy that covers all workflows.

# {policies_text}

# Please create a single comprehensive wiki that:
# only removes redundancies while preserving all important details

# Provide the complete merged wiki."""

#     # Call LLM
#     print(f"Merging {len(wikis)} wikis using gpt-5...")
#     merged_policy = call_llm(MERGE_PROMPT, 'gpt-5')
    
#     return merged_policy



