import os
import sys
import json
import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Step 1: Set GPU Environment (Using 4, 5, 6, 7) ===
# This line still sets the visible devices to 4, 5, 6, and 7.
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# === Step 2: Load Your Selected Queries CSV ===
selected_df = pd.read_csv("selected_queries.csv") 

# === Step 3: Load dev.json for Evidence (Verify paths) ===
DEV_JSON_PATH = "/data/camll_model/students_data/models/llama3/cmput692/MINIDEV/dev.json"
try:
    with open(DEV_JSON_PATH, "r") as f:
        dev_data = json.load(f)
except FileNotFoundError:
    print(f"Error: DEV_JSON_PATH not found at {DEV_JSON_PATH}. Please check the path.")
    sys.exit(1)

evidence_map = {(item["db_id"], item["question"]): item.get("evidence", "") for item in dev_data}

# === Step 4: Load Your Llama-3.3-70B Model (FULL PRECISION) ===
MODEL_PATH = "/data/camll_model/students_data/models/llama3/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"

# *** IMPORTANT: 4-bit quantization config is REMOVED as requested. ***
# The model will attempt to load in its default precision (usually float16/bfloat16).
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto", # Automatically shards across visible devices (4, 5, 6, 7)
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency without quantization
    )
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        do_sample=False,        
        temperature=0.01,       
        max_new_tokens=500,     
        return_full_text=False  
    )
except Exception as e:
    print(f"Error loading model: {e}")
    # If this fails, it is highly likely due to insufficient VRAM.
    print("\n\n*** MODEL LOADING FAILED. The 70B model requires >140GB of VRAM without quantization. Please check your GPU memory. ***")
    sys.exit(1)


# === Step 5: Load dev_tables.json and Parse Schema ===
SCHEMA_PATH = "/data/camll_model/students_data/models/llama3/cmput692/MINIDEV/dev_tables.json"
try:
    with open(SCHEMA_PATH, "r") as f:
        schema_data = json.load(f)
except FileNotFoundError:
    print(f"Error: SCHEMA_PATH not found at {SCHEMA_PATH}. Please check the path.")
    sys.exit(1)


def get_schema_from_json(db_id):
    entry = next((item for item in schema_data if item["db_id"] == db_id), None)
    if not entry:
        return "Schema not found."

    tables = entry["table_names_original"]
    columns = entry["column_names_original"]
    types = entry["column_types"]

    schema_text = ""
    for i, table in enumerate(tables):
        schema_text += f"Table: {table}\n"
        for j, (table_idx, col_name) in enumerate(columns):
            if table_idx == i and col_name != "*":
                schema_text += f"  - {col_name} ({types[j]})\n"
    
    return schema_text


# Simple prompt used before----->

# === SQL Extraction Function ===
#   def extract_sql_only(text):
#       # Try triple backtick block first
#       sql_blocks = re.findall(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
#       if sql_blocks:
#           return sql_blocks[-1].strip()
#   
#       # Fallback: extract after "SQL:" until next blank line or end
#       match = re.search(r"SQL:\s*(SELECT.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
#       if match:
#           return match.group(1).strip()
#   
#       return "SQL_NOT_FOUND"
#   
#   # === Step 5: Generate SQL for Each Query ===
#   results = []
#   
#   for _, row in selected_df.iterrows():
#       question_id = row["question_id"]
#       db_id = row["db_id"]
#       question = row["question"]
#   
#       schema_text = get_schema_from_json(db_id)
#       if schema_text == "Schema not found.":
#           print(f"Warning: Schema not found for db_id {db_id}")
#           continue
#   
#       prompt = f"""
#   You are a SQL expert. Given the following database schema and question, generate a valid SQL query.
#   
#   Schema:
#   {schema_text}
#   
#   Question:
#   {question}
#   
#   SQL:
#   """



# === SQL Extraction Function ===
def extract_sql_only(text):
    # Extracts content inside the last ```sql...``` block
    sql_blocks = re.findall(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if sql_blocks:
        return sql_blocks[-1].strip()

    # Fallback search for "Final SQL query: SELECT..."
    match = re.search(r"Final SQL query:\s*(SELECT.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return "SQL_NOT_FOUND"

# === Step 6: Generate SQL for Each Query (with enhanced debugging) ===
results = []

for idx, row in selected_df.iterrows():
    question_id = row["question_id"]
    db_id = row["db_id"]
    question = row["question"]

    schema_text = get_schema_from_json(db_id)
    if schema_text.startswith("Schema not found."):
        print(f"âš ï¸ Schema not found for {db_id}. Skipping...")
        results.append({
            "question_id": question_id,
            "db_id": db_id,
            "question": question,
            "predicted_sql": "SQL_NOT_GENERATED_SCHEMA_MISSING",
        })
        continue

    evidence = evidence_map.get((db_id, question), "")
    
    # --- PROMPT ---
    prompt = f"""
You are an **expert SQLite SQL generator**. Your sole task is to analyze the provided schema, evidence, and question to generate **only** a single, correct, and efficient SQLite SQL query.

**STRICT GENERATION RULES:**
1. **Output ONLY the SQL query** inside a single ```sql...``` block. Do not output any other text before or after the block.
2. Ensure the query is valid for **SQLite**.

**Schema Information:**
{schema_text}

**Evidence/Hints:**
{evidence}

**Question:** {question}

Step-by-step reasoning:
1. Deconstruct the question and evidence.
2. Identify necessary tables and columns from the schema.
3. Construct the final SQLite query.

Final SQL query:
```sql
"""
    
    print(f"ðŸ§  Generating SQL for question_id {question_id} (Index {idx}/{len(selected_df)-1})...")
    
    try:
        output = generator(prompt, max_new_tokens=500)[0]["generated_text"]
    except Exception as e:
        print(f"*** GENERATION FAILED for ID {question_id} with error: {e} ***")
        clean_sql = f"GENERATION_ERROR: {str(e)}"
        
    else:
        full_output = prompt + output 
        clean_sql = extract_sql_only(full_output)

        # DEBUGGING: Print raw output if extraction fails (returns empty string or SQL_NOT_FOUND)
        if clean_sql == "" or clean_sql == "SQL_NOT_FOUND":
            print(f"\n*** EXTRACTION FAILED for ID {question_id}. Check Raw Output. ***")
            print("--- RAW MODEL OUTPUT ---")
            print(output)
            print("------------------------\n")
            
            # Label the failed query more clearly in the final results list
            clean_sql = "SQL_NOT_GENERATED_EMPTY_OR_MISSING"

    results.append({
        "question_id": question_id,
        "db_id": db_id,
        "question": question,
        "predicted_sql": clean_sql
    })

# === Step 7: Save Results to final_predictions_for_evaluation.json ===
OUTPUT_FILENAME = "final_predictions_for_evaluation.json"
final_sql_list = [item["predicted_sql"] for item in results]

with open(OUTPUT_FILENAME, "w") as f:
    json.dump(final_sql_list, f, indent=4)

print(f"\n\nâœ… SQL Generation Complete.")
print(f"Results saved to {OUTPUT_FILENAME}. Total queries: {len(final_sql_list)}")
