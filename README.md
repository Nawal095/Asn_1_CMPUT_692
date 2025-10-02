## Execution Instructions

### Prerequisites
- Ensure Python 3.11+ is installed.
- Install required dependencies: `pip install pandas torch transformers`.
- Verify access to GPUs with sufficient VRAM (at least 140GB for Llama 3.3 70B without quantization).
- Clone the BIRD Mini-Dev repository: `git clone https://github.com/bird-bench/mini_dev.git`.

### Setup
1. **Download Datasets**:
   - Obtain the Mini-Dev dataset from [Hugging Face](https://huggingface.co/datasets/birdsql/bird_mini_dev) and place `mini_dev_sqlite.jsonl` in the `../sqlite/` directory relative to the evaluation script.
   - Download the ground truth file `mini_dev_sqlite_gold.sql` and place it in `../sqlite/`.

2. **Prepare Input Files**:
   - Ensure `selected_queries.csv` is available in the project root, containing `question_id`, `db_id`, and `question` columns.
   - Verify `dev.json` and `dev_tables.json` are correctly path at `/data/camll_model/students_data/models/llama3/cmput692/MINIDEV/` or adjust paths in the script.

3. **Model Configuration**:
   - Load the Llama-3.3-70B-Instruct model from HUGGING_FACE or update `MODEL_PATH` to your model location.
   - Set in your environment for multi-GPU support.

### Running the Evaluation
1. **Generate SQL Queries**:
   - Run the Python script to generate predictions: `python generate_sql.py` (replace with your script name, e.g., `generate_sql.py`).
   - This will produce `final_predictions_for_evaluation.json` containing the predicted SQL queries.

2. **Format Predictions**:
   - Ensure the output `final_predictions_for_evaluation.json` is a valid JSON list of SQL strings. If errors like `JSONDecodeError: Expecting property name enclosed in double quotes` occur, validate the JSON format (e.g., ensure no trailing commas or invalid characters) using a linter or manually edit the file.
   - Post-process the output to match the required format: each SQL query followed by `\t----- bird -----\t<db_id>` (e.g., using a script or manual adjustment if needed).

3. **Execute Evaluation**:
   - Navigate to the `evaluation` directory: `cd evaluation`.
   - Run the evaluation script: `./run_evaluation.sh`.
   - The script will compute Execution (EX), Reward-based Valid Efficiency Score (R-VES), and Soft F1-Score metrics for SQLite dialect, comparing against `mini_dev_sqlite_gold.sql` which is in the sqlite folder.

### Troubleshooting
- **FileNotFoundError**: Verify `../sqlite/mini_dev_sqlite.jsonl` exists and the path is correct relative to `evaluation_ex.py`. Adjust the path in `run_evaluation.sh` if necessary.
- **IndexError**: Ensure the number of predicted queries matches the ground truth count (40). Check `final_predictions_for_evaluation.json` for missing or extra entries.
- **JSONDecodeError**: Validate the JSON structure of `final_predictions_for_evaluation.json`. Use `jq` or a JSON validator to fix syntax errors (e.g., unquoted keys or invalid characters).
- **Script Errors**: If `./run_evaluation.sh: line 63: ct}: command not found` appears, check for syntax errors in the shell script (e.g., missing `fi` or `done`) and correct them.

### Expected Output

Below is an example of the evaluation output when running the `./run_evaluation.sh` script, demonstrating the performance metrics for Execution Metric Score & Soft F1-Score and Reward-based Valid Efficiency Score (R-VES) on the Mini Dev dataset using SQLite dialect:

Execution Evaluation:

                     simple  moderate  challenging  total
count                13      17        10           40
====================================== EX =====================================
EX                   92.31   94.12     80.00        90.00


R-VES:
                     simple  moderate  challenging  total
count                13      17        10           40
====================================== R-VES =====================================
R-VES                92.19   92.54     74.64        87.95

Soft F1:

                     simple  moderate  challenging  total
count                13      17        10           40
====================================== Soft-F1 =====================================
Soft-F1              92.31   94.12     82.22        90.56

- Results are aggregated by difficulty (Simple, Moderate, Challenging) and total, reflecting the 40-query subset performance.

### Notes
- Adjust file paths and model locations based on your local setup.
- Refer to the BIRD Mini-Dev repository [README](https://github.com/bird-bench/mini_dev) for additional setup details if issues persist.