# Agent Mode Troubleshooting

## Common Issues and Solutions

### Recursion Limit Error

**Symptom:** Agent returns error saying "exceeded maximum steps" or "recursion limit"

**Cause:** The agent tried to take more than 15 steps (tool calls) to complete the workflow. This usually happens when:
1. The agent gets confused and repeats the same actions
2. The workflow is too complex
3. Required files (like labeled data) don't exist

**Solution:**

**Option 1: Use Manual Mode (Recommended)**
- Manual Mode gives you fine-grained control
- You can see exactly what's happening at each step
- Better for most use cases

**Option 2: Simplify Your Request**
Try making your request more specific:
- ❌ Bad: "Generate synthetic data with labeling and training"
- ✅ Good: "Generate 1000 synthetic samples"

**Option 3: Be Explicit About Skipping Labeling**
Most workflows should skip labeling:
- ✅ "Generate 1000 samples without labeling"
- ✅ "Train a model and generate synthetic data"

### Agent Takes Too Long

**Symptom:** Request times out after 5 minutes

**Cause:** Training large models or generating many samples takes time

**Solution:**
1. Use Manual Mode for better progress tracking
2. Reduce the number of samples or epochs in your request
3. Check orchestrator logs: `tail -f logs/orchestrator.log`

### Agent Can't Find Dataset

**Symptom:** Error about missing files or invalid paths

**Cause:** Agent needs the full file path to your dataset

**Solution:**
Make sure you've uploaded a dataset before using Agent Mode. The frontend automatically passes the dataset path to the agent.

### Parquet Magic Bytes Error

**Symptom:** Error: "Parquet magic bytes not found in footer"

**Cause:** The agent pipelines require Parquet format, but a CSV file was passed

**Solution:**
This is now automatically handled! The backend converts CSV files to Parquet before passing them to the agent. If you see this error, restart the Flask backend to get the latest fix.

### Labeling Errors

**Symptom:** Agent fails when trying to label data

**Cause:** The self-correcting labeling pipeline requires:
- `hand_labeled_examples_path` - Ground truth data for validation
- These files usually don't exist

**Solution:**
**Skip labeling for most workflows!** The agent now defaults to skipping labeling unless you explicitly request it.

## Best Practices for Agent Mode

### ✅ Good Use Cases
- "Generate 1000 synthetic samples"
- "Create synthetic data for my dataset"
- "Train a model with 10 epochs and generate 500 samples"

### ❌ Avoid These
- Complex multi-step workflows with many conditionals
- Requests requiring files that don't exist
- Workflows that need precise parameter tuning

### When to Use Manual Mode vs Agent Mode

**Use Manual Mode when:**
- You want fine-grained control over parameters
- You need to see progress at each step
- You're experimenting with different settings
- You want to customize labeling functions

**Use Agent Mode when:**
- You want quick synthetic data generation
- You're okay with default settings
- You want natural language interface
- Your workflow is straightforward

## Configuration

The agent has these limits configured:
- **Max Steps:** 15 (prevents infinite loops)
- **Timeout:** 5 minutes (for long-running operations)
- **Temperature:** 0.7 (for LLM creativity)

These are set in `backend/agents/serve_orchestrator.py` and can be adjusted if needed.

## Logs and Debugging

### View Orchestrator Logs
```bash
tail -f logs/orchestrator.log
```

### View Flask API Logs
```bash
tail -f logs/flask.log
```

### Check What the Agent Is Doing
The orchestrator logs show:
- Each tool call the agent makes
- Return values from each tool
- Any errors encountered
- Final response

## Emergency: Reset Everything

If the agent is completely stuck:

```bash
# Stop all services
./stop.sh

# Clear any stuck processes
pkill -f serve_orchestrator
pkill -f flask_api

# Restart
./start.sh
```

## Still Having Issues?

1. Check that NVIDIA_API_KEY is set in `.env`
2. Verify both Flask (5000) and Orchestrator (8000) are running
3. Try Manual Mode instead
4. Check the logs for specific error messages
5. Simplify your request to the most basic workflow

Remember: **Agent Mode is experimental**. For production workflows, use Manual Mode for reliability and control.

