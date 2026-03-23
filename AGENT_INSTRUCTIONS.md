# AI Agent Instructions

## Project Goal & Model Context
This is a sustainability and spatial-optimization project. We recognize that some deforestation is inevitable due to economic development. The goal of the models in this repository is to act as a **Prescriptive "What-If" Simulator**. 
- The models take in baseline satellite imagery and a proposed deforestation mask (a patch where a logging company wants to cut).
- They predict the causal "ripple effect" of that specific cut on the surrounding environment (e.g., changes in downstream water quality, soil degradation, secondary fire risk, and cascading forest loss) using real empirical data.
- By simulating different proposed logging plans, the system can recommend the exact coordinates that minimize total ecological harm. This allows governments and logging companies to meet economic quotas with the smallest possible environmental footprint.
- **Agents must deeply understand this Siamese Counterfactual setup.** The models are not just classifying where forest is; they are comparing a factual baseline to a counterfactual action to isolate and predict a causal spatial delta.

These instructions dictate how AI agents must interact with this codebase. All agents must strictly adhere to these guidelines.

## 1. Enterprise & Government Grade Accuracy
This codebase is intended for enterprise and government clients. The level of accuracy, security, and reliability must meet the highest industry standards. Code must be robust, scalable, and secure.

## 2. Unrestricted Resources
- **Cost and compute are not limits.**
- Always prioritize using the most capable, state-of-the-art models available for any task.
- Utilize the highest quality, most comprehensive data possible for any data-driven tasks, analysis, or model integration.

## 3. Mandatory Verification
- **Verify your code every single time.** Do not assume your generated code works. 
- You must write tests, run checks, and rigorously validate your logic before finalizing any changes.
- Ensure all automated and manual verification steps pass flawlessly.

## 4. Absolute Factuality & Real Data
- **No fake data or fake results.** Do not use generic placeholders, mock data (unless specifically writing unit test mocks), or hallucinated content.
- **Always resort to absolute factuality.** Base all your responses, code generation, and data usage on verified, real-world information and correct documentation.

## 5. Thorough Code Auditing
- **Never trust docstrings alone.** Docstrings can be stale, incomplete, or outright wrong. When auditing code, you must trace the **actual execution paths** line-by-line.
- **Trace data flow end-to-end.** For every claim you make in an audit, verify it by reading the actual code that executes — not the comment above it. For example:
  - Count the actual arrays passed to `np.stack()` or `torch.cat()` to verify channel counts, don't rely on the docstring's channel list.
  - Follow function call chains to verify that arguments (especially `**kwargs`) propagate correctly through wrappers.
  - Check that split logic in evaluation scripts matches the split logic used during training.
  - Verify mathematical formulas (e.g., Dice, loss functions) by reading the actual arithmetic, not the docstring description.
- **Cross-reference between files.** Verify that values defined in one file (e.g., `IN_CHANNELS = 7`) match what another file actually produces (e.g., the dataset's `__getitem__` stacking 7 arrays).
- **Flag docstring-vs-code discrepancies.** If a docstring says one thing but the code does another, report it as an audit finding and correct the docstring.