# ev | Lightweight agent eval CLI 

`ev` is a lightweight and local agent eval and prompt refinemenet CLI, to stress-test AI agents and make prompts more robust.

It does three main things:

- Runs a suite of JSON test cases against a prompt pair (`system_prompt.j2` + `user_prompt.j2`)
- Evaluates results against explicit criteria defined in `eval.md`
- Iteratively improves the prompts, only accepting new versions that perform better



---


### Key Features

- **Multi-criteria evals**: Test prompts against any number of criteria defined in `eval.md`.
- **Deterministic scoring**: Test cases N times to harden prompts and create edge cases resistance.
- **Iterative refinement**: Automatically proposes and tests improved prompt versions.
- **Version gating**: Only snapshots a new version when it clearly outperforms the current one.
- **Native**: Everything is runs locally, no external infra.
- **Lightweight**: No feature bloat.
- **Model-flexibility**: Currenly supports OpenAI and Groq models via `provider[name]` notation (model support requests welcome)

---

## Table of contents

- [Core concepts](#core-concepts)
- [Installation and requirements](#installation-and-requirements)
- [Configuration and API keys](#configuration-and-api-keys)
- [Project layout](#project-layout)
- [Creating and setting up a test](#creating-and-setting-up-a-test)
- [Running evaluations](#running-evaluations)
  - [`ev run` for optimization](#ev-run-for-optimization)
  - [`ev eval` for evaluation only](#ev-eval-for-evaluation-only)
- [Understanding the outputs](#understanding-the-outputs)
  - `summary.json`
  - `versions/` and `log.json`
- [Other CLI commands](#other-cli-commands)
  - `ev list`
  - `ev copy`
  - `ev delete`
  - `ev version`
- [Models and cycles](#models-and-cycles)

---

## Core concepts

- **Eval**  
  A test is a folder under `evals` (for example `evals/myAgent`). It contains:
  - JSON cases in `cases/`
  - Criteria definitions in `eval.md`
  - A Pydantic schema in `schema.py`
  - Prompt templates in `system_prompt.j2` and `user_prompt.j2`

- **Case**  
  A single JSON input file under `cases/`. E.g. the data you want to test. One eval should have many cases.

- **Eval criteria**  
  Each `#` heading in `eval.md` defines one criterion.  
  You can have many criteria, and each is judged independently.

- **Cycles**  
  A cycle means evaluating all cases once.  
  More cycles reduce randomness and stabilize the score.

  Total evaluations per iteration:  
  `cases × cycles`

- **Iterations**  
  Each iteration:  
  1. Evaluate the current prompts  
  2. Generate improved prompts  
  3. Re-evaluate the candidate  
  4. Compare pass rates

  Total model calls per run:  
  **cases × cycles × iterations**

  Think of it as a contest: each iteration tries to produce a better prompt.

- **Pass rate**  
  Criteria scores are averaged across cases, then averaged across criteria.  
  This avoids one noisy criterion dominating the result.

- **Versions**  
  A new version is created only if the **best candidate** from the run beats the active version.  
  One new version max per `ev run`.


---

## Installation and requirements

```bash
pip install evx
````

or with `uv`:

```bash
uv tool install evx
```

Requires Python >=3.12


### Verify installation

```bash
ev --help
```

---

## Configuration and API keys

The CLI reads configuration from:

* `.env` file by default
* Or environment variables if you request it

### `.env` based config

By default, keys are loaded from `.env` (must be in root)

Currenly supported `.env` vars:

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gk-...
```

### Key source flag

You can control where keys are loaded from using the `--key` flag:

* `--key file` or `-k file` (default) loads from `.env`
* `--key env` or `-k env` loads from `os.environ`

Examples:

```bash
# use keys from .env
ev create myAgent
ev run myAgent -i 3 --cycles 2 --key file

# use keys from environment variables
set OPENAI_API_KEY=sk-...
set GROQ_API_KEY=gk-...
ev run myAgent -i 3 --cycles 2 --key env
```

---

## Project layout

At the top level, the tool expects an `evals` directory.

```text
<repo-root>/
  evals/
    myAgent/
      cases/
        example.json
      eval.md
      schema.py
      system_prompt.j2
      user_prompt.j2
      versions/
        base - <timestamp>/
          system_prompt.j2
          user_prompt.j2
          summary.json
        <other versions>/
      versions/log.json
```



---

## Creating and setting up a test

### 1) Scaffold a new eval

```bash
ev create myAgent
```

This will:

* Create `evals/myAgent`
* Add `cases/example.json`
* Add a blank `eval.md`
* Add a minimal `schema.py`
* Add basic `system_prompt.j2` and `user_prompt.j2`

which will create files in `evals/myAgent/...`

### 2) Define your response schema

Open `evals/myAgent/schema.py` and define the expected model. For example:

```python
from pydantic import BaseModel

class Response(BaseModel):
    risk_class: str
    recommendation: str
    explanation: str
```

This schema is used when the cases are generated in evals.

### 3) Define your eval criteria

Edit `evals/myAgent/eval.md` and declare your criteria:

```markdown
# classification
- The classification should be one of ["low", "medium", "high"] and should match the scenario.
- ...

# use_of_data
-The answer should use the provided input fields and not ignore key details.
- ...

# explanation
- The explanation should be honest, clear, and concise.
- ...
```

Each `# heading` becomes a separate criterion that the eval agent scores.

### 4) Add cases

Add JSON files under `evals/myAgent/cases/`. One file per test case.

```json
// evals/myAgent/cases/case1.json
{
  "business_name": "Acme Widgets",
  "sector": "Manufacturing",
  "revenue": 5000000
}
```

```json
// evals/myAgent/cases/case2.json
{
  "business_name": "Beta Health",
  "sector": "Healthcare",
  "revenue": 12000000
}
```

### 5) Refine your prompts

Edit:

* `evals/myAgent/system_prompt.j2`
* `evals/myAgent/user_prompt.j2`

You can access test case JSON fields via `{{ data.<field> }}`.

Example `user_prompt.j2`:

```jinja2
A business owner is applying for a loan.

Business name: {{ data.business_name }}
Sector: {{ data.sector }}
Revenue: {{ data.revenue }}

Classify the credit risk and tell the business owner what you recommend they do next.
Respond using the JSON schema described in your system instructions.
```

---

## Running evaluations

### `ev run` for optimization

`ev run` runs the whole loop:

1. Evaluates the current active version across all cases
2. Lets an agent propose changes to the prompts
3. Evaluates the candidate version
4. Only accepts and snapshots the candidate if the pass rate is higher than the current best

Basic usage:

```bash
ev run myAgent
```

Common options:

```bash
# Run 3 optimization iterations, single cycle per case
ev run myAgent -i 3

# Run 5 iterations, 2 cycles per case
ev run myAgent -i 5 -c 2

# Use a specific shared model for both generation and eval
ev run myAgent -m "groq[moonshotai/kimi-k2-instruct]"

# Different models for generation and eval
ev run myAgent \
  --gen-model "groq[moonshotai/kimi-k2-instruct]" \
  --eval-model "openai[gpt-5]"
```

New versions are only gnerated if the run beat the active version.


#### `ev run` Flags

A simple list of all flags supported by `ev run`:

`-i`, `--iterations`
- Number of self-improvement loops to run.  
- Each iteration proposes improved prompts and accepts them only if pass rate increases.

`-c`, `--cycles`
- Number of evaluation cycles per case.  
- Scores are averaged across cycles to reduce randomness.

`-m`, `--model`
- Sets a single model for both generation and evaluation.

`--gen-model`
- Overrides only the generation model.  
- Takes precedence over `--model`.

`--eval-model`
- Overrides only the evaluation model.  
- Takes precedence over `--model`.

`-k`, `--key`
- Where to load API keys from.  
- `file` (default, loads from `.env`) or `env` (loans from environment variables).


### `ev eval` for evaluation only

`ev eval` runs the test suite against the current active version without changing any prompts or creating new versions.

```bash
ev eval myAgent
````

With options:

```bash
# Multiple cycles for stability checking
ev eval myAgent -c 3

# Custom model overrides
ev eval myAgent -m "groq[moonshotai/kimi-k2-instruct]"
```

### `ev eval` flags

`--eval-model`

* Overrides only the evaluation model.
* Takes precedence over `--model`.

`-k`, `--key`

* Where to load API keys from.
* `file` (default, loads from `.env`) or `env` (loads from environment variables).


### Understanding the active version

Each test has one **active version**: the best-performing prompt pair so far.

A new version is created **only if** a candidate from the current `ev run` achieves a **higher pass rate** than the active version.  
If no candidate beats it, **no new version is saved**.

Only **one** new version can be created per `ev run` (the best candidate of that run).  
This keeps history clean and ensures every version is a strict improvement.

---


## Understanding the outputs

### Summary table (console)

At the end of an eval, you will see something like:

```text
=== SUMMARY TABLE ===
Version: base - 18 Nov 2025 14-22-10
Pass rate: 96.0 percent
Cycles: 1

Case                 | Criteria            | Score     
-------------------- | ------------------- | ----------
1                    | classification      | 100 percent  
                     | use_of_data         | 67 percent
                     | explanation         | 100 percent  
-------------------- | ------------------- | ----------
2                    | classification      | 100 percent  
                     | use_of_data         | 100 percent  
                     | explanation         | 100 percent  
-------------------- | ------------------- | ----------
```

Notes:

* `Pass rate` is the average across criteria, not just number of fully passing cases.
* `Score` is per criterion, expressed in percent.
* Each score is averaged across cycles when `--cycles > 1`.



### `summary.json`

For each version, `summary.json` is written under:

```text
evals/<test>/versions/<version-id>/summary.json
```

It contains:

* `version` - the version identifier
* `total_cases`
* `passed_cases` - cases where all criteria passed
* `pass_rate` - overall criteria based pass rate
* `cycles` - number of cycles used in this run
* `cases` - per case metrics

You can use this file for dashboards or CI integration.

### `versions/log.json`

`evals/<test>/versions/log.json` tracks versions:

```json
[
  {
    "version": "base - 18 Nov 2025 14-22-10",
    "pass_rate": 0.83,
    "is_active": false,
    "date": "2025-11-18T14:22:10.123456",
    "cycles": 1
  },
  {
    "version": "abcd1234 - 18 Nov 2025 15-01-42",
    "pass_rate": 0.95,
    "is_active": true,
    "date": "2025-11-18T15:01:42.789012",
    "cycles": 1
  }
]
```

The `is_active` flag marks which version will be used when you run `ev run` or `ev eval`.

---

## Other CLI commands

### `ev list` - list tests

Lists tests under `evals`:

```bash
ev list
```

Example output:

```text
› Available tests
  myAgent
  creditRisk_v2
  onboarding_bot
```

### `ev copy` - copy a test

Duplicates an existing test folder:

```bash
ev copy myAgent
```

This creates `evals/myAgent_copy`.


### `ev delete` - delete a test

Deletes a test and everything inside it:

```bash
ev delete myAgent
```

You can add `-y` to skip confirmation:

```bash
ev delete myAgent -y
```

Use with care.

### `ev version` - show active version

Displays the active version for a test:

```bash
ev version myAgent
```

Output:

```text
› Fetching active version for 'myAgent'
  path: <repo>/evals/myAgent
✓ Active version: abcd1234 - 18 Nov 2025 15-01-42
```

---

## Models and cycles

````markdown
### Models

You can control which LLMs are used for generation and evaluation.

* `-m, --model` sets both generation and eval model.
* `--gen-model` overrides only the generation model.
* `--eval-model` overrides only the eval model.

The format is:

```text
provider[identifier]
````

Examples:

```bash
ev run myAgent -m "openai[gpt-5]"
ev run myAgent --gen-model "groq[moonshotai/kimi-k2-instruct]" --eval-model "openai[gpt-5]"
```

Resolution is handled by your `resolve_model_config` helper.

---

### Supported models

| Provider | Model Identifier            |
| -------- | --------------------------- |
| openai   | gpt-5                       |
| openai   | gpt-5-mini                  |
| openai   | gpt-5-nano                  |
| groq     | openai/gpt-oss-120b         |
| groq     | qwen/qwen3-32b              |
| groq     | moonshotai/kimi-k2-instruct |


### Cycles

`--cycles` or `-c` repeats the eval multiple times per case to check stability.

* `cycles = 1` (default) - single pass
* `cycles = N` - each criterion score is averaged across `N` runs

Example:

```bash
ev eval myAgent -c 3
```

If a criterion is flaky, you will see it reflected in non 100 percent scores.
