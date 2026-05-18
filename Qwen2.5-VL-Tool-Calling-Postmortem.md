# Qwen2.5-VL Tool-Calling Postmortem

**Status:** Blocked. The full 8,188-item benchmark cannot complete on Qwen2.5-VL-(7B|32B)-Instruct with the current `rag_agent` tool set under vLLM 0.10.2 because every guided-decoding backend either crashes the engine or rejects the auto-built tool schema. The path forward is a model swap to the non-VL `Qwen/Qwen2.5-32B-Instruct`.

**Date of investigation:** 2026-05-17 (citations expanded same day to add two HuggingFace discussion threads as independent corroboration)
**Environment:** NCSA Delta (`gpue06`), Python 3.12, vLLM 0.10.2, LangChain agent in `rag_agent/main.py`, mirage conda env

---

## 1. What we were trying to do

Run `Inference/generate.py` against the `standard_without_db` benchmark (8,188 items) using `Qwen/Qwen2.5-VL-7B-Instruct` (later upgraded to `-32B-Instruct`) as the test LLM, served by a local vLLM OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1`. The benchmark exercises a LangChain `create_tool_calling_agent` (`MainAgent` in `rag_agent/main.py`) with six tools:

```
_tracked_retrieve_content
_tracked_evaluate_confidence
_tracked_web_search
_tracked_extract_keywords
_tracked_add_web_content
_tracked_add_pdf_content
```

Each tool exposes several `Optional[str]` parameters (`location`, `month_year`, `title`, etc.) for the agent to fill in opportunistically.

## 2. The symptom

Every item received by the worker failed with the same family of exceptions:

```
[RAG Worker] Item #...: ✗ Exception during RAG: Expecting ':' delimiter: line 1 column 7 (char 6)
[RAG Worker] Item #...: ✗ Exception during RAG: Expecting property name enclosed in double quotes: line 2 column 1 (char 4)
[RAG Worker] Item #...: ✗ Exception during RAG: Expecting ':' delimiter: line 3 column 5 (char 8)
```

All are `json.JSONDecodeError`s raised inside the agent's `rag_runner.run_debug(...)` call. Items are repeatedly retried (`Soft fail → fallback to original query`) and the rerun produces the same or a sibling error. Net effect: 100% per-item failure, no successful retrievals written.

## 3. Investigation timeline

### 3.1 Initial misdiagnosis: "ADK is failing"

We initially assumed Google ADK was the agent framework (printlines superficially looked like ADK output). On reading `rag_agent/main.py` we confirmed the stack is actually **LangChain** (`langchain.agents.create_tool_calling_agent` + `AgentExecutor` + `langchain_openai.ChatOpenAI`). The `_RunnerShim` class in `main.py` is an ADK-shaped wrapper around a LangChain `AgentExecutor` to preserve the `await runner.run_debug(...)` contract used by `Inference/generate.py`.

### 3.2 First hypothesis: model emits malformed JSON, parser doesn't recover

The error positions (`char 6`, `char 4`, etc.) are consistent with a model emitting:

- Python-dict syntax: `{'name': 'retrieve_content', ...}` (single quotes → "property name enclosed in double quotes" error)
- Markdown code fences: `` ```json\n{...}\n``` `` (first four chars are `` ``` `` + `\n` → "line 2 column 1 char 4")
- Bare-key JSON: `{name: "retrieve_content", ...}` (missing quotes around key)
- Missing-colon JSON: `{"name retrieve_content", ...}` (line 1 col 7 char 6)

Where the `json.loads` fires turned out to be **inside LangChain's `create_tool_calling_agent`**: when LangChain receives an OpenAI-format `tool_calls` array, it calls `json.loads()` on each `arguments` *string* before invoking the tool. `AgentExecutor(handle_parsing_errors=True)` does **not** catch this, because the failure is in argument deserialization, not in output parsing.

### 3.3 Direct curl test of vLLM revealed the *real* immediate problem

A minimal one-tool curl against the running vLLM server returned:

```json
{
  "content": "I'm sorry, as an AI language model, I don't have direct access ... \"retrieve_content\" ...",
  "tool_calls": [],
  "usage": { "prompt_tokens": 32, "total_tokens": 199 }
}
```

Two telling facts:

- `prompt_tokens: 32` is far too small. A properly rendered Qwen2.5 tool prompt with one tool should be ~180–260 tokens.
- The model says it doesn't have access to `retrieve_content`.

**Conclusion:** the tools were never injected into the prompt. The chat template baked into `Qwen/Qwen2.5-VL-7B-Instruct`'s `tokenizer_config.json` does not contain the `{% if tools %}` branch — the Qwen team intentionally stripped it ([commit reference](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/commit/d91279c190bb874c1f90cf26c70c4261bbf7488c)).

### 3.4 Fix: supply a tool-aware chat template

We installed the community Qwen2.5-VL tool-aware template authored by `@edwardzjl` (commit `30561e775c867ed6f74f930f64bcca53ce97eb29`), saved to `/u/ssingh38/chat_templates/qwen25_vl_tool.jinja`, and relaunched vLLM with:

```
--chat-template /u/ssingh38/chat_templates/qwen25_vl_tool.jinja
--chat-template-content-format openai
```

#### False step 1: downloaded a 404 HTML page as the template

First attempt used `curl -L` against `main` on the vLLM repo for `tool_chat_template_qwen2_5.jinja`. GitHub returned a 200-status HTML "Not Found" page (`14 bytes` containing `404: Not Found`). vLLM loaded that as the template, and the curl response then showed `prompt_tokens: 6` and the model hallucinated a generic "404 not found" page in English+Chinese. Resolved by writing the template via heredoc and verifying file size / first line / `grep -c "<tool_call>"`.

#### After the template was installed correctly

Curl response on a one-tool simple query:

```json
{
  "finish_reason": "tool_calls",
  "tool_calls": [{
    "function": {
      "name": "retrieve_content",
      "arguments": "{\"query\": \"corn diseases in Minnesota\"}"
    }
  }],
  "usage": { "prompt_tokens": 177 }
}
```

Clean. Model emits valid tool calls on simple prompts.

### 3.5 Re-running the full pipeline: same failures

Reran `generate.py`. Got the **exact same** `json.JSONDecodeError` messages from item #1 onwards. The curl test had passed on 1 tool and a 14-character user message; the agent fails on 6 tools and 200–700-char queries.

Diagnosis: **for complex prompts, Qwen2.5-VL still emits well-formed `<tool_call>...</tool_call>` tags but with malformed JSON *inside* them.** vLLM's `hermes` tool-call parser dutifully extracts the (malformed) string as `arguments`. LangChain then crashes on `json.loads(arguments)`.

This is consistent with the upstream issue thread on the Qwen team's own repo ([QwenLM/Qwen3-VL#1093](https://github.com/QwenLM/Qwen3-VL/issues/1093), closed Oct 2025), where community member `@huaiyizhao` summarized:

> "The Qwen2.5-VL series ARE NOT TRAINED to perform tool calls. 'Changing the chat template' does not work as expected because the models are not TRAINED to do so. Nonetheless, since the model has basic ... understanding of English, it may TRY to call tools based on the semantics of the new chat template, but the success rate is not high and STRANGE things (emojis, addCriterion ...) can happen."

### 3.6 Switching 7B-VL → 32B-VL

Per the same issue thread, 32B-VL "performs better than the VL-72B when it comes to tool usage" and one commenter reported the @edwardzjl template "worked." Switched the model to `Qwen/Qwen2.5-VL-32B-Instruct`. Simple curl test was clean. **The full pipeline still produced the same `Expecting ':' delimiter: line 1 column 7 (char 6)` errors.** Model size alone is not the fix.

### 3.7 First attempt at structured-output enforcement: `--guided-decoding-backend xgrammar`

Idea: constrain the model's *tokens* so the JSON inside `<tool_call>` cannot be malformed. Added `--guided-decoding-backend xgrammar` to vLLM, plus `temperature=0` and `extra_body={"guided_decoding_backend": "xgrammar"}` in `ChatOpenAI`, plus a tolerant `json.loads` shim in `_RunnerShim.run_debug`. Result on vLLM 0.10.2:

```
nanobind: leaked 2 instances!
 - leaked instance of type "xgrammar.xgrammar_bindings.GrammarMatcher"
 - leaked instance of type "xgrammar.xgrammar_bindings.CompiledGrammar"
EngineCore encountered an issue.
vllm.v1.engine.exceptions.EngineDeadError
```

Engine **died entirely**. After the crash, every subsequent request returned HTTP 500 forever (API server stays up, but no engine behind it). Root cause: a ref-counting bug in xgrammar's nanobind layer, triggered by the schema vLLM auto-builds for `tool_choice="required"` with our six tools.

### 3.8 Second attempt: `--guided-decoding-backend outlines`

Switched backend. Engine started, simple curl passed twice + a stress-test curl (long agent-style query). Full pipeline restart → first 4 items hit the *same* `Expecting ':' delimiter: line 1 column 7 (char 6)` errors as before, then the engine crashed with:

```
torch.AcceleratorError: CUDA error: operation not supported on global/shared address space
File ".../v1/worker/gpu_model_runner.py", line 3742, in _to_list
    self.transfer_event.synchronize()
```

The engine-side `dump_input` showed the request was using guided decoding correctly:

```python
guided_decoding=GuidedDecodingParams(
    json={'type': 'array', 'minItems': 1, 'items': {'type': 'object', 'anyOf': [...]}},
    backend='outlines',
    ...
)
```

The crash is in `_bookkeeping_sync` while CUDA-syncing the `grammar_bitmask` produced by outlines for this schema. Different bug from xgrammar (CUDA-side vs binding-side) but same triggering schema.

Two important observations from this run:

1. **The same `partial_json_parser.loads` error appeared in vLLM's own logs at ERROR level but the request still returned 200 OK.** This is `extract_tool_call_required_streaming` parsing partial streamed JSON as it accumulates; the partial parser is supposed to throw on incomplete input and be retried. vLLM 0.10.2 logs it noisily but does not actually fail the request. So this is log noise, not the root cause of the agent-side failures.
2. **The first 4 items failed with the *same* `char 6` JSON error as before guided decoding was enabled.** That means outlines was not actually enforcing the schema on the `arguments` field for those items — either the auto-tool-choice path didn't wire guided decoding through, or outlines failed schema compilation silently. The crash then prevented further investigation.

### 3.9 Third attempt: `--guided-decoding-backend lm-format-enforcer`

LMFE uses a completely different implementation path (pure Python token-trie state machine, no nanobind, no CUDA bitmask transfer), so neither of the previous crash classes can recur. Relaunched with LMFE. Simple curl returned:

```
HTTP/1.1 400 Bad Request
```

Server-side traceback:

```
File ".../lmformatenforcer/jsonschemaparser.py", line 182, in get_parser
    parsers = [get_parser(parsing_state, schema) for schema in value_schema.anyOf]
File ".../lmformatenforcer/jsonschemaparser.py", line 269, in get_parser
    raise Exception("Unsupported type " + str(value_schema.type))
Exception: Unsupported type None
```

LMFE could not compile the schema. The error chain:

1. Tools like `_tracked_retrieve_content` have parameters declared as `Optional[str] = None`.
2. LangChain's `StructuredTool.from_function` converts `Optional[str]` to a JSON schema fragment of `{"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}`. This is the standard JSON-Schema representation of a nullable field.
3. vLLM passes that schema verbatim to LMFE.
4. LMFE walks the `anyOf` to build a token-level parser.
5. LMFE has parsers registered for `string`, `integer`, `number`, `boolean`, `object`, `array` — but **not** for `"null"`.
6. It throws `Unsupported type None` (Pydantic converts the JSON string `"null"` to Python `None` when materializing the schema model, so the error prints `None` rather than `null`).

Key positive: LMFE returned a **400** (graceful) instead of crashing the engine. This is the cleanest failure of the three backends. But the request still cannot be served.

## 4. Root causes — three layers, all real

The investigation surfaced three independent problems that compound. None of them in isolation explains the full failure pattern; all three are required:

### Root cause A — Qwen2.5-VL is not trained for tool calling

Strongly supported by three independent upstream reports — one on the Qwen team's GitHub issue tracker and two on the official Qwen HuggingFace model pages:

1. [**QwenLM/Qwen3-VL#1093**](https://github.com/QwenLM/Qwen3-VL/issues/1093) — *"Tool Call Issues with Qwen2.5-VL Models (7B & 72B) under vLLM"*. Opened April 11, 2025 by `@edwardzjl`; closed October 10, 2025 by Qwen team member `@ShuaiBai623`. The most informative summary in the thread, from `@huaiyizhao` (Oct 13, 2025): *"The Qwen2.5-VL series ARE NOT TRAINED to perform tool calls. 'Changing the chat template' does not work as expected because the models are not TRAINED to do so."* The same thread also documents that the *non-VL* `Qwen2.5-7B-Instruct` / `Qwen2.5-72B-Instruct` work correctly on identical code paths (per `@edwardzjl`), isolating the regression to VL post-training specifically.

2. [**Qwen/Qwen2.5-VL-7B-Instruct discussion #39**](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/39) — *"Chat template does not work"* by `@tonydavis629` (Apr 2, 2025): *"The chat template in chat_template.json is out of date. It does not include tools ... The tokenizer_config.json has a different, but also incorrect, chat template. The tokenizer template includes tools, but is expecting an out of date message format where content can be a string, instead of a list of dicts as is the new standard format. As a result, there is no way to use this model with tools."* Independent confirmation from a different HF user on the official 7B model page itself.

3. [**Qwen/Qwen2.5-VL-32B-Instruct-AWQ discussion #10**](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ/discussions/10) — *"(vLLM) Tool calling broken after update to tokenizer_config.json"* by `@m1das13` (Apr 11, 2025): identifies the specific commit (`66c370b`) that removed tool support from the 32B-AWQ template, and reports a workaround (`--tokenizer-revision 05440b7` to pin to the pre-strip tokenizer). Follow-up by `@maleal` (Sep 6, 2025) on vLLM 0.10.1.1: *"Qwen 2.5 VL is a great model but tool calling works only with 'required' in vllm."* — exactly the constraint our agent uses (`tool_choice="required"` in `rag_agent/main.py`). `@maleal` also reports that the `--tokenizer-revision` workaround has since stopped working on newer vLLM (raises `ValueError: Unrecognized model in Qwen/Qwen2.5-VL-72B-Instruct-AWQ`).

These three reports cross-confirm that:

- The official VL `tokenizer_config.json` has had its tool-call sections stripped on multiple checkpoints since at least April 2025.
- This was a deliberate Qwen-team action: see [HuggingFace commit `d91279c1` on `Qwen/Qwen2.5-VL-72B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/commit/d91279c190bb874c1f90cf26c70c4261bbf7488c) and the analogous `66c370b` on the 32B-AWQ.
- Multiple users on multiple model sizes (7B, 32B-AWQ, 72B) and multiple vLLM versions (0.8.2, 0.10.1.1, 0.10.2) reproduce the same failure modes (emoji closing tags, malformed JSON, `addCriterion` spew).
- The non-VL text Qwen2.5 models work on identical code paths.

These statements are not literal Alibaba/Qwen-team policy declarations — but the combination of (i) the deliberate template-stripping commits on multiple model cards, (ii) the GitHub issue being closed without a fix or refutation, and (iii) consistent behavior reported by multiple independent users across model sizes and vLLM versions, makes it the most defensible interpretation of the available evidence.

Even with the `@edwardzjl` tool-aware template installed, VL-7B/32B produce well-formed tool calls on **simple** prompts but degrade under realistic load (multiple tools, long instruction, multi-paragraph user query). Failure modes documented in the upstream threads include:

- Closing tags as emoji (`📐`, `⚗`) instead of `</tool_call>`
- Token spew like `<tool_call>\naddCriterion\n<tool_call>\n\n\n\n\n`
- Python-dict syntax inside the tags
- Markdown code fences inside the tags

### Root cause B — `Optional[X]` in tool signatures produces a schema all three vLLM 0.10.2 backends choke on

The schema vLLM auto-builds when `tool_choice="required"` is set, for our six tools with `Optional[location|month_year|title|...]`, is roughly:

```python
{
  'type': 'array', 'minItems': 1,
  'items': {
    'type': 'object',
    'anyOf': [
      { 'properties': { 'name': {'enum': ['_tracked_retrieve_content']},
                        'parameters': {
                            'location':   {'anyOf': [{'type':'string'},{'type':'null'}], 'default': None},
                            'month_year': {'anyOf': [{'type':'string'},{'type':'null'}], 'default': None},
                            ...
                        } } },
      { 'properties': { 'name': {'enum': ['_tracked_evaluate_confidence']}, ... } },
      ... (4 more tool branches)
    ]
  }
}
```

Empirical results on vLLM 0.10.2:

| Backend | Schema compile | Inference behavior |
|---|---|---|
| `xgrammar` | succeeds | crashes engine via nanobind `GrammarMatcher`/`CompiledGrammar` leak |
| `outlines` | succeeds | crashes engine on CUDA `grammar_bitmask` `transfer_event.synchronize()` |
| `lm-format-enforcer` | **fails** with `Unsupported type None` | engine stays alive, request 400s |

The nested `anyOf` (tool-level *and* parameter-level) plus `{"type": "null"}` branches is the common trigger. Different backends, different failure modes, **same triggering schema**.

### Root cause C — LangChain's tool-call argument deserializer is unprotected

`AgentExecutor(handle_parsing_errors=True)` does not wrap the `json.loads(tool_call.arguments)` step performed by `create_tool_calling_agent`. So any malformed `arguments` string from the model bubbles up to `generate.py`'s `except Exception as e:` block and the item is recorded as a failure. This is the proximate cause of every `Expecting ':' delimiter` and `Expecting property name enclosed in double quotes` line in the log.

A monkey-patched tolerant `json.loads` shim was added to `_RunnerShim.run_debug` as a safety net. It works for the malformed-JSON case but does not help when the engine itself is dead (HTTP 500) or when the schema cannot be compiled (HTTP 400).

## 5. Things we tried, with outcomes

| Attempt | Change | Outcome |
|---|---|---|
| 1 | Baseline launch with `--enable-auto-tool-choice --tool-call-parser hermes` | 100% failure: tools not even rendered into prompt (chat template lacks `{% if tools %}` branch) |
| 2 | Pulled `tool_chat_template_qwen2_5.jinja` from vLLM `main` | File was a 404 HTML page (14 bytes). Made things worse: prompt collapsed to 6 tokens, model hallucinated. |
| 3 | Installed @edwardzjl Qwen2.5-VL tool-aware chat template via heredoc | Simple 1-tool curl: works (`prompt_tokens` 32 → 177, valid `tool_calls`). Full pipeline: same `json.JSONDecodeError`s as before. |
| 4 | Switched model 7B-VL → 32B-VL | Simple 1-tool curl: works. Full pipeline: same errors. |
| 5 | Added `--guided-decoding-backend xgrammar`, `temperature=0`, `extra_body`, tolerant `json.loads` shim | Engine crashes after a few items via nanobind ref-counting bug. All subsequent requests 500. |
| 6 | Switched to `--guided-decoding-backend outlines` | Engine crashes via CUDA `transfer_event.synchronize()` error on `grammar_bitmask`. Also: first 4 items still hit the `char 6` JSON error, suggesting guided decoding wasn't being applied to those items. |
| 7 | Switched to `--guided-decoding-backend lm-format-enforcer` | Engine survives (good). Schema compile fails with `Unsupported type None` because LMFE can't handle the `{"type":"null"}` branch produced by `Optional[X]` in our tool signatures. All tool-call requests 400. |

Server hygiene flags that *should remain* on any future relaunch regardless of model: `--gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 16` — caps KV-cache pressure under multi-tool agent loops, prevents OOM as a secondary failure mode.

## 6. Code changes made during the investigation

These are still in the tree:

1. **`rag_agent/main.py`** — added `temperature=0` and `model_kwargs.tool_choice="required"` to `ChatOpenAI`. The `extra_body={"guided_decoding_backend": "..."}` is currently set; the value should be revisited whichever path forward is taken.
2. **`rag_agent/main.py`** — `_RunnerShim.run_debug` was augmented with a tolerant `json.loads` shim that monkey-patches `json.loads` for the duration of `_executor.ainvoke(...)`. Falls back to stripping markdown fences and then to `ast.literal_eval`. Safe to keep regardless of path forward.
3. **vLLM launch command** — various flags added/removed across iterations: `--chat-template`, `--chat-template-content-format openai`, `--guided-decoding-backend`, `--gpu-memory-utilization`, `--max-model-len`, `--max-num-seqs`.
4. **`/u/ssingh38/chat_templates/qwen25_vl_tool.jinja`** — installed the @edwardzjl Qwen2.5-VL tool-aware template via heredoc (88 lines, ~3 KB). Only needed for VL models; redundant on text Qwen2.5.

## 7. The fence

After (1) installing a verified tool-aware chat template, (2) testing both 7B and 32B VL checkpoints, (3) trying all three of vLLM 0.10.2's guided-decoding backends, and (4) layering tolerant client-side parsing, the failure mode is no longer one bug — it is the intersection of three things that we cannot fix without changing the model or the tool signatures:

1. **Qwen2.5-VL was not trained for tool calling.** No prompting or chat-template fix turns this around. Cross-confirmed by three independent upstream reports (see §4-A).
2. **vLLM 0.10.2 cannot reliably guide-decode a schema with nullable parameters across multiple tools.** xgrammar crashes the engine, outlines crashes the engine, LMFE rejects the schema. The root trigger — `anyOf: [..., {"type": "null"}]` from `Optional[X]` in Python signatures wrapped in an outer per-tool `anyOf` — is in every request our agent sends.
3. **LangChain's tool-call argument deserialization is not protected by `handle_parsing_errors`.** Any malformed `arguments` string from the model becomes an unhandled exception. We have a monkey-patch that mitigates this but cannot save us when the engine is dead.

Each of these three is independently a known limitation. Together they form a fence: there is no flag combination in the current stack that produces a working VL-based pipeline for this benchmark.

## 8. Recommended path forward

### Primary (recommended): swap to non-VL Qwen2.5-32B-Instruct

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct \
  --served-model-name Qwen/Qwen2.5-VL-32B-Instruct \
  --host 127.0.0.1 --port 11434 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --trust-remote-code
```

Notes:

- The `--served-model-name Qwen/Qwen2.5-VL-32B-Instruct` alias means **no code changes** in `generate.py` / `MainAgent`. The server answers to the old name.
- **No `--chat-template`.** The text Qwen2.5 ships with a tools-aware template in its `tokenizer_config.json`.
- **No `--guided-decoding-backend`.** The text model was trained for tool calling and keeps the `<tool_call>` JSON well-formed without external constraints, so the failure paths in §4-B all disappear.
- Keep `temperature=0` and the tolerant `json.loads` shim as belt-and-suspenders.
- Same VRAM (~64 GB BF16 or ~22 GB AWQ-int4), same tokenizer family.
- The only thing lost is image understanding, which the RAG worker is not using (queries are pure text).

A defensible methodology footnote for any writeup: *"VL backbone tested in standard_without_db; tool-call layer required substitution to text-only Qwen2.5-32B-Instruct due to known Qwen2.5-VL function-calling limitations (QwenLM/Qwen3-VL issue #1093, plus Qwen2.5-VL-7B-Instruct HF discussion #39 and Qwen2.5-VL-32B-Instruct-AWQ HF discussion #10) and vLLM 0.10.2 structured-output backend incompatibilities with the agent's tool schema."*

### Secondary (only if a VL backbone is non-negotiable): drop the schema-forcing path on VL-32B

```python
# rag_agent/main.py
llm = ChatOpenAI(
    model=SGLANG_MODEL,
    base_url=SGLANG_BASE_URL,
    api_key=API_KEY,
    temperature=0,
    model_kwargs={"tool_choice": "auto"},   # was "required"
)
```

Relaunch vLLM without `--guided-decoding-backend`. The engine no longer has a schema to compile, so the §4-B failure modes vanish. Tool-call success depends entirely on the model. Expect a meaningful per-item failure rate; the tolerant `json.loads` shim is what makes the run complete instead of crash.

(Note from `@maleal` in HF discussion #10: on vLLM 0.10.1.1, `tool_choice="auto"` consistently failed on VL-32B-AWQ — only `"required"` worked. So this Secondary path may have its own VL-specific failures even before our backend issues compound; the only known way to make `auto` work on VL is to fall back to a *very old* tokenizer revision that no longer loads on current vLLM. This is the main reason Option B is rated ~60% rather than ~90%.)

### Tertiary (high engineering cost, high regret risk): remove `Optional` from tool signatures

Change every `_tracked_*` parameter declared as `Optional[X] = None` to a sentinel default (e.g. `str = ""`) and update the function bodies to check `if location:` instead of `if location is not None:`. This eliminates the `{"type":"null"}` branches in the auto-built schema, which would (likely) unblock LMFE. Then `--guided-decoding-backend lm-format-enforcer` with `tool_choice="required"` can be re-enabled.

The reason this is high-regret: if any downstream code in `MainAgent` distinguishes "absent" from "empty" semantically, the change will produce subtly wrong benchmark answers, not loud crashes. Touching tool signatures mid-evaluation is the sort of change that should be paired with a regression test.

## 9. Appendix: things that *did* work, kept as reference

- **Direct vLLM curl tests on simple 1-tool prompts** with VL-32B + @edwardzjl chat template + `outlines` backend: clean. `prompt_tokens` ≈ 200–260, `finish_reason: "tool_calls"`, valid `arguments` JSON. This is what made it tempting to keep iterating on the VL path.
- **`--served-model-name` aliasing**: vLLM accepts multiple aliases (`--served-model-name foo bar`) and answers to both. Useful for swapping the underlying model without touching client code.
- **Heredoc-installed Jinja template + 4-line verification (`wc -l`, `head -3`, `grep -c "<tool_call>"`, `file`)**: caught the 404-page-as-template failure mode immediately.
- **`curl -fL`** (with `-f`): fails on HTTP 4xx/5xx instead of saving the error body as a "successful" download. Should be used by default when fetching template files.

## 10. Appendix: external references

### Primary evidence — Qwen team and HuggingFace community

- [**QwenLM/Qwen3-VL issue #1093 — "Tool Call Issues with Qwen2.5-VL Models (7B & 72B) under vLLM"**](https://github.com/QwenLM/Qwen3-VL/issues/1093). Opened Apr 2025 by `@edwardzjl`, closed Oct 2025 by Qwen team member `@ShuaiBai623`. Contains the `@huaiyizhao` "VL series are not trained for tool calls" verdict, plus extensive failure-mode documentation (emoji closing tags, `addCriterion` spew, `tool_choice="required"` mitigation, `@edwardzjl`'s custom chat template).
- [**HuggingFace `Qwen/Qwen2.5-VL-7B-Instruct` discussion #39 — "Chat template does not work"**](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/39). By `@tonydavis629`, Apr 2, 2025. Independent confirmation on the official 7B model page that *both* `chat_template.json` and `tokenizer_config.json` ship broken-for-tools templates: *"there is no way to use this model with tools."*
- [**HuggingFace `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` discussion #10 — "(vLLM) Tool calling broken after update to tokenizer_config.json"**](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ/discussions/10). By `@m1das13`, Apr 11, 2025. Identifies specific commit `66c370b` that removed tool support from the 32B-AWQ template and documents the `--tokenizer-revision 05440b7` workaround (which has since stopped working per `@maleal`'s Sep 2025 follow-up on vLLM 0.10.1.1). Also corroborates that `tool_choice="required"` is the only working `tool_choice` value for VL on current vLLM.
- [**HuggingFace commit `d91279c1` on `Qwen/Qwen2.5-VL-72B-Instruct`**](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/commit/d91279c190bb874c1f90cf26c70c4261bbf7488c) — the Qwen team's commit that stripped tool-calling sections from the 72B `tokenizer_config.json`. Direct, authoritative, documents the deliberate nature of the regression.

### Supporting references

- [**@edwardzjl Qwen2.5-VL tool-aware chat template (commit `30561e7...`)**](https://github.com/edwardzjl/chat-templates/blob/30561e775c867ed6f74f930f64bcca53ce97eb29/qwen2_5/chat_template.jinja) — the community-built Jinja template we installed in §3.4.
- vLLM structured-output backends documented in `vllm.engine.arg_utils` — `xgrammar`, `outlines`, `lm-format-enforcer`, `guidance`, `auto`.
- `lm-format-enforcer` JSON schema parser source: `lmformatenforcer/jsonschemaparser.py` — note the explicit dispatch in `get_parser` lacking a `null` branch (this is what produced our `Unsupported type None` error in §3.9).
