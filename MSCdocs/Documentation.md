Here is your structured `.md` documentation file content:

---

# RAG Pipeline Architecture & Change Log

This document serves as a **living architectural record** of major design decisions and structural changes made to the RAG + Generation pipeline.

Every time a significant change is made to the system, it should be documented here along with:

* What changed
* Why it changed
* What problem it solves
* Any trade-offs introduced

---

# 1️⃣ Original System Design

## Architecture Overview

Original pipeline structure:

```
Input Dataset
      ↓
Multiprocessing Pool Workers (N)
      ↓
Each worker independently:
    1. Run RAG
    2. Run Generation
      ↓
Write Output
```

### Characteristics

* Each pool worker triggered its own RAG execution.
* All RAG calls hit a **single GPU endpoint**.
* RAG and Generation were tightly coupled per request.
* No centralized RAG coordination.
* No backpressure control.
* No GPU-aware scaling.

---

## ❌ Flaws in Original Design

### 1. GPU Bottleneck

* Only **1 GPU**
* But multiple pool workers tried to use it simultaneously.
* Led to:

  * Queuing delays
  * Timeouts
  * Unpredictable latency

---

### 2. No Concurrency Control for RAG

* Multiple workers could overload the RAG model.
* No centralized request queue.
* No control over inflight requests.

---

### 3. Tight Coupling Between RAG and Generation

If RAG failed:

* Generation logic was unclear.
* Could skip or inconsistently handle failures.

---

### 4. Not Scalable to Multiple GPUs

If new GPUs were added:

* System would not automatically scale.
* No endpoint replication.
* No load balancing.

---

# 2️⃣ New Architecture (Current Implementation)

We redesigned the pipeline into a **multi-stage, GPU-aware architecture**.

---

## New High-Level Pipeline

```
Input Dataset
      ↓
Shared RAG Request Queue
      ↓
Multiple RAG Worker Processes (1 per GPU)
      ↓
RAG Response Queue
      ↓
Multiprocessing Generation Pool
      ↓
Write Output
```

---

## Stage 1: Shared RAG Request Queue

* All items first go into a **shared multiprocessing queue**
* RAG workers pull from this queue
* This ensures:

  * Controlled concurrency
  * No GPU overload
  * Proper backpressure

---

## Stage 2: RAG Worker Layer (GPU-Aware)

For each detected GPU:

* One `rag_worker_process` is created
* Each worker connects to:

  * `http://127.0.0.1:11434`
  * `http://127.0.0.1:11435`
  * `http://127.0.0.1:11436`
  * etc.

Each GPU runs:

* One independent SGLang server
* One independent model replica
* Tensor parallel size = 1
* Fully independent batching & scheduling

### Benefits

* True data-parallel scaling
* No shared model memory
* No GPU contention
* Near-linear scaling with number of GPUs

---

## Stage 3: Generation Pool (CPU Parallel)

After RAG completes:

* Generation is executed using multiprocessing pool
* Generation parallelism is independent of GPU count
* Scales with `--num_processes`

---

## RAG Failure Handling Strategy

We implemented a **best-of-both-worlds approach**:

### Case 1: RAG Success

* Append retrieved context
* Run generation

### Case 2: Soft Failure

Examples:

* No results found
* Short answer
* Non-critical tool issue

→ Fallback to original query
→ Continue generation

### Case 3: Hard Failure

Examples:

* Timeout
* Connection error
* Server unreachable

→ Retry RAG (limited attempts)
→ If still failing → Skip generation
→ Log status

---

# 3️⃣ Multi-GPU Dynamic Scaling

The system dynamically:

* Detects number of GPUs using `torch.cuda.device_count()`
* Builds endpoint list automatically
* Creates one RAG worker per GPU

No code change required to scale from:

* 1 GPU → 2 GPUs → 4 GPUs → N GPUs

Just start additional model servers on sequential ports.

---

# 4️⃣ Rank0 Initialization & Collection Reset Design

This section documents a critical architectural fix related to ChromaDB.

---

## The Problem

ChromaDB uses persistent collections.

When `reset_collection()` is called:

* It deletes the collection
* Then recreates it

However:

Each worker holds a **collection handle tied to an internal UUID**.

If one worker deletes the collection:

* Other workers still hold stale handles
* This causes:

```
Collection [UUID] does not exist
```

Even though the collection exists.

---

## The Solution: Rank0 Barrier Initialization

We implemented a startup coordination mechanism.

---

### Step 1: Start Rank0 First

Only the first RAG worker (Rank0):

* Calls `reset_collection()`
* Recreates the collection

---

### Step 2: Rank0 Signals READY

Rank0 sends:

```
("READY", endpoint)
```

through `rag_status_q`

---

### Step 3: Main Process Waits

The main process:

* Blocks until Rank0 reports READY
* If Rank0 fails → crash early

---

### Step 4: Start Remaining Workers

Only after Rank0 is READY:

* Other RAG workers are started
* They connect to the already-created collection
* No stale handles occur

---

## Why This Is Necessary

Because:

* Deleting a Chroma collection invalidates existing handles.
* Multiprocessing does not auto-refresh collection references.
* PersistentClient + SQLite backend makes this especially fragile.

This barrier ensures:

* Deterministic initialization
* No race conditions
* No UUID mismatch errors
* Stable startup behavior

---

# 5️⃣ Self-Healing Collection Rebinding

In addition to Rank0 control, we added a safety mechanism:

If `_tracked_retrieve_content()` detects:

```
Collection does not exist
```

It:

1. Rebinds the collection via `get_or_create_collection()`
2. Rebinds dependent tools
3. Retries once

This converts a crash into a recoverable event.

---

## Why This Matters

Multi-process systems must assume:

* External state may change
* Handles may become stale
* Recovery should be automatic

This makes the system production-resilient.

---

# 6️⃣ Current Architecture Summary

### ✔ GPU-Aware

One RAG worker per GPU.

### ✔ Scalable

Auto-detects GPU count.

### ✔ Fault-Tolerant

Handles stale collection references.

### ✔ Deterministic Startup

Rank0 barrier prevents race conditions.

### ✔ Clean Separation

RAG and Generation decoupled.

---

# 7️⃣ Future Changes Section

When making major changes, document:

```
Date:
Change:
Reason:
Impact:
Trade-offs:
```

---

Example Entry:

```
Date: YYYY-MM-DD
Change: Added self-healing collection rebinding.
Reason: Workers crashed due to stale Chroma UUID handles.
Impact: Increased reliability in multi-process environment.
Trade-offs: Slight overhead on first failure detection.
```

KeywordExtractor Fresh Client per Call:

```
Date: 2025-03-18
Change: KeywordExtractor now creates a fresh Client per extract_keywords call.
Reason: Reused Client accumulated messages across calls, causing context overflow, timeouts, and "Unknown error" failures in RAG logs.
Impact: Eliminates message accumulation; each keyword extraction is stateless. Prevents context window overflow and improves reliability.
Trade-offs: Slight overhead from creating a new Client per call (negligible compared to LLM latency).
```

---

# Final Notes

This pipeline is now:

* Multi-GPU scalable
* Backpressure-controlled
* Failure-aware
* Deterministically initialized
* Production-oriented

---

**End of Documentation**

---
