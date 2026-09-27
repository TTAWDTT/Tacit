# v0.18 interrupted by local inference stall

The parser-adapter diagnostic was stopped before the first Director response completed. The local Qwen3-14B request reached 54 decoded tokens and then made no further progress for several minutes while the GPU remained near full utilization. No complete response was parsed, delivered, or scored, so v0.18 has no task or parser outcome.

This is a hardware-throughput failure under the pinned local setup, distinct from v0.17's confirmed parser bug. The model server and partial trace remain under ignored `.cache/`; no output from the incomplete request is published. The next feasibility run uses the already downloaded Qwen3-8B Q4_K_M artifact, previously loaded with full GPU offload in this workspace, and is separately preregistered. Do not combine its outcome with either 14B attempt.
