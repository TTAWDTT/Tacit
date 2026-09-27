# PrefixSum v0.2

A fresh-seed, role-explicit adaptation of Silo-Bench task II-11 (Prefix Sum), whose upstream benchmark and generator are released under the Unlicense. Each agent owns one private contiguous segment and returns its own part of the global cumulative sum. The agent prompts explicitly assign Agent 0 as the sender (must message before submitting) and Agent 1 as the receiver (must receive first and never send).

`generate.py` derives fresh independently seeded cases with segment lengths 6, 15, and 30, distinct from the v0.7 audit attempts. Its manifest records each file hash, seed, and length. The generator validates every local output and confirms that agent 1's first expected prefix changes across alternative valid agent-0 segments with the same agent-1 shard, so the local observation is insufficient.

This small suite evaluates protocol transfer beyond a scalar-sum output. It is still synthetic and algorithmic; it does not represent natural workplace communication or multi-party long-horizon tasks.
