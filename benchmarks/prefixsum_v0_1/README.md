# PrefixSum v0.1

A local, deterministic two-agent adaptation of Silo-Bench task II-11 (Prefix Sum), whose upstream benchmark and generator are released under the Unlicense. This subset keeps the core task semantics: each agent owns one private contiguous segment; each must return its own part of the global cumulative sum. For two agents, agent 1's result depends on the exact sum of agent 0's segment.

`generate.py` derives independently seeded cases with segment lengths 6, 15, and 30. Its manifest records each file hash, seed, and length. The generator validates every local output and confirms that agent 1's first expected prefix changes across alternative valid agent-0 segments with the same agent-1 shard, so the local observation is insufficient.

This small suite evaluates protocol transfer beyond a scalar-sum output. It is still synthetic and algorithmic; it does not represent natural workplace communication or multi-party long-horizon tasks.
