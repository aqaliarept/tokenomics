# Port Go emulator to Python with PettingZoo AEC interface

The existing Go emulator implements the tokenomics simulation as an HTTP REST API. For RL training, we need millions of episodes across thousands of parallel environments. We decided to port the ~400 lines of Go domain logic to pure Python and expose it as a PettingZoo AEC (Agent Environment Cycle) environment, rather than wrapping the Go API from Python.

## Considered Options

**Option A: Python HTTP wrapper over Go API.** Keeps a single source of truth for rules. But each cycle requires 12+ HTTP round-trips (GET/POST per agent), making RL training orders of magnitude slower than in-process calls. Vectorized environments (Gymnasium's `VectorEnv`) would require spawning hundreds of Go server processes on separate ports. Debugging across the language boundary is painful.

**Option B: Pure Python port.** Duplicates the rules, but the domain logic is small (~400 lines of Go, no I/O, no complex dependencies). In-process Python calls eliminate serialization overhead. Standard vectorized environment patterns work natively. PettingZoo AEC integrates directly with MAPPO, RLlib, and TorchRL — the target training frameworks from the research roadmap.

We chose **Option B**. The Go emulator remains the reference implementation for human-facing interaction and validation. The Python port is the RL training environment.

## Why PettingZoo AEC specifically

The simulation has a strict turn order within each cycle: producing agents decide on investments first, then ordering agents distribute tokens, then the environment runs production. PettingZoo's AEC API models exactly this — agents step sequentially in a defined order. The alternative Parallel API assumes simultaneous action, which doesn't match the phased cycle structure. AEC also makes the Go emulator's explicit state machine (`OrdersPlacement → Ordering`) redundant, since the framework enforces turn order.
