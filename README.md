# spice-agent

A minimal LLM agent that designs analog circuits by driving the **ngspice** simulator through tool calling.

The model is given one tool, `run_ngspice`. It writes a netlist, runs the simulation, reads the measured result, and adjusts component values in a loop until the design meets the spec. This is a small-scale version of the agent-driven EDA flow in *TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules* (Liang & Nagulu, 2026), where an LLM agent drives Cadence Virtuoso/Spectre through [virtuoso-bridge-lite](https://github.com/Arcadia-1/virtuoso-bridge-lite).

## How it works

```
task + tool description ──► LLM ──► tool call (netlist)
                             ▲              │
                             │              ▼
                     simulation output ◄── run_ngspice() ── ngspice
```

1. The LLM receives the design task and a description of the `run_ngspice` tool.
2. It replies with a tool call containing a SPICE netlist.
3. Python runs ngspice in batch mode and returns the output (including `meas` results).
4. The LLM reads the result and either adjusts the design and simulates again, or reports the final design.

## Results

### Example 1: RC low-pass, 1 kHz cutoff, C = 100 nF (±2%)
- Chosen R: 1592 Ω
- Measured cutoff: 999.74 Hz (−0.03% error)
- Simulations: 1
- Log: [`sample_run.txt`](sample_run.txt)

### Example 2: RC low-pass with 600 Ω source, 10 kΩ load, E12 values only (±0.5%)
- Final design: 180 Ω series resistor, 0.22 µF capacitor
- Measured cutoff: 999.65 Hz (−0.035% error)
- Simulations: 4
- Log: [`sample_run_2.txt`](sample_run_2.txt)

**Observations:** In Example 2 the agent's first estimate ignored the 10 kΩ load, so it missed by 7.6% and had to iterate. Its search was not monotonic (it tried 130 Ω after 150 Ω was already too high), but simulation feedback kept it converging to a correct design.

## Setup

```bash
sudo apt install ngspice
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LLM_API_KEY="your-key"
python -u spice_agent_free.py | tee sample_run.txt
```

The script uses the OpenAI-compatible Chat Completions API, so it works with Groq (default) or Google Gemini by changing `BASE_URL` and `MODEL` at the top of the file.

## Notes

- Older ngspice versions fail on `.meas` lines that use functions like `vm()` or `vdb()`. The task prompt therefore instructs the model to put the analysis inside a `.control` block.
- Each API call has a 90 s timeout, and the loop is capped at `MAX_STEPS` iteration
