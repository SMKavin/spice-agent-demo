"""
spice_agent_sky130.py - LLM agent that sizes a SKY130 common-source amplifier
by driving ngspice. Same agent loop as spice_agent_free.py; only the task changes.

Run INSIDE the IIC-OSIC-TOOLS container (it has the SKY130 models):
  cd /foss/designs/spice-agent
  pip install openai                 (inside the container)
  export LLM_API_KEY="your-groq-key"
  python -u spice_agent_sky130.py | tee sample_run_sky130.txt

Check the PDK path first with: echo $PDK_ROOT
If it is not /foss/pdks, update SKY130_LIB below.
"""

import json
import os
import subprocess
import tempfile
import time

from openai import OpenAI, RateLimitError

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"

SKY130_LIB = "/foss/pdks/sky130A/libs.tech/ngspice/sky130.lib.spice"

client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ["LLM_API_KEY"],
    timeout=90,
    max_retries=2,
)
MAX_STEPS = 12

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_ngspice",
            "description": (
                "Run a SPICE netlist in ngspice batch mode and return the simulator "
                "output. The netlist must be complete (title line first, ending with "
                ".end). Put analyses, print and meas commands inside a .control block."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "netlist": {"type": "string", "description": "Full SPICE netlist text"}
                },
                "required": ["netlist"],
            },
        },
    }
]


def run_ngspice(netlist: str) -> str:
    """Write the netlist to a temp file, run ngspice, return its output."""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist)
        path = f.name
    try:
        result = subprocess.run(
            ["ngspice", "-b", path], capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        output = "ERROR: ngspice not found. Install it and make sure it is on PATH."
    except subprocess.TimeoutExpired:
        output = "ERROR: simulation timed out after 120 s."
    finally:
        os.remove(path)
    # Send the model only the useful lines (results and errors), not the
    # ngspice banner, to keep each request small (free tiers limit tokens/min).
    keep = [
        line for line in output.splitlines()
        if "=" in line or "rror" in line or "arning" in line
    ]
    return "\n".join(keep)[-1500:] or output[-1500:]


TASK = f"""Design a resistively loaded common-source amplifier in the SKY130 process.

Circuit: NMOS sky130_fd_pr__nfet_01v8 with source and body at ground, drain to
node out, a load resistor RD from VDD to out, and the gate driven by a DC bias
VBIAS plus a 1 V AC small-signal source. VDD = 1.8 V.

Specifications (all must be met):
- Small-signal gain magnitude at 1 kHz between 10 and 12 V/V
- DC output voltage between 0.8 V and 1.0 V
- Supply current at most 100 uA

Design variables: transistor W (in um, between 0.5 and 20), L (in um, between 0.15
and 2), RD, and VBIAS. Use plain numbers in the netlist, no parameters or braces.

Procedure:
1. Estimate an initial design from square-law or gm/Id reasoning.
2. Simulate, check every spec, and adjust the design until all specs are met.
3. Report the final W, L, RD, VBIAS, the measured gain, DC output voltage,
   supply current, and the number of simulations.

Use this netlist template (change only the numbers):
* SKY130 common-source amplifier
.lib {SKY130_LIB} tt
Vdd vdd 0 1.8
Vin in 0 DC 0.6 AC 1
RD vdd out 20k
XM1 out in 0 0 sky130_fd_pr__nfet_01v8 L=0.5 W=2
.control
op
print v(out) i(Vdd)
ac dec 20 10 1Meg
meas ac gain find vm(out) at=1k
.endc
.end

Notes: i(Vdd) is printed as a negative number; its magnitude is the supply current.
Keep the .control block format exactly; this ngspice setup fails on .meas lines in
the netlist body."""


def main():
    messages = [{"role": "user", "content": TASK}]
    sim_count = 0

    for step in range(MAX_STEPS):
        print(f"\n... calling {MODEL} (step {step + 1}) ...", flush=True)
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=TOOLS
                )
                break
            except RateLimitError:
                print("    rate limited, waiting 30 s ...", flush=True)
                time.sleep(30)
        else:
            print("Gave up after repeated rate limits.")
            return
        msg = response.choices[0].message
        messages.append(msg)

        if msg.content:
            print(f"\n[LLM]\n{msg.content}")

        if not msg.tool_calls:
            break

        for call in msg.tool_calls:
            sim_count += 1
            args = json.loads(call.function.arguments)
            netlist = args["netlist"]
            print(f"\n[Simulation {sim_count}: netlist]\n{netlist}")
            output = run_ngspice(netlist)
            print(f"[Simulation {sim_count}: ngspice output tail]\n{output[-800:]}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )
    else:
        print("\nStopped: reached MAX_STEPS without the model finishing.")

    print(f"\nTotal simulations run: {sim_count}")


if __name__ == "__main__":
    main()