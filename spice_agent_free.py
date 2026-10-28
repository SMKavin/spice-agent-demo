import json
import os
import subprocess
import tempfile

from openai import OpenAI

# Gemini:
#BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
#MODEL = "gemini-3.8-flash"
# Groq - uncomment these two lines instead:
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"

# timeout: give up on a stuck request after 90 s instead of the default 10 min
client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ["LLM_API_KEY"],
    timeout=90,
    max_retries=2,
)
MAX_STEPS = 10
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_ngspice",
            "description": (
                "Run a SPICE netlist in ngspice batch mode and return the simulator "
                "output, including any .meas results. The netlist must be complete "
                "(title line first, ending with .end). Use .meas statements to print "
                "the values you need to check the spec."
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
            ["ngspice", "-b", path], capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
    except FileNotFoundError:
        output = "ERROR: ngspice not found. Install it and make sure it is on PATH."
    except subprocess.TimeoutExpired:
        output = "ERROR: simulation timed out after 60 s."
    finally:
        os.remove(path)
    return output[-4000:]


# --- The design task -------------------------------------------------------------
TASK = """Design an RC low-pass filter with a -3 dB cutoff of 1 kHz (within 0.5%).
The source has 600 ohm internal resistance, and the output drives a 10 kohm load.
R and C must be standard E12 values (e.g. 1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2 x 10^n).
You may combine at most two resistors in series to hit the spec.
Simulate, measure, and adjust until the spec is met. Report the final component
values, measured cutoff, error %, and number of simulations.

Procedure:
1. Compute an initial value of R from theory.
2. Write an ngspice netlist with an AC sweep and a meas command that finds the
   frequency where the output magnitude drops to 0.7071 of its low-frequency value.
3. Run it with the run_ngspice tool and read the measured cutoff.
4. Adjust R and re-simulate until the cutoff is within spec.

Important ngspice formatting rule (this ngspice version fails on .meas lines that
use functions like vdb() or vm()): put the analysis inside a .control block, with
no leading dots on the commands, and use magnitude instead of dB. For example:
  .control
  ac dec 100 1 1Meg
  meas ac f3db when vm(out)=0.7071
  .endc
  .end

Finally, report the chosen R, the measured cutoff, the error in %, and how many
simulations you ran."""


def main():
    messages = [{"role": "user", "content": TASK}]
    sim_count = 0

    for step in range(MAX_STEPS):
        print(f"\n... calling {MODEL} (step {step + 1}) ...", flush=True)
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message
        messages.append(msg)

        if msg.content:
            print(f"\n[LLM]\n{msg.content}")

        if not msg.tool_calls:
            break  # model is done

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