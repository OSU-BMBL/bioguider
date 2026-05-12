"""System test for ConsistencyCollectionStep — the new command-line-invocation
collection.  This calls a real LLM, so it is a smoke test: it asserts the step
returns well-formed ``cli_invocations`` (Python *and* R invocations are picked
up) alongside the usual function/class collection, and logs what was collected
so a human can eyeball the extraction quality.
"""

import logging
import textwrap

from bioguider.agents.consistency_collection_step import ConsistencyCollectionStep
from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState

logger = logging.getLogger(__name__)


CLI_DOCUMENTATION = textwrap.dedent("""\
    ## Usage

    Train a model from the command line:

    ```bash
    python scripts/train.py --epochs 20 --lr 0.001 --gpu data/train.h5
    ```

    Then run the downstream analysis (implemented in R):

    ```bash
    Rscript bin/analyze.R --input results.csv --threshold 0.05 --verbose
    ```

    Programmatically, `scripts/train.py` simply constructs a `Trainer` and calls
    `Trainer.fit(dataset)`.
    """)


def test_collection_step_extracts_cli_invocations(llm, step_callback):
    state = ConsistencyEvaluationState(
        domain="user guide/API documentation",
        documentation=CLI_DOCUMENTATION,
        step_output_callback=step_callback,
    )
    step = ConsistencyCollectionStep(llm=llm)
    state = step.execute(state)

    cli = state["cli_invocations"]
    assert isinstance(cli, list) and len(cli) >= 2, cli
    for c in cli:
        logger.info("cli invocation: %s", c)

    programs = " ".join(str(c.get("program", "")) for c in cli).lower()
    assert "train.py" in programs, programs
    assert "analyze.r" in programs, programs

    langs = {str(c.get("language", "")).lower() for c in cli}
    assert "python" in langs, langs
    assert "r" in langs, langs

    # every collected invocation should at least carry an options list and a source
    for c in cli:
        assert isinstance(c.get("options", []), list), c
        assert c.get("source"), c

    # the function/class collection still works alongside it
    fc_names = {str(f.get("name", "")) for f in state["functions_and_classes"]}
    logger.info("functions/classes collected: %s", fc_names)
    assert {"Trainer", "fit"} & fc_names, fc_names
