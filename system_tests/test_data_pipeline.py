
import pytest
from bioguider.rag.data_pipeline import DatabaseManager, count_tokens

@pytest.mark.skip()
def test_database_manager_initialization():
    """Test the initialization of the DatabaseManager."""
    db_manager = DatabaseManager()
    assert db_manager is not None, "DatabaseManager should be initialized successfully."

    doc_documents, code_documents = db_manager.prepare_database("https://github.com/biocypher/biochatter.git", "github")
    assert doc_documents is not None, "Document documents should be prepared successfully."
    assert code_documents is not None, "Code documents should be prepared successfully."


content = """
I have several sentences from BioChatter documents, please help me to verify them respectively to check if they are related to BioChatter installation:
1. '# BioChatter Vignettes\n\nHere, we demonstrate BioChatter usage in various use cases to get you started\nwith your own implementation. Find the different use cases in the navigation\nside bar on the left.\n'
2. '# BioChatter API Reference Documentation\n\nHere we collect documentation of BioChatter module APIs. For detailed\ninformation on each module, please refer to the navigation side bar.\n'
3. '---\ntitle: BioChatter - Conversational AI for Biomedical Applications\ndescription: A framework for deploying, testing, and evaluating conversational AI models in the biomedical domain.\n---\n# Home\n\nGenerative AI models have shown tremendous usefulness in increasing\naccessibility and automation of a wide range of tasks. Yet, their application to\nthe biomedical domain is still limited, in part due to the lack of a common\nframework for deploying, testing, and evaluating the diverse models and\nauxiliary technologies that are needed. `biochatter` is a Python package\nimplementing a generic backend library for the connection of biomedical\napplications to conversational AI. We describe the framework in [this\npaper](https://www.nature.com/articles/s41587-024-02534-3); for a more hands-on\nexperience, check out our two web app implementations:\n\n<div class="grid cards" markdown>\n\n-   :material-clock-fast:{ .lg .middle } &nbsp; __BioChatter Light__\n\n    ---\n\n    Agile framework in pure Python built with [Streamlit](https://streamlit.io),\n    for fast prototyping and iteration.\n\n    [:octicons-arrow-right-24: Go To BioChatter Light](https://light.biochatter.org)\n\n-   :fontawesome-solid-wand-magic-sparkles:{ .lg .middle } &nbsp; __BioChatter Next__\n\n    ---\n\n    Advanced client-server architecture based on\n    [FastAPI](https://fastapi.tiangolo.com) and\n    [Next.js](https://nextjs.org).\n\n    [:octicons-arrow-right-24: Go To BioChatter Next](https://next.biochatter.org)\n\n</div>\n\nBioChatter is part of the [BioCypher](https://github.com/biocypher) ecosystem,\nconnecting natively to BioCypher knowledge graphs.\n\n![BioChatter Overview](images/biochatter_overview.png)\n\n!!! tip "Hot Topics"\n\n    BioChatter natively extends [BioCypher](https://biocypher.org) knowledge\n    graphs. Check there for more information.\n\n    We have also recently published a perspective on connecting knowledge and\n    machine learning to enable causal reasoning in biomedicine, with a\n    particular focus on the currently emerging "foundation models." You can read\n    it [here](https://www.embopress.org/doi/full/10.1038/s44320-024-00041-w).\n\n## Installation\n\nTo use the package, install it from PyPI, for instance using pip (`pip install\nbiochatter`) or Poetry (`poetry add biochatter`).\n\n### Extras\n\nThe package has some optional dependencies that can be installed using the\nfollowing extras (e.g. `pip install biochatter[xinference]`):\n\n- `xinference`: support for querying open-source LLMs through Xorbits Inference\n\n- `ollama`: support for querying open-source LLMs through Ollama\n\n- `podcast`: support for podcast text-to-speech (for the free Google TTS; the\npaid OpenAI TTS can be used without this extra)\n\n- `streamlit`: support for streamlit UI functions (used in BioChatter Light)\n\n[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/pypi/pyversions/biochatter)](https://www.python.org) [![PyPI version](https://img.shields.io/pypi/v/biochatter)](https://pypi.org/project/biochatter/) [![Downloads](https://static.pepy.tech/badge/biochatter)](https://pepy.tech/project/biochatter) '
"""
def test_count_tokens():
    tokens = count_tokens(content)
    print(tokens)
    assert tokens > 0, "Token count should be greater than zero."
