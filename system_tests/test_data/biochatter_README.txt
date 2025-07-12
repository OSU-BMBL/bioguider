BioChatter
License	License: MIT	Python	Python
Package	PyPI version Downloads DOI	Build status	CI Docs
Tests	Coverage	Docker	Latest image Image size
Development	Project Status: Active – The project has reached a stable, usable state and is being actively developed. Code style Ruff	Contributions	PRs Welcome Contributor Covenant
Description
🤖 BioChatter is a community-driven Python library that connects biomedical applications to conversational AI, making it easy to leverage generative AI models in the biomedical domain.

🌟 Key Features
Generic backend for biomedical AI applications
Seamless integration with multiple LLM providers
Native connection to BioCypher knowledge graphs
Extensive testing and evaluation framework
Living benchmark of specific biomedical applications
🚀 Demo Applications and Utilities
BioChatter Light - Simple Python frontend (repo)

BioChatter Next - Advanced Next.js frontend (repo)

BioChatter Server - RESTful API server

📖 Learn more in our paper.

Installation
To use the package, install it from PyPI, for instance using pip (pip install biochatter) or Poetry (poetry add biochatter).

Extras
The package has some optional dependencies that can be installed using the following extras (e.g. pip install biochatter[xinference]):

xinference: support for querying open-source LLMs through Xorbits Inference

podcast: support for podcast text-to-speech (for the free Google TTS; the paid OpenAI TTS can be used without this extra)

streamlit: support for streamlit UI functions (used in BioChatter Light)

Usage
Check out the documentation for examples, use cases, and more information. Many common functionalities covered by BioChatter can be seen in use in the BioChatter Light code base. Built with Material for MkDocs

🤝 Getting involved
We are very happy about contributions from the community, large and small! If you would like to contribute to BioCypher development, please refer to our contribution guidelines and the developer docs. :)

If you want to ask informal questions, talk about dev things, or just chat, please join our community at https://biocypher.zulipchat.com!

Imposter syndrome disclaimer: We want your help. No, really. There may be a little voice inside your head that is telling you that you're not ready, that you aren't skilled enough to contribute. We assure you that the little voice in your head is wrong. Most importantly, there are many valuable ways to contribute besides writing code.

This disclaimer was adapted from the Pooch project.

More information about LLMs
Check out this repository for more info on computational biology usage of large language models.

Citation
If you use BioChatter in your work, please cite our paper.