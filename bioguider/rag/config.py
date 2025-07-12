import os
from typing import List
from adalflow import GoogleGenAIClient
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.components.model_client.azureai_client import AzureAIClient


DEFAULT_EXCLUDED_DIRS: List[str] = [
    # Virtual environments and package managers
    "./.venv/", "./venv/", "./env/", "./virtualenv/",
    "./node_modules/", "./bower_components/", "./jspm_packages/",
    # Version control
    "./.git/", "./.svn/", "./.hg/", "./.bzr/",
    # Cache and compiled files
    "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
    # Build and distribution
    "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
    # Documentation
    "./docs/", "./_docs/", "./site-docs/", "./_site/",
    # IDE specific
    "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
    # Logs and temporary files
    "./logs/", "./log/", "./tmp/", "./temp/",
]

DEFAULT_EXCLUDED_FILES: List[str] = [
]

configs = {
    "embedder": {
        "batch_size": 500,
        "model_client": OpenAIClient,
        "model_kwargs": {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        },
    },
    "retriever": {
        "top_k": 20,
    },
    "generator": {
        "model_client": GoogleGenAIClient,
        "model_kwargs": {
            "model": "gemini-2.5-flash-preview-04-17",
            "temperature": 0.7,
            "top_p": 0.8,
        },
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 350,
        "chunk_overlap": 100,
    },
    "file_filters": {
        "excluded_dirs": [
            "./.venv/", "./venv/", "./env/", "./virtualenv/", 
            "./node_modules/", "./bower_components/", "./jspm_packages/",
            "./.git/", "./.svn/", "./.hg/", "./.bzr/",
            "./__pycache__/", "./.pytest_cache/", "./.mypy_cache/", "./.ruff_cache/", "./.coverage/",
            "./dist/", "./build/", "./out/", "./target/", "./bin/", "./obj/",
            "./_docs/", "./site-docs/", "./_site/",
            "./.idea/", "./.vscode/", "./.vs/", "./.eclipse/", "./.settings/",
            "./logs/", "./log/", "./tmp/", "./temp/",
        ],
        "excluded_files": [
            "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json",
            "poetry.lock", "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
            ".lock", ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk",
            ".env", ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv",
            ".gitignore", ".gitattributes", ".gitmodules", ".github", ".gitlab-ci.yml",
            ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc", ".editorconfig",
            ".jshintrc", ".pylintrc", ".flake8", "mypy.ini", "pyproject.toml",
            "tsconfig.json", "webpack.config.js", "babel.config.js", "rollup.config.js",
            "jest.config.js", "karma.conf.js", "vite.config.js", "next.config.js",
            "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css",
            "*.map", "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar",
            "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.class", "*.exe", "*.o", "*.a",
            "*.jpg", "*.jpeg", "*.png", "*.gif", "*.ico", "*.svg", "*.webp",
            "*.mp3", "*.mp4", "*.wav", "*.avi", "*.mov", "*.webm",
            "*.csv", "*.tsv", "*.xls", "*.xlsx", "*.db", "*.sqlite", "*.sqlite3",
            "*.pdf", "*.docx", "*.pptx",
        ],
    },
    "repository": {
        # Maximum repository size in MB
        "size_limit_mb": 50000,
    },
}

def get_embedder_config():
    return configs["embedder"]

def create_model_client():
    openai_type = os.environ.get("OPENAI_API_TYPE")
    is_azure = openai_type == "azure" if openai_type is not None else False
    if not is_azure:
        return OpenAIClient()
    return AzureAIClient(
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_version=os.environ.get("OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )
def create_model_kwargs():
    openai_type = os.environ.get("OPENAI_API_TYPE")
    is_azure = openai_type == "azure" if openai_type is not None else False
    if not is_azure:
        return {
            "model": "text-embedding-3-small",
            "dimensions": 256,
            "encoding_format": "float",
        }
    return {
        "model": os.environ.get("OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME"), 
        "dimensions": 256, 
        "encoding_format": "float", 
    }