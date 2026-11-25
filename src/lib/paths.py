from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = PROJECT_ROOT_DIR / "input"
ARTICLES_DIR = INPUT_DIR / "articles"
ONTOLOGIES_DIR = INPUT_DIR / "ontologies"
EXTRACTS_DIR = INPUT_DIR / "extracts"
CONFIG_DIR = PROJECT_ROOT_DIR / "config"
DEFAULT_RUN_CONFIG_PATH = CONFIG_DIR / "default_run_config.json"
LOCAL_LLM_CONFIG_PATH = CONFIG_DIR / "local_llm_config.json"
OUTPUT_DIR = PROJECT_ROOT_DIR / "output"
LANGGRAPH_CHECKPOINTS_DIR = OUTPUT_DIR / "langgraph_checkpoints"