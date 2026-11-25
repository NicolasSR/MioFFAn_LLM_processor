from pathlib import Path
from argparse import ArgumentParser
from collections.abc import Callable
import json

from src.lib.paths import ARTICLES_DIR, ONTOLOGIES_DIR, EXTRACTS_DIR, LOCAL_LLM_CONFIG_PATH

def _list_files(file_repo_dir: Path, file_extension: str) -> dict[str, str]:
    """List files in a given directory with a specific extension."""
    if not file_repo_dir.exists():
        return {}
    return {d.stem: str(d) for d in sorted(file_repo_dir.glob(f"*{file_extension}"))}

def list_articles():
    """List HTML files in the articles directory."""
    return _list_files(ARTICLES_DIR, ".html")

def list_ontologies():
    """List TTL files in the ontologies directory."""
    return _list_files(ONTOLOGIES_DIR, ".ttl")

def list_extracts():
    """List extract files in the extracts directory."""
    return _list_files(EXTRACTS_DIR, ".json")

def list_llm_connection_profiles():
    """List LLM connection profiles from the local LLM configuration file."""
    if not LOCAL_LLM_CONFIG_PATH.exists():
        return {}
    with open(LOCAL_LLM_CONFIG_PATH, "r") as f:
        config = json.load(f)
    return {profile: profile for profile in config.keys()}

def _resolve_filename(dir_path: Path, filename_listing_method: Callable, target_filename: str | None, parser: ArgumentParser) -> str:
    inverted_files_dict = filename_listing_method()
    if not inverted_files_dict:
        parser.error(f"No appropriate files within {dir_path}.")
    while target_filename is None or target_filename not in inverted_files_dict:
        if target_filename is not None:
            print(f"'{target_filename}' is not a valid filename in '{dir_path}/'.")
        print("Available filenames:")
        for file_name in inverted_files_dict.keys():
            print(f"  {file_name}")
        print()
        try:
            target_filename = input("Enter sample filename: ").strip()
        except EOFError:
            parser.error("No filename was provided (input closed).")
    return inverted_files_dict[target_filename]

def resolve_article_filename(article_filename: str, parser: ArgumentParser) -> Path:
    return Path(_resolve_filename(ARTICLES_DIR, list_articles, article_filename, parser))

def resolve_ontology_filename(ontology_filename: str, parser: ArgumentParser) -> Path:
    return Path(_resolve_filename(ONTOLOGIES_DIR, list_ontologies, ontology_filename, parser))

def resolve_extract_filename(extract_filename: str, parser: ArgumentParser) -> Path:
    return Path(_resolve_filename(EXTRACTS_DIR, list_extracts, extract_filename, parser))

def resolve_llm_connection_profile(llm_connection_profile: str, parser: ArgumentParser) -> str:
    return _resolve_filename(LOCAL_LLM_CONFIG_PATH, list_llm_connection_profiles, llm_connection_profile, parser)