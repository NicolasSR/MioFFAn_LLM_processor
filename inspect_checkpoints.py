import argparse
import json
from pprint import pprint

from langgraph.checkpoint.sqlite import SqliteSaver

from src.lib.paths import LANGGRAPH_CHECKPOINTS_DIR
from src.lib.knowledge_base_api import parse_kb_from_string, print_math_entities


def list_threads(db_path: str):
    """Lists distinct thread IDs recorded in the database."""
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        cursor = checkpointer.conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = [row[0] for row in cursor.fetchall()]
        print("\nAvailable Thread IDs:")
        for t in threads:
            print(f" - {t}")
        return threads


def inspect_thread(db_path: str, thread_id: str, verbose: bool = False):
    """Iterates through checkpoint history in reverse-chronological order."""
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        config = {"configurable": {"thread_id": thread_id}}
        history = list(checkpointer.list(config))

        if not history:
            print(f"No checkpoints found for thread_id='{thread_id}'.")
            return

        print(f"\nFound {len(history)} checkpoint(s) for thread: '{thread_id}'")
        print("=" * 70)

        for idx, checkpoint_tuple in enumerate(history):
            cfg = checkpoint_tuple.config.get("configurable", {})
            checkpoint_id = cfg.get("checkpoint_id")
            metadata = checkpoint_tuple.metadata or {}
            checkpoint_data = checkpoint_tuple.checkpoint or {}

            step = metadata.get("step", "N/A")
            source = metadata.get("source", "unknown")
            writes = metadata.get("writes", {})
            executed_nodes = list(writes.keys()) if writes else [metadata.get("source")]

            # Values are inside checkpoint["channel_values"]
            channel_values = checkpoint_data.get("channel_values", {})

            # 2. To see what is scheduled next:
            # Look for internal branch channels in checkpoint channel_values:
            scheduled_nodes = [
                k.replace("branch:to:", "")
                for k in channel_values.keys()
                if k.startswith("branch:to:")
                ]

            # Pending tasks/nodes scheduled next (if available)
            tasks = getattr(checkpoint_tuple, "tasks", None)
            next_nodes = [t.name for t in tasks] if tasks else metadata.get("parents", {})

            print(f"\n[{idx + 1}] Checkpoint ID: {checkpoint_id}")
            print(f"    Step: {step} | Source: {source}")
            if tasks is not None:
                print(f"    Pending task(s): {next_nodes}")

            if writes:
                print(f"    Writes emitted at this step: {list(writes.keys())}")

            print("    Channel Values (State):")
            if verbose:
                pprint(channel_values, indent=6)
            else:
                for k, v in channel_values.items():
                    val_repr = str(v)
                    if len(val_repr) > 80:
                        val_repr = val_repr[:77] + "..."
                    print(f"      - {k}: {val_repr}")

            print(f"    Executed: {executed_nodes}")
            print(f"    Target Node Scheduled Next: {scheduled_nodes}")

            print("-" * 70)


def inspect_specific_state(db_path: str, thread_id: str, checkpoint_id: str, print_entities: bool, write_ttl: bool):
    """Fetches full state payload at a specific checkpoint point-in-time."""
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        checkpoint_tuple = checkpointer.get_tuple(config)
        if not checkpoint_tuple:
            print(f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'.")
            return

        if print_entities or write_ttl:
            kb_state = checkpoint_tuple.checkpoint.get("channel_values", {}).get("knowledge_base")
            if not kb_state:
                print(f"No knowledge base found in checkpoint '{checkpoint_id}'.")
                return
            if print_entities:
                kb = parse_kb_from_string(kb_state)
                print_math_entities(kb)
            elif write_ttl:
                ttl_filename = f"checkpoint_{checkpoint_id}_kb.ttl"
                with open(ttl_filename, "w", encoding="utf-8") as f:
                    f.write(kb_state)
                print(f"Knowledge base at checkpoint '{checkpoint_id}' written to '{ttl_filename}'.")
            
        else:
            print(f"\n--- State Snapshot for Checkpoint: {checkpoint_id} ---")
            print("\nMetadata:")
            pprint(checkpoint_tuple.metadata)
            print("\nValues:")
            pprint(checkpoint_tuple.checkpoint.get("channel_values", {}))
            print("\nPending Writes:")
            pprint(checkpoint_tuple.pending_writes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect LangGraph SqliteSaver checkpoints.")
    parser.add_argument("--db", default=LANGGRAPH_CHECKPOINTS_DIR / "pipeline_checkpoints.db", help="Path to SQLite database")
    parser.add_argument("--thread", help="Thread ID to inspect")
    parser.add_argument("--checkpoint-id", help="Specific checkpoint ID for deep inspection")
    parser.add_argument("--write-ttl", action="store_true", help="Write TTL file for deep inspection")
    parser.add_argument("--print-entities", action="store_true", help="Write information about the entities in the knowledge base")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print complete state dicts")

    args = parser.parse_args()

    if not args.thread:
        list_threads(args.db)
        print("\nTip: Run with --thread <THREAD_ID> to inspect state history.")
    elif args.checkpoint_id:
        inspect_specific_state(args.db, args.thread, args.checkpoint_id, args.print_entities, args.write_ttl)
    else:
        inspect_thread(args.db, args.thread, verbose=args.verbose)