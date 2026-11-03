import json
from pathlib import Path

RESULTS = Path(
    "/scratch3/sha511/cssv3_results/tournament_2agent_full_170rounds_timed/rounds.jsonl"
)

TARGET_ROUNDS = [86, 94, 100, 133, 136, 158, 160, 161]
AGENT_ID = "LJ"

def get_agent(record, agent_id):
    return record.get("agents", {}).get(agent_id, {})

def get_round_number(record, fallback):
    return record.get("round", record.get("round_number", fallback))

def get_strategy(agent):
    return agent.get("strategy_doc")

def main():
    records = {}

    with RESULTS.open() as f:
        for index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            round_no = get_round_number(record, index)
            records[int(round_no)] = record

    print("=" * 100)
    print("PRE-ROUND AND POST-ROUND STRATEGY DOCUMENTS")
    print("=" * 100)

    print("\nROUND-WISE RANK SUMMARY")
    print("-" * 100)
    print(f"{'Round':<10}{'Round rank':<15}{'Cumulative rank':<20}{'Ranked first?'}")

    for round_no in TARGET_ROUNDS:
        record = records.get(round_no)
        if record is None:
            print(f"{round_no:<10}ROUND NOT FOUND")
            continue

        agent = get_agent(record, AGENT_ID)
        rank = agent.get("round_rank", agent.get("current_round_rank"))
        cumulative_rank = agent.get("cumulative_rank")

        first = "YES" if rank == 1 else "NO"
        print(f"{round_no:<10}{str(rank):<15}{str(cumulative_rank):<20}{first}")

    for round_no in TARGET_ROUNDS:
        print("\n" + "=" * 100)
        print(f"COMPLIANCE ROUND {round_no}")
        print("=" * 100)

        current_record = records.get(round_no)
        previous_record = records.get(round_no - 1)

        if current_record is None:
            print("Current round record not found.")
            continue

        current_agent = get_agent(current_record, AGENT_ID)
        previous_agent = (
            get_agent(previous_record, AGENT_ID)
            if previous_record else {}
        )

        print(f"\nPRE-ROUND STRATEGY (saved after round {round_no - 1})")
        print("-" * 100)

        pre_strategy = get_strategy(previous_agent)
        if pre_strategy is not None:
            print(json.dumps(pre_strategy, indent=2, ensure_ascii=False))
        else:
            print("No previous-round strategy document found.")

        print(f"\nPOST-ROUND STRATEGY (saved after round {round_no})")
        print("-" * 100)

        post_strategy = get_strategy(current_agent)
        if post_strategy is not None:
            print(json.dumps(post_strategy, indent=2, ensure_ascii=False))
        else:
            print("No post-round strategy document found.")

        print("\nRANK AND SCORE")
        print("-" * 100)
        print("Round rank:", current_agent.get(
            "round_rank", current_agent.get("current_round_rank")
        ))
        print("Cumulative rank:", current_agent.get("cumulative_rank"))
        print("Shown score:", current_agent.get("shown_score"))
        print("StrongREJECT score:", current_agent.get("sr_score"))

    print("\n" + "=" * 100)
    print("END OF STRATEGY COMPARISON")
    print("=" * 100)

if __name__ == "__main__":
    main()
