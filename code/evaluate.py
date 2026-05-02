import argparse
import csv
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of agent output.")
    parser.add_argument("--expected", type=str, default="../support_tickets/sample_support_tickets.csv", help="Expected output CSV")
    parser.add_argument("--actual", type=str, default="../support_tickets/output.csv", help="Actual output CSV")
    args = parser.parse_args()

    expected_path = Path(args.expected)
    if not expected_path.is_absolute():
        expected_path = Path.cwd() / args.expected

    actual_path = Path(args.actual)
    if not actual_path.is_absolute():
        actual_path = Path.cwd() / args.actual

    with open(expected_path, mode='r', encoding='utf-8') as f_exp, \
         open(actual_path, mode='r', encoding='utf-8') as f_act:
        
        reader_exp = csv.DictReader(f_exp)
        reader_act = csv.DictReader(f_act)

        exp_rows = list(reader_exp)
        act_rows = list(reader_act)

        if len(exp_rows) != len(act_rows):
            print(f"Warning: row counts differ! Expected {len(exp_rows)}, got {len(act_rows)}")

        total = min(len(exp_rows), len(act_rows))
        if total == 0:
            print("No rows to evaluate.")
            return

        status_correct = 0
        req_type_correct = 0
        exact_matches = 0

        print(f"Evaluating {total} tickets...\n")

        for i in range(total):
            exp = exp_rows[i]
            act = act_rows[i]

            exp_status = exp.get("Status", "").strip().lower()
            act_status = act.get("status", "").strip().lower()

            exp_rt = exp.get("Request Type", "").strip().lower()
            act_rt = act.get("request_type", "").strip().lower()

            status_match = (exp_status == act_status)
            rt_match = (exp_rt == act_rt)

            if status_match:
                status_correct += 1
            if rt_match:
                req_type_correct += 1
            if status_match and rt_match:
                exact_matches += 1
                
            if not status_match or not rt_match:
                print(f"Ticket {i+1} mismatch:")
                if not status_match:
                    print(f"  Status: Expected '{exp_status}', got '{act_status}'")
                if not rt_match:
                    print(f"  Request Type: Expected '{exp_rt}', got '{act_rt}'")

        print("\n--- Accuracy Report ---")
        print(f"Status Match:       {status_correct}/{total} ({(status_correct/total)*100:.1f}%)")
        print(f"Request Type Match: {req_type_correct}/{total} ({(req_type_correct/total)*100:.1f}%)")
        print(f"Both Exact Match:   {exact_matches}/{total} ({(exact_matches/total)*100:.1f}%)")

if __name__ == "__main__":
    main()
