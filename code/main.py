import argparse
import csv
import os
from pathlib import Path

from router import hardcoded_response, safety_check, detect_domain
from retriever import load_all_chunks, BM25Retriever
from llm import generate_response

# ---------------------------------------------------------------------------
# Domain-sensitive minimum BM25 score thresholds
# ---------------------------------------------------------------------------
# Instead of a single fixed threshold for all domains, each domain gets its
# own minimum score.  If the best BM25 hit falls below this value, we treat
# retrieval as "no relevant document found" and force escalation.
#
# Design rationale:
#   • visa (5.0)       — Even a weak corpus match is better than escalating a
#                        genuine card emergency (lost card, fraud, chargeback).
#                        False-positive replies here are cheaper than leaving a
#                        panicked cardholder without any guidance.
#   • hackerrank (15.0) — Balanced: the corpus is dense and well-structured,
#                        so a score below 15 genuinely means no relevant doc.
#   • claude (15.0)     — Same reasoning as HackerRank; the Anthropic docs are
#                        comprehensive and specific.
#   • unknown (20.0)    — Higher bar because we have no domain signal to filter
#                        by, so BM25 searches the entire corpus.  A score that
#                        would be meaningful in a focused domain search may just
#                        be noise in the global index.
# ---------------------------------------------------------------------------
DOMAIN_MIN_SCORE: dict[str, float] = {
    "hackerrank": 15.0,
    "claude":     15.0,
    "visa":        5.0,
    "unknown":    20.0,
}

def main():
    parser = argparse.ArgumentParser(description="Process support tickets.")
    parser.add_argument("--input", type=str, default="../support_tickets/support_tickets.csv", help="Path to input CSV file")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / args.input

    output_path = repo_root / "support_tickets" / "output.csv"
    
    print("Loading chunks and building retriever...")
    chunks = load_all_chunks()
    retriever = BM25Retriever(chunks)
    
    print(f"Reading from {input_path}")
    print(f"Writing to {output_path}")
    
    replied_count = 0
    escalated_count = 0

    with open(input_path, mode='r', encoding='utf-8') as f_in, \
         open(output_path, mode='w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = ['issue', 'subject', 'company', 'response', 'product_area', 'status', 'request_type', 'justification']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for idx, row in enumerate(reader, start=1):
            issue = row.get('Issue', '')
            subject = row.get('Subject', '')
            company = row.get('Company', '')
            
            ticket_id = f"Ticket {idx}"
            print(f"--- Processing {ticket_id} ---")
            
            # 1. hardcoded_response
            hc = hardcoded_response(issue, company)
            if hc:
                status = hc.get('status', '')
                out_dict = {
                    'issue': issue,
                    'subject': subject,
                    'company': company,
                    'response': hc.get('response', ''),
                    'product_area': hc.get('product_area', ''),
                    'status': status,
                    'request_type': hc.get('request_type', ''),
                    'justification': hc.get('justification', '')
                }
                writer.writerow(out_dict)
                print(f"{ticket_id}: Decision = hardcoded_response (status: {status})")
                
                if status == 'escalated':
                    escalated_count += 1
                elif status == 'replied':
                    replied_count += 1
                continue
                
            # 2. safety_check
            force_escalate, escalate_reason = safety_check(issue, company)
            if force_escalate:
                print(f"{ticket_id}: Safety check triggered ({escalate_reason})")
                
            # 3. detect_domain
            domain = detect_domain(company, issue)
            print(f"{ticket_id}: Domain detected as '{domain}'")
            
            # 4. Build query
            query = f"{subject} {issue}".strip()
            
            # 5. Retrieve with confidence score
            domain_filter = domain if domain != 'unknown' else None
            retrieved_chunks, top_score = retriever.query_with_confidence(query, domain=domain_filter, top_k=3)
            print(f"{ticket_id}: BM25 top score = {top_score:.2f}")
            
            # 5b. Domain-sensitive retrieval confidence gate
            min_score = DOMAIN_MIN_SCORE.get(domain, 20.0)
            if not force_escalate and top_score < min_score:
                force_escalate = True
                escalate_reason = (
                    f"BM25 top score ({top_score:.2f}) below domain threshold "
                    f"({min_score:.1f} for '{domain}'): no relevant document found."
                )
                print(f"{ticket_id}: Below domain threshold -> forcing escalation")

            # 6. Call generate_response (pass retrieval_score for justification context)
            llm_result = generate_response(
                ticket_issue=issue,
                ticket_subject=subject,
                domain=domain,
                retrieved_chunks=retrieved_chunks,
                force_escalate=force_escalate,
                escalate_reason=escalate_reason,
                retrieval_score=top_score
            )
            
            status = llm_result.get('status', '')
            # 7. Validate output has all 5 fields
            out_dict = {
                'issue': issue,
                'subject': subject,
                'company': company,
                'response': llm_result.get('response', ''),
                'product_area': llm_result.get('product_area', ''),
                'status': status,
                'request_type': llm_result.get('request_type', ''),
                'justification': llm_result.get('justification', '')
            }
            writer.writerow(out_dict)
            
            print(f"{ticket_id}: Decision = LLM (status: {status})")
            
            if status == 'escalated':
                escalated_count += 1
            elif status == 'replied':
                replied_count += 1

    print("\n================ SUMMARY ================")
    print(f"Total tickets processed : {replied_count + escalated_count}")
    print(f"Total Replied           : {replied_count}")
    print(f"Total Escalated         : {escalated_count}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
