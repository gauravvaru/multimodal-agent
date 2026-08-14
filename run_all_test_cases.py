import os
import requests

BASE_URL = "http://localhost:8000"

AUDIO_FILE = "/Users/gauravvaru/Desktop/harvard.wav"
PDF_MEETING_NOTES = "/Users/gauravvaru/Desktop/test2_meeting_notes.pdf"
CODE_SCREENSHOT = "/Users/gauravvaru/Desktop/test3_code_screenshot.png"
PDF_WITH_YOUTUBE_URL = "/Users/gauravvaru/Desktop/test4_youtube_request.pdf" 
PDF_FOR_COMPARISON = "/Users/gauravvaru/Documents/Tender_Analysis_Internship_Report.pdf"
AUDIO_FOR_COMPARISON = "/Users/gauravvaru/Desktop/harvard_real.mp3"

CASES = [
    {
        "name": "Test 1 - Audio Transcription + Summary",
        "query": "Transcribe this audio and summarize it.",
        "files": [AUDIO_FILE],
        "expect_status_in": {"success", "partial"},
    },
    {
        "name": "Test 2 - PDF + Natural Language Query (action items)",
        "query": "What are the action items in this document?",
        "files": [PDF_MEETING_NOTES],
        "expect_status_in": {"success", "partial"},
    },
    {
        "name": "Test 3 - Image with Code (explain)",
        "query": "Explain this code and flag any bugs.",
        "files": [CODE_SCREENSHOT],
        "expect_status_in": {"success", "partial"},
    },
    {
        "name": "Test 4 - Cross-Input: PDF containing YouTube URL",
        "query": "Hit the YT URL in this PDF and give me a summary of it.",
        "files": [PDF_WITH_YOUTUBE_URL],
        "expect_status_in": {"success", "partial"},
    },
    {
        "name": "Test 5 - Multi-File Unified Query (audio + PDF)",
        "query": "Do the audio and the document discuss the same topic?",
        "files": [PDF_FOR_COMPARISON, AUDIO_FOR_COMPARISON],
        "expect_status_in": {"success", "partial"},
    },
]


def run_case(case: dict) -> dict:
    files = []
    opened = []
    try:
        for path in case["files"]:
            f = open(path, "rb")
            opened.append(f)
            files.append(("files", (os.path.basename(path), f)))

        resp = requests.post(
            f"{BASE_URL}/api/v1/agent/run",
            data={"query": case["query"]},
            files=files,
            timeout=180,
        )
    finally:
        for f in opened:
            f.close()

    result = {"name": case["name"], "http_status": resp.status_code}

    if resp.status_code != 200:
        result["pass"] = False
        result["reason"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
        return result

    body = resp.json()
    status = body.get("status")
    final_answer = body.get("final_answer", "")
    errors = body.get("errors", [])

    ok = status in case["expect_status_in"] and bool(final_answer.strip())
    result["pass"] = ok
    result["status"] = status
    result["answer_preview"] = final_answer[:200]
    result["errors"] = errors
    return result


def main() -> None:
    print(f"Running {len(CASES)} test cases against {BASE_URL}\n")
    results = []
    for case in CASES:
        print(f"--- {case['name']} ---")
        try:
            r = run_case(case)
        except FileNotFoundError as e:
            r = {"name": case["name"], "pass": False, "reason": f"Missing file: {e}"}
        except requests.exceptions.RequestException as e:
            r = {"name": case["name"], "pass": False, "reason": f"Request failed: {e}"}

        results.append(r)
        status_label = "PASS" if r.get("pass") else "FAIL"
        print(f"[{status_label}] {r}\n")

    print("=" * 60)
    passed = sum(1 for r in results if r.get("pass"))
    print(f"SUMMARY: {passed}/{len(results)} test cases passed")
    for r in results:
        label = "PASS" if r.get("pass") else "FAIL"
        print(f"  [{label}] {r['name']}")


if __name__ == "__main__":
    main()