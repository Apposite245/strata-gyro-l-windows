"""Answer check for a running Strata server: 16 problems with exact answers (10 AIME 2024/2025, 6 multi-step),
thinking on, temperature 0.6, one run each. A smoke test, not a benchmark: the AIME problems may be in the training data.

Usage:  python tests/answer_check.py [http://127.0.0.1:8098]
"""
import json, re, sys, time, urllib.request
from fractions import Fraction

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8098").rstrip("/")
PROBLEMS = [
    ("aime24-aya", "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop.", "204"),
    ("aime24-log", "There exist real numbers x and y, both greater than 1, such that log_x(y^x) = log_y(x^(4y)) = 10. Find xy.", "25"),
    ("aime24-tokens", "Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.", "809"),
    ("aime24-realpart", "Find the largest possible real part of (75+117i)z + (96+144i)/z, where z is a complex number with |z| = 4.", "540"),
    ("aime24-root13", "Let w be a 13th root of unity with w != 1. Find the remainder when the product over k = 0 to 12 of (2 - 2w^k + w^(2k)) is divided by 1000.", "321"),
    ("aime24-beautiful", "Let b >= 2 be an integer. Call a positive integer n b-eautiful if it has exactly two digits when expressed in base b and these two digits sum to sqrt(n). For example, 81 is 13-eautiful because 81 = 63 in base 13 and 6 + 3 = sqrt(81). Find the least integer b >= 2 for which there are more than ten b-eautiful integers.", "211"),
    ("aime24-lottery", "Jen enters a lottery by picking 4 distinct numbers from S = {1, 2, ..., 10}. 4 numbers are randomly chosen from S. She wins a prize if at least two of her numbers were 2 of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is m/n where m and n are relatively prime positive integers. Find m+n.", "116"),
    ("aime24-aimeville", "Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.", "73"),
    ("aime25-bases", "Find the sum of all integer bases b > 9 for which 17_b is a divisor of 97_b.", "70"),
    ("aime25-pairs", "Find the number of ordered pairs (x, y), where both x and y are integers between -100 and 100 inclusive, such that 12x^2 - xy - 6y^2 = 0.", "117"),
    ("monty4", "A game show has 4 doors: a car behind one, nothing behind the others. You pick a door. The host, who knows where the car is, opens one of the other doors that has nothing behind it. You then switch to one of the two remaining unopened doors (not your original), chosen uniformly at random. What is the probability that you win the car? Give an exact fraction.", "3/8"),
    ("bounce", "A ball is dropped from a height of 20 m onto a hard floor. Each bounce, its speed just after impact is half its speed just before impact. Ignoring air resistance, what total distance in meters does the ball travel before coming to rest? Give an exact value.", "100/3"),
    ("tartaric", "How many stereoisomers does tartaric acid (2,3-dihydroxybutanedioic acid) have in total, counting the meso form?", "3"),
    ("mod13", "What is the remainder when 2^100 is divided by 13?", "3"),
    ("digitsum", "How many four-digit positive integers have digits that sum to 9?", "165"),
    ("mississippi", "In how many distinct ways can the letters of MISSISSIPPI be arranged so that no two S's are adjacent?", "7350"),
]

SUFFIX = "\n\nEnd your reply with a last line of the form 'Answer: <value>' (an integer, or an exact fraction a/b)."


def parse(text):
    final = re.split(r"</think>", text)[-1]
    m = re.findall(r"Answer:\s*\**\s*\$?\\?(?:boxed\{)?\s*(-?\d+(?:\s*/\s*\d+)?|\\frac\{\d+\}\{\d+\}|-?\d+\.\d+)", final)
    if not m:
        return None
    v = m[-1].replace(" ", "")
    f = re.match(r"\\frac\{(\d+)\}\{(\d+)\}", v)
    try:
        return Fraction(int(f[1]), int(f[2])) if f else Fraction(v)
    except (ValueError, ZeroDivisionError):
        return None


score, tokens, seconds = 0, 0, 0.0
for pid, q, ans in PROBLEMS:
    body = {"model": "x", "messages": [{"role": "user", "content": q + SUFFIX}], "max_tokens": 24000,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.05}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    s = time.time() - t
    n = out["usage"]["completion_tokens"]
    got = parse(out["choices"][0]["message"].get("content") or "")
    ok = got is not None and got == Fraction(ans)
    score, tokens, seconds = score + ok, tokens + n, seconds + s
    print(f"{pid:18} {'PASS' if ok else 'FAIL'}  got {str(got):>8}  want {ans:>6}  {n:6} tok {s:7.1f} s", flush=True)
print(f"\n{score}/{len(PROBLEMS)} | {tokens} tokens in {seconds:.0f} s ({tokens / max(seconds, 1):.1f} tok/s)")