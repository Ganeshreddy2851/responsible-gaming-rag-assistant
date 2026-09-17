import json

from rag_app.chains import build_rag_chain

with open(
    "tests/answer_evaluation.json",
    "r",
    encoding="utf-8"
) as file:
    test_cases = json.load(file)


rag_chain = build_rag_chain(k=3)

ABSTENTION_TEXT = (
    "The information was not found in the provided documents."
)

passed = 0


for test in test_cases:

    question = test["question"]
    expected_keywords = test["expected_keywords"]
    should_abstain = test["should_abstain"]

    result = rag_chain.invoke(question)

    answer = result["answer"]

    if should_abstain:

        test_passed = (
            ABSTENTION_TEXT.lower()
            in answer.lower()
        )

    else:

        test_passed = all(
            keyword.lower() in answer.lower()
            for keyword in expected_keywords
        )

    if test_passed:
        passed += 1

    print("\n" + "=" * 60)

    print("QUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nEXPECTED KEYWORDS:")
    print(expected_keywords)

    print("\nSHOULD ABSTAIN:")
    print(should_abstain)

    print("\nRESULT:")
    print("PASS" if test_passed else "FAIL")


total = len(test_cases)

percentage = (
    passed / total
) * 100


print("\n" + "=" * 60)
print("ANSWER EVALUATION RESULTS")
print("=" * 60)

print(
    f"Passed: {passed}/{total} "
    f"({percentage:.1f}%)"
)

