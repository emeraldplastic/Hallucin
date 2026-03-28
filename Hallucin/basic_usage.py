"""
examples/basic_usage.py
-----------------------
A runnable demo of the hallucination detector.
Run: python examples/basic_usage.py
"""

from hallucination_detector import detect

CONTEXT = """
The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars
in Paris, France. It was designed and built by Gustave Eiffel's company between
1887 and 1889. The tower stands 330 metres (1,083 ft) tall and was the world's
tallest man-made structure for 41 years. Approximately 7 million people visit
it every year.
"""

# --- Test 1: Mostly grounded response
print("=== Test 1: Grounded response ===")
response_good = (
    "The Eiffel Tower stands 330 metres tall and was built by Gustave Eiffel "
    "between 1887 and 1889. It is located in Paris, France."
)
result = detect(context=CONTEXT, response=response_good)
result.report()

# --- Test 2: Response with hallucinations
print("=== Test 2: Hallucinated response ===")
response_bad = (
    "The Eiffel Tower is 500 metres tall and was designed by Leonardo da Vinci "
    "in 1750. It is located in Rome and attracts 2 million visitors annually."
)
result = detect(context=CONTEXT, response=response_bad)
result.report()

# --- Test 3: Partial response
print("=== Test 3: Partial response ===")
response_mixed = (
    "The Eiffel Tower is in Paris and is very tall. "
    "It was built sometime in the 19th century and has millions of visitors."
)
result = detect(context=CONTEXT, response=response_mixed)
result.report()
