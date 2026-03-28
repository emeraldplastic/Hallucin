from hallucination_detector import detect
from hallucination_detector.scorer import load_model

model = load_model()

CONTEXT = """
The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars
in Paris, France. It was designed and built by Gustave Eiffel's company between
1887 and 1889. The tower stands 330 metres (1,083 ft) tall and was the world's
tallest man-made structure for 41 years. Approximately 7 million people visit
it every year.
"""

print("=== Test 1: Grounded response ===")
result = detect(
    context=CONTEXT,
    response="The Eiffel Tower stands 330 metres tall and was built by Gustave Eiffel between 1887 and 1889. It is located in Paris, France.",
    model_name=model,
)
result.report()

print("=== Test 2: Hallucinated response ===")
result = detect(
    context=CONTEXT,
    response="The Eiffel Tower is 500 metres tall and was designed by Leonardo da Vinci in 1750. It is located in Rome and attracts 2 million visitors annually.",
    model_name=model,
)
result.report()

print("=== Test 3: Partial response ===")
result = detect(
    context=CONTEXT,
    response="The Eiffel Tower is in Paris and is very tall. It was built sometime in the 19th century and has millions of visitors.",
    model_name=model,
)
result.report()