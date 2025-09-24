from langsmith import Client
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated
from chatagent import graph, UserState, app
import os

load_dotenv()
LANGSMITH_KEY = os.getenv("LANGSMITH_KEY")

if not LANGSMITH_KEY:
    raise ValueError("LANGSMITH_KEY environment variable is not set")

client = Client(api_key=LANGSMITH_KEY)

# Create dataset name
dataset_name="DSAGPT Learning Assistant: Testing Pt.1"

# Create golden dataset
examples = [
    {
        "question": {"question": "What is a data structure?"},
        "response": {
            "answer": """A data structure is a way of organizing and storing data in a computer so that it can be accessed and modified efficiently. 
            It provides a means to manage large amounts of data efficiently for uses such as large databases and internet indexing services."""
        }
    },
    {
        "question": {"question": "What is an algorithm?"},
        "response": {
            "answer": """An algorithm is a step-by-step procedure or formula for solving a problem. 
            It is a finite sequence of well-defined instructions, typically used for calculations, data processing, and automated reasoning."""}
    },
    {
        "question": {"question": "I would like to see an example of binary search"},
        "response": {
            "answer": """Sure! I'd be happy to show you an example of binary search. To provide a relevant example, could you please share a few details?
            - Programming Language: e.g., Python, Java, or C++.
            - Array Size: A small example (e.g., 10 elements) or larger?
            - Sorted Array: Provide a sample sorted array if you have one in mind.
            Once I have this information, I can give you a code example and explanation. Thanks! """}
    },
    {
    "question": {"question": "What is a linked list?"},
    "response": {
      "answer": "A linked list is a linear data structure where elements are stored in nodes, and each node points to the next one in the sequence. Unlike arrays, linked lists do not have a fixed size and allow efficient insertion and deletion."
        }
    },
    {
    "question": {"question": "Why should I learn data structures and algorithms?"},
    "response": {
      "answer": "Learning DSA is essential for efficient problem-solving in programming. It helps in optimizing code, preparing for technical interviews, and understanding how software systems work under the hood."
        }
    },
    {
    "question": {"question": "How do I choose the right data structure for a problem?"},
    "response": {
      "answer": "Consider the operations you need (e.g., insertion, deletion, search) and their frequency. For fast lookups, use hash tables; for ordered data, use trees; for LIFO/FIFO, use stacks/queues. Analyze time and space complexity."
        }
    },
    {
    "question": {"question": "I would like to see an example of quicksort."},
    "response": {
      "answer": "Hi there! I'd be happy to show you an example of quicksort. To get started, could you please share a few details?\n\n- Programming Language: e.g., Python, Java.\n- Array Example: Provide a sample unsorted array.\n- Variant: Standard quicksort or with optimizations?\n\nOnce I have this info, I'll provide a code example and step-by-step explanation!"
        }
    },
    {
    "question": {"question": "What is the difference between stack and queue?"},
    "response": {
      "answer": "A stack follows Last-In-First-Out (LIFO) principle, like a pile of plates. A queue follows First-In-First-Out (FIFO), like a line at a counter. Stacks are used for recursion; queues for scheduling."
     }
    },
    {
    "question": {"question": "What is Big O notation?"},
    "response": {
      "answer": "Big O notation is a mathematical notation that describes the limiting behavior of a function when the argument tends towards infinity. In DSA, it's used to classify algorithms by their worst-case runtime or space complexity."
        }
    },
    {
    "question": {"question": "What is recursion?"}, 
    "response": {
      "answer": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. It requires a base case to terminate and is useful for problems like tree traversals or factorial calculations."
        }
    },
    {
    "question": {"question": "What is dynamic programming?"}, 
    "response": {
      "answer": "Dynamic programming is an optimization technique for solving complex problems by breaking them into simpler subproblems and storing results to avoid redundant calculations. It's used in problems like knapsack or Fibonacci sequence."
        }
    },
    {
    "question": {"question": "Explain binary tree traversal methods."}, 
    "response": {
      "answer": "Binary tree traversals include:\n- Inorder: Left, Root, Right\n- Preorder: Root, Left, Right\n- Postorder: Left, Right, Root\n- Level Order: Breadth-first, level by level.\nEach has uses like inorder for sorted output in BSTs."
        }
    },
    {
    "question": {"question": "What is the time complexity of binary search?"}, 
    "response": {
      "answer": "The time complexity of binary search is O(log n) in the worst and average cases, as it halves the search space with each step. It requires a sorted array."
        }
    }
]


if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in examples],
        outputs=[{"response": ex["response"]} for ex in examples],
        dataset_id=dataset.id
    )


# Target function
async def run_graph(inputs: dict) -> dict:
    """Run graph and track the final response."""
    # Create initial state with the question string and empty chat history
    state = UserState(
        message=inputs['question']['question'],
        chat_history=[],  # Initialize with empty chat history
        status="pending",
        config={
            "env": "test",
            "chat_history": []  # Add chat history to config
        }
    )
    
    try:
        # Run the graph
        result = await app.ainvoke(state)
        # Extract the response from the result
        if hasattr(result, 'response'):
            return {"response": result.response}
        elif isinstance(result, dict) and 'response' in result:
            return {"response": result['response']}
        else:
            return {"response": "No response generated"}
    except Exception as e:
        print(f"Error in run_graph: {str(e)}")
        return {"response": f"Error: {str(e)}"}


# LLM-as-judge instructions
grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the STUDENT RESPONSE.

Here is the grade criteria to follow:
(1) Grade the student responses based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student response does not contain any conflicting statements.
(3) It is OK if the student response contains more information than the ground truth response, as long as it is factually accurate relative to the  ground truth response.

Correctness:
True means that the student's response meets all of the criteria.
False means that the student's response does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""

# LLM-as-judge output schema
class Grade(TypedDict):
    """Compare the expected and actual answers and grade the actual answer."""
    reasoning: Annotated[str, ..., "Explain your reasoning for whether the actual response is correct or not."]
    is_correct: Annotated[bool, ..., "True if the student response is mostly or exactly correct, otherwise False."]

# Judge LLM
grader_llm = init_chat_model("gpt-4o-mini", temperature=0).with_structured_output(Grade, method="json_schema", strict=True)

# Evaluator function
async def final_answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Evaluate if the final response is equivalent to reference response."""
    # Note that we assume the outputs has a 'response' dictionary. We'll need to make sure
    # that the target function we define includes this key.
    user = f"""QUESTION: {inputs['question']}
    GROUND TRUTH RESPONSE: {reference_outputs['response']}
    STUDENT RESPONSE: {outputs['response']}"""

    grade = await grader_llm.ainvoke([{"role": "system", "content": grader_instructions}, {"role": "user", "content": user}])
    return grade["is_correct"]


# Evaluation job and results
async def main():
    experiment_results = await client.aevaluate(
        run_graph,
        data=dataset_name,
        evaluators=[final_answer_correct],
        experiment_prefix="dsagpt-agent-gpt4o-e2e",
        num_repetitions=1,
        max_concurrency=4,
    )
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())