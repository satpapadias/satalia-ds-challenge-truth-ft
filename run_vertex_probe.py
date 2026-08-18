import json
from truthclf.llm import VertexClient

def run_probe():
    print("Initializing VertexClient for gemini-2.5-flash...")
    client = VertexClient("gemini-2.5-flash")
    
    messages = [
        {"role": "system", "content": "You are a binary classifier. Respond with strictly one word: True or False."},
        {"role": "user", "content": "Statement: The sky is blue.\nAnswer:"}
    ]
    
    print("Submitting prompt for classification...")
    try:
        # Clear the cache programmatically just in case
        client.cache.flush() 
        results = client.classify([messages])
        print("\nSuccess! Parsed Probabilities:")
        print(json.dumps(results[0], indent=2))
    except Exception as e:
        print("\nFailed with error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_probe()
