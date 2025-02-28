import openai
from retriever import retrieve_relevant_text

openai.api_key = "your-api-key"

def analyze_privacy_policy(policy_text):
    """Compare a privacy policy with legal requirements using LLM."""
    
    # Retrieve relevant legal text from ChromaDB
    relevant_legal_texts = retrieve_relevant_text(policy_text)

    # Construct LLM prompt
    prompt = f"""
    The following is a privacy policy text:
    
    {policy_text}

    Below are relevant legal texts regarding data compliance:
    
    {''.join(relevant_legal_texts)}

    Compare the privacy policy against the legal texts and:
    - Identify any **compliance gaps**.
    - Suggest **improvements**.
    - Provide a **compliance score (0-100)**.

    Return the response in a structured format.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[{"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# Example privacy policy analysis
policy_text = "We collect user data but do not specify how it is stored or protected."
analysis_result = analyze_privacy_policy(policy_text)

print("\n🔹 LLM Compliance Analysis:")
print(analysis_result)
