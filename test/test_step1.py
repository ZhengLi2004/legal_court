from mas.llm import GPTChat, Message
from mas.utils import EmbeddingFunc, cosine_similarity

def test_llm_call():
    print("\n>>> Testing LLM Wrapper...")
    llm = GPTChat()
    msgs = [Message(role="user", content="请输出'Hello Law'.")]
    res = llm(msgs)
    print(f"LLM Result: {res}")
    assert "Hello" in res or "Law" in res or len(res) > 0

def test_embedding_utils():
    print("\n>>> Testing Embedding Utils...")
    ef = EmbeddingFunc(model_path="./bge-m3")
    vec1 = ef.embed_query("盗窃罪")
    vec2 = ef.embed_query("偷窃")
    sim = cosine_similarity(vec1, vec2)
    print(f"Similarity ('盗窃罪' vs '偷窃'): {sim:.4f}")
    assert sim > 0.5

if __name__ == "__main__":
    try:
        test_llm_call()
        test_embedding_utils()
        print("\n✅ Step 2 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 2 Failed: {e}")