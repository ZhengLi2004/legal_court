import os
import time
import argparse
GREEN = "\033[92m"  # 美化输出
RED = "\033[91m"
RESET = "\033[0m"
def log_pass(component: str, message: str = ""): print(f"{GREEN}✅ [PASS] {component:<20} {message}{RESET}")
def log_fail(component: str, message: str = ""): print(f"{RED}❌ [FAIL] {component:<20} {message}{RESET}")
# 测试类
class SystemTester:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("LEGAL_LLM_KEY")
        self.base_url = "http://47.102.193.166:8060/v1"
    # 依赖包安装情况检测
    def test_dependencies(self):
        print("\n>>> Phase 1: Checking Core Dependencies")

        required_packages = [
            "metagpt", "networkx", "chromadb", "sentence_transformers", 
            "numpy", "langchain", "openai"
        ]

        all_passed = True

        for pkg in required_packages:
            try:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "unknown")
                log_pass(pkg, f"(v{version})")

            except ImportError as e:
                log_fail(pkg, f"Not installed: {e}")
                all_passed = False

        return all_passed
    # 检查 NetworkX 图结构
    def test_graph(self):
        print("\n>>> Phase 2: Testing Graph Logic (NetworkX)")

        try:
            import networkx as nx
            G = nx.DiGraph()
            G.add_node("Fact_1", type="FACT", content="盗窃行为")
            G.add_node("Law_1", type="LAW", content="刑法264条")
            G.add_edge("Fact_1", "Law_1", type="SUPPORT")

            if G.number_of_nodes() == 2 and G.number_of_edges() == 1:
                log_pass("ShadowGraph Mock", "Nodes/Edges operations OK")
                return True
            
            else:
                log_fail("ShadowGraph Mock", "Graph state mismatch")
                return False
            
        except Exception as e:
            log_fail("ShadowGraph Mock", str(e))
            return False
    # 检查 ChromaDB 向量存储读写    
    def test_chroma(self):
        print("\n>>> Phase 3: Testing Vector Store (ChromaDB + Existing BGE-M3)")
        model_path = os.path.abspath("./bge-m3")
        
        if not os.path.exists(model_path):
            log_fail("Model Check", f"Path not found: {model_path}")
            print(f"    ⚠️  Please ensure the folder 'BGE-M3' exists in: {os.getcwd()}")
            return False
            
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_path
            )
            
            client = chromadb.Client()
            try: client.delete_collection("test_collection")
            except: pass

            collection = client.create_collection(
                name="test_collection",
                embedding_function=local_ef 
            )

            test_doc = "刑法第二百六十四条：盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑..."
            
            collection.add(
                documents=[test_doc],
                metadatas=[{"type": "law_article"}],
                ids=["law_264"]
            )
            
            query_text = "入室偷窃判几年？"
            
            results = collection.query(
                query_texts=[query_text],
                n_results=1
            )
            
            retrieved_id = results['ids'][0][0] if results['ids'][0] else None
            
            if retrieved_id == "law_264":
                log_pass("ChromaDB Integration", f"Retrieved '{retrieved_id}' for query '{query_text}'")
                return True
            
            else:
                log_fail("ChromaDB Integration", f"Semantic search failed. Got: {results}")
                return False

        except Exception as e:
            log_fail("ChromaDB Integration", f"Error: {e}")
            return False
    # 检查 LLM API 连接
    def test_llm(self):    
        print("\n>>> Phase 4: Testing Legal LLM API ('法衡')")
        
        if not self.api_key:
            print(f"{RED}⚠️  SKIPPING: API Key not found. Set LEGAL_LLM_KEY env var.{RESET}")
            return False
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="法衡",
                messages=[{"role": "user", "content": "请回复数字'1'。"}],
                max_tokens=10
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            log_pass("API Connection", f"Latency: {latency:.2f}s | Response: {content}")
            return True
    
        except Exception as e:
            log_fail("API Connection", f"Failed: {e}")
            return False
    # 检查 MetaGPT 基础环境    
    def test_metagpt(self):
        print("\n>>> Phase 5: Testing MetaGPT Environment")

        try:
            from metagpt.roles import Role
            from metagpt.schema import Message
            log_pass("MetaGPT Import", "Role & Message schemas loaded")
            return True
        
        except ImportError as e:
            log_fail("MetaGPT Import", str(e))
            return False
        
        except Exception as e:
            log_fail("MetaGPT Runtime", str(e))
            return False
        
def main():
    parser = argparse.ArgumentParser(description="G-Memory (Legal Adapter) System Check")
    # 定义模块映射
    modules = {
        'deps': 'test_dependencies',
        'graph': 'test_graph',
        'chroma': 'test_chroma',
        'llm': 'test_llm',
        'metagpt': 'test_metagpt'
    }

    parser.add_argument('module', nargs='?', default='all', choices=['all'] + list(modules.keys()),
                        help="Choose a specific module to test (default: all)")
    
    parser.add_argument('--key', type=str, help="API Key for LLM test")
    args = parser.parse_args()
    tester = SystemTester(api_key=args.key)
    # 默认全部测试
    if args.module == "all":
        results = [
            tester.test_dependencies(),
            tester.test_graph(),
            tester.test_chroma(),
            tester.test_metagpt(),
            tester.test_llm()
        ]

        if all(results): print(f"\n{GREEN}✨ All Systems Go! Ready for Phase 2.{RESET}")
        else: print(f"\n{RED}💀 System check failed. Please fix errors above.{RESET}")
    
    else:
        method_name = modules[args.module]
        getattr(tester, method_name)()

if __name__ == "__main__": main()