# simple_rag_nvidia.py
# RAG using FREE NVIDIA API (No Anthropic needed)
# NVIDIA APIを使用した無料RAGシステム（Anthropic不要）

import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple, Optional

# ============================================
# Load NVIDIA API key from .env file
# .envファイルからNVIDIA APIキーを読み込み
# ============================================
load_dotenv()

# Get NVIDIA API key from environment variable
# 環境変数からNVIDIA APIキーを取得
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# NVIDIA API endpoint (free tier)
# NVIDIA APIエンドポイント（無料枠）
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Check if API key is set / APIキーの確認
if not NVIDIA_API_KEY:
    print("⚠️ WARNING: NVIDIA_API_KEY not found in .env file")
    print("⚠️ 警告：.envファイルにNVIDIA_API_KEYが見つかりません")
    print("\nPlease create a .env file with:")
    print("以下の内容で.envファイルを作成してください：")
    print("NVIDIA_API_KEY=your-nvidia-api-key-here")
    print("\nGet your free API key from: https://build.nvidia.com/")
    print("無料APIキーを取得：https://build.nvidia.com/")


class SimpleRAG:
    """
    A simple RAG system using FREE NVIDIA API
    無料NVIDIA APIを使用するシンプルなRAGシステム

    Workflow: Search -> Add Context -> Generate
    ワークフロー：検索 -> コンテキスト追加 -> 生成
    """

    def __init__(self, api_key: str = None):
        """
        Initialize RAG system with NVIDIA API
        NVIDIA APIでRAGシステムを初期化

        Args / パラメータ:
            api_key: NVIDIA API key / NVIDIA APIキー
        """
        # Set API key / APIキーを設定
        if api_key is None:
            api_key = NVIDIA_API_KEY

        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is required. "
                "Please set it in .env file or pass it as parameter."
                "NVIDIA_API_KEYが必要です。"
                ".envファイルに設定するか、パラメータとして渡してください。"
            )

        self.api_key = api_key
        self.base_url = NVIDIA_BASE_URL

        # ============================================
        # Knowledge base with 15+ documents about Claude/AI
        # Claude/AIに関する15以上のドキュメントを含む知識ベース
        # ============================================
        self.documents = [
            # Claude models / Claudeモデルシリーズ
            "Claude is an AI assistant developed by Anthropic in 2021",
            "Claudeは2021年にAnthropicが開発したAIアシスタントです",

            "Claude Opus 4.7 is the latest and most powerful model, released in May 2026",
            "Claude Opus 4.7は2026年5月にリリースされた最新かつ最強のモデルです",

            "Claude Sonnet 4.5 is the fastest model for everyday tasks",
            "Claude Sonnet 4.5は日常タスクに最適な最速モデルです",

            "Claude Haiku 4.5 is the most affordable and lightweight model",
            "Claude Haiku 4.5は最も手頃で軽量なモデルです",

            # Model capabilities / モデル機能
            "Claude models support a 200,000 token context window, can process long documents",
            "Claudeモデルは20万トークンのコンテキストウィンドウをサポートし、長文ドキュメントを処理できます",

            "Claude can analyze images, charts, and diagrams using vision capabilities",
            "Claudeは視覚認識機能を使用して画像、チャート、ダイアグラムを分析できます",

            "Claude supports file uploads including PDF, TXT, CSV, and image files",
            "ClaudeはPDF、TXT、CSV、画像ファイルなどのアップロードをサポートしています",

            "Claude features low hallucination rates and high factual accuracy",
            "Claudeは低い幻覚率と高い事実精度が特徴です",

            # Features / 機能
            "Claude supports multiple languages including English, Chinese, Japanese, and Spanish",
            "Claudeは英語、中国語、日本語、スペイン語などの複数言語をサポートしています",

            "Claude API offers competitive pricing with pay-per-token billing",
            "Claude APIは従量課金制で競争力のある価格を提供しています",

            "Claude can be accessed via web interface at claude.ai",
            "Claudeはclaude.aiのWebインターフェースからアクセスできます",

            # Safety and ethics / 安全性と倫理
            "Claude is designed with Constitutional AI to ensure helpful and harmless responses",
            "Claudeは有用で無害な応答を保証するConstitutional AIで設計されています",

            "Claude refuses unsafe requests and avoids generating harmful content",
            "Claudeは不安全なリクエストを拒否し、有害なコンテンツの生成を避けます",

            # Development / 開発関連
            "Claude API allows developers to build AI applications quickly",
            "Claude APIにより開発者はAIアプリケーションを迅速に構築できます",

            "Claude supports function calling and tool use for complex workflows",
            "Claudeは複雑なワークフローのための関数呼び出しとツール使用をサポートしています",

            "Claude offers enterprise-grade security and data privacy",
            "Claudeはエンタープライズレベルのセキュリティとデータプライバシーを提供します",

            # Additional facts / 追加情報
            "Claude 3.5 Sonnet excels at code generation and technical tasks",
            "Claude 3.5 Sonnetはコード生成と技術的タスクに優れています",

            "Claude demonstrates strong reasoning abilities in mathematics and logic",
            "Claudeは数学と論理における強力な推論能力を示します",

            "Claude API supports streaming responses for real-time applications",
            "Claude APIはリアルタイムアプリケーションのためのストリーミング応答をサポートしています",
        ]

        print(f"✅ Initialized RAG system with {len(self.documents)} documents")
        print(f"✅ {len(self.documents)}個のドキュメントでRAGシステムを初期化しました\n")

    def add_document(self, doc: str, doc_jp: str = None) -> None:
        """
        Add a single document to knowledge base
        単一ドキュメントを知識ベースに追加

        Args / パラメータ:
            doc: Document text in English / 英語のドキュメントテキスト
            doc_jp: Document text in Japanese (optional) / 日本語のドキュメントテキスト（オプション）
        """
        self.documents.append(doc)
        if doc_jp:
            self.documents.append(doc_jp)
        print(f"   + Added document: {doc[:50]}...")

    def simple_search(self, query: str) -> Tuple[Optional[str], List[Tuple[str, int]]]:
        """
        Simple keyword search (simulates vector search)
        シンプルなキーワード検索（ベクトル検索をシミュレーション）

        Args / パラメータ:
            query: User's question / ユーザーの質問
        Returns / 戻り値:
            (Best matching document, list of all matches with scores)
            (最適なドキュメント, スコア付きの全マッチングリスト)
        """
        # Convert query to lowercase words / クエリを小文字の単語に変換
        query_words = query.lower().split()

        # Calculate score for each document / 各ドキュメントのスコアを計算
        matches = []
        for doc in self.documents:
            doc_lower = doc.lower()
            # Count how many query words appear in document
            # クエリの単語がドキュメントに何回現れるかカウント
            score = sum(1 for word in query_words if word in doc_lower)
            if score > 0:
                matches.append((doc, score))

        # Sort by score and return the best one / スコア順にソートして最適を返す
        matches.sort(key=lambda x: x[1], reverse=True)

        best_match = matches[0][0] if matches else None
        return best_match, matches

    def generate_with_context(self, query: str, show_matches: bool = False) -> str:
        """
        Generate answer using retrieved context via NVIDIA API
        NVIDIA APIを使用して取得したコンテキストで回答を生成

        Steps / ステップ:
        1. Retrieve relevant document / 関連ドキュメントを検索
        2. Build prompt with context / コンテキストを含むプロンプトを構築
        3. Call NVIDIA API to generate / NVIDIA APIを呼び出して生成

        Args / パラメータ:
            query: User's question / ユーザーの質問
            show_matches: Whether to print matching documents / マッチングドキュメントを表示するか
        """

        # ========== STEP 1: RETRIEVAL / ステップ1：検索 ==========
        context, all_matches = self.simple_search(query)

        # Show matches if requested / 必要に応じてマッチを表示
        if show_matches and all_matches:
            print(f"\n📊 Found {len(all_matches)} matching documents:")
            print(f"📊 {len(all_matches)}個のマッチングドキュメントが見つかりました：")
            for i, (doc, score) in enumerate(all_matches[:3], 1):  # Show top 3
                print(f"   {i}. [Score/スコア: {score}] {doc[:80]}...")

        # ========== STEP 2: AUGMENTATION / ステップ2：拡張 ==========
        # Build system prompt / システムプロンプトを構築
        system_prompt = """You are a helpful AI assistant. 
                        Answer based on the provided information.
                        If the information doesn't contain the answer, say "I don't have that information in my knowledge base."
                        Be concise and accurate.

                        あなたは有用なAIアシスタントです。
                        提供された情報に基づいて回答してください。
                        情報に答えが含まれていない場合は、「知識ベースにその情報がありません」と言ってください。
                        簡潔かつ正確に回答してください。"""

        # Build user prompt with context / コンテキスト付きユーザープロンプトを構築
        if context:
            user_prompt = f"""Based on the following information, answer the question.

                            INFORMATION / 情報:
                            {context}

                            QUESTION / 質問:
                            {query}

                            ANSWER / 回答:"""
        else:
            user_prompt = f"QUESTION / 質問: {query}\n\nANSWER / 回答:"

        # ========== STEP 3: GENERATION / ステップ3：生成 ==========
        # Call NVIDIA API using OpenAI-compatible endpoint
        # OpenAI互換エンドポイントを使用してNVIDIA APIを呼び出し

        try:
            # Create OpenAI client for NVIDIA API
            # NVIDIA API用のOpenAIクライアントを作成
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.api_key,
                timeout=30
            )

            # Make the API call / API呼び出しを実行
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3,
                top_p=0.95
            )

            # Extract answer / 回答を抽出
            answer = response.choices[0].message.content

            # Add source citation if context was used
            # コンテキストが使用された場合、ソース引用を追加
            if context:
                answer += f"\n\n[Source: Retrieved from knowledge base]"
                answer += f"\n[ソース：知識ベースから取得]"

            return answer

        except Exception as e:
            return f"Error / エラー: {str(e)}"

    def search_and_answer(self, query: str) -> None:
        """
        Search and display answer with detailed output
        検索して詳細出力付きで回答を表示
        """
        print("\n" + "="*70)
        print(f"🔍 Query / 質問: {query}")
        print("="*70)

        answer = self.generate_with_context(query, show_matches=True)

        print("\n" + "="*70)
        print(f"💡 Answer / 回答:")
        print("="*70)
        print(answer)
        print("="*70)

    def show_knowledge_base(self) -> None:
        """Display all documents in knowledge base"""
        print("\n" + "="*70)
        print("📚 KNOWLEDGE BASE / 知識ベース")
        print("="*70)
        for i, doc in enumerate(self.documents, 1):
            print(f"{i:2d}. {doc}")
        print("="*70)
        print(f"Total / 合計: {len(self.documents)} documents / ドキュメント\n")


# ============================================
# Helper function to create .env file
# 補助関数：.envファイルを作成
# ============================================

def setup_env_file():
    """Create .env file template if it doesn't exist"""
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# NVIDIA API Key (Free from https://build.nvidia.com/)\n")
            f.write("# NVIDIA APIキー（https://build.nvidia.com/ から無料取得）\n")
            f.write("NVIDIA_API_KEY=your-nvidia-api-key-here\n")
            f.write("\n# How to get your free API key:\n")
            f.write("# 無料APIキーの取得方法：\n")
            f.write("# 1. Go to https://build.nvidia.com/\n")
            f.write("#    https://build.nvidia.com/ にアクセス\n")
            f.write("# 2. Sign up for free account\n")
            f.write("#    無料アカウントに登録\n")
            f.write("# 3. Go to API section and generate key\n")
            f.write("#    APIセクションでキーを生成\n")
            f.write("# 4. Copy the key and paste it above\n")
            f.write("#    キーをコピーして上記に貼り付け\n")
        print("📝 Created .env template file. Please add your NVIDIA API key.")
        print("📝 .envテンプレートファイルを作成しました。NVIDIA APIキーを追加してください。")
        return False
    return True


# ============================================
# Main execution / メイン実行
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("🤖 RAG SYSTEM WITH FREE NVIDIA API")
    print("🤖 無料NVIDIA APIを使用したRAGシステム")
    print("="*70)

    # Setup .env file if needed / 必要に応じて.envファイルを設定
    setup_env_file()

    # Check if API key is available / APIキーが利用可能か確認
    if not NVIDIA_API_KEY:
        print("\n❌ NVIDIA_API_KEY not found!")
        print("❌ NVIDIA_API_KEYが見つかりません！")
        print("\nPlease:")
        print("手順：")
        print("1. Edit the .env file and add your NVIDIA API key")
        print("   .envファイルを編集してNVIDIA APIキーを追加")
        print("2. Get free key from: https://build.nvidia.com/")
        print("   無料キーを取得：https://build.nvidia.com/")
        print("3. Then run this script again")
        print("   その後、このスクリプトを再実行")
        exit(1)

    # Initialize RAG system / RAGシステムを初期化
    try:
        rag = SimpleRAG()
    except ValueError as e:
        print(f"\n❌ {e}")
        exit(1)

    # Show knowledge base / 知識ベースを表示
    rag.show_knowledge_base()

    # ============================================
    # Test queries / テスト質問
    # ============================================

    test_queries = [
        "What is the latest Claude model?",
        "What are the different Claude models?",
        "Can Claude analyze images?",
        "What safety features does Claude have?",
        "How many languages does Claude support?",
        "What is the context window size of Claude?",
    ]

    print("\n" + "="*70)
    print("🧪 TESTING RAG SYSTEM WITH REAL QUERIES")
    print("🧪 実際の質問でRAGシステムをテスト")
    print("="*70)

    # Run each query / 各質問を実行
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_queries)}")
        rag.search_and_answer(query)

        # Pause between queries to avoid rate limits
        # レート制限を避けるためクエリ間に一時停止
        if i < len(test_queries):
            input(
                "\nPress Enter to continue to next question...\n次の質問に進むにはEnterキーを押してください...")

    # ============================================
    # Interactive mode / インタラクティブモード
    # ============================================

    print("\n" + "="*70)
    print("💬 INTERACTIVE MODE")
    print("💬 インタラクティブモード")
    print("="*70)
    print("Type 'quit' to exit / 終了するには 'quit' と入力")

    while True:
        user_query = input("\n🔍 Your question / 質問: ").strip()

        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye! / さようなら！")
            break

        if user_query:
            rag.search_and_answer(user_query)
        else:
            print("Please enter a question / 質問を入力してください")
