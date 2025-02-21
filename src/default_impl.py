import pickle, os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma as ChromaVectorStore
from langchain_core.embeddings import Embeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.storage import SQLStore

from langchain_graphrag.indexing.graph_generation import (
    GraphGenerator,
    EntityRelationshipDescriptionSummarizer,
    EntityRelationshipExtractor,
    GraphsMerger
)
from langchain_graphrag.indexing import TextUnitExtractor, SimpleIndexer
from langchain_graphrag.indexing.artifacts import IndexerArtifacts
from langchain_graphrag.indexing.graph_clustering.leiden_community_detector import (
    HierarchicalLeidenCommunityDetector,
)
from langchain_graphrag.indexing.artifacts_generation import (
    CommunitiesReportsArtifactsGenerator,
    EntitiesArtifactsGenerator,
    RelationshipsArtifactsGenerator,
    TextUnitsArtifactsGenerator,
)
from langchain_graphrag.indexing.report_generation import (
    CommunityReportGenerator,
    CommunityReportWriter,
)

from src.common import JSONDataset, create_dir


def save_artifacts(artifacts: IndexerArtifacts, path: Path):
    artifacts.entities.to_parquet(f"{path}/entities.parquet")
    artifacts.relationships.to_parquet(f"{path}/relationships.parquet")
    artifacts.text_units.to_parquet(f"{path}/text_units.parquet")
    artifacts.communities_reports.to_parquet(
        f"{path}/communities_reports.parquet")

    if artifacts.merged_graph is not None:
        with path.joinpath("merged-graph.pickle").open("wb") as fp:
            pickle.dump(artifacts.merged_graph, fp)

    if artifacts.summarized_graph is not None:
        with path.joinpath("summarized-graph.pickle").open("wb") as fp:
            pickle.dump(artifacts.summarized_graph, fp)

    if artifacts.communities is not None:
        with path.joinpath("community_info.pickle").open("wb") as fp:
            pickle.dump(artifacts.communities, fp)


def make_embedding_instance(
    model: str,
    cache_dir: Path
) -> Embeddings:
    underlying_embedding = HuggingFaceEmbeddings(
        model_name=model)

    embedding_db_path = "sqlite:///" + str(cache_dir.joinpath("embedding.db"))
    print("Embedding DB path:", embedding_db_path)
    store = SQLStore(namespace=model, db_url=embedding_db_path)
    store.create_schema()

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embedding,
        document_embedding_cache=store,
    )
    

# === Parameters ===
hf_token = "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"
dataset_path = Path("./data/fairy-tails/extended.json")
embeddings_cache_dir = Path("./embeddings/cache")
embeddings_vector_store_dir = Path("./embeddings/vector-store")

# === Text units ===
json_dataset = JSONDataset(dataset_path)
documents = []
for sample in json_dataset.get_documents():
    documents.append(Document(sample))
splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
text_unit_extractor = TextUnitExtractor(text_splitter=splitter)

# === LLM instance ===

# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# llm = ChatHuggingFace(llm=llm)

llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0.0,
    base_url="https://api.together.xyz/v1",
    api_key="",
    cache=SQLiteCache("openai_cache.db"),  # always a good idea to use Cache
)


# === Creation of various objects/dependencies ===

extractor = EntityRelationshipExtractor.build_default(llm=llm)

summarizer = EntityRelationshipDescriptionSummarizer.build_default(llm=llm)

graph_generator = GraphGenerator(
    er_extractor=extractor,
    graphs_merger=GraphsMerger(),
    er_description_summarizer=summarizer,
)

embedding_model = "deepvk/USER-bge-m3"
entities_collection_name = f"entity-deepvk-USER-bge-m3"
entities_vector_store = ChromaVectorStore(
    collection_name=entities_collection_name,
    persist_directory=embeddings_vector_store_dir.as_posix(),
    embedding_function=make_embedding_instance(
        model=embedding_model,
        cache_dir=embeddings_cache_dir,
    ),
    
)

entities_artifacts_generator = EntitiesArtifactsGenerator(
    entities_vector_store=entities_vector_store
)

report_generator = CommunityReportGenerator.build_default(
    llm=llm,
    chain_config={"tags": ["community-report"]},
)

report_writer = CommunityReportWriter()

communities_report_artifacts_generator = CommunitiesReportsArtifactsGenerator(
    report_generator=report_generator,
    report_writer=report_writer,
)


# === Indexing ===

indexer = SimpleIndexer(
    text_unit_extractor=text_unit_extractor,
    graph_generator=graph_generator,
    community_detector=HierarchicalLeidenCommunityDetector(),
    entities_artifacts_generator=entities_artifacts_generator,
    relationships_artifacts_generator=RelationshipsArtifactsGenerator(),
    text_units_artifacts_generator=TextUnitsArtifactsGenerator(),
    communities_report_artifacts_generator=communities_report_artifacts_generator,
)

artifacts = indexer.run(documents)


# === Saving artifacts ===
checkpoint_dir = create_dir(dataset_path)
save_artifacts(artifacts, checkpoint_dir)
artifacts.report()
