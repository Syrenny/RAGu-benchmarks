import argparse 

from src.common import RAGDataset, create_dir
from ragu.utils.parameters import get_parameters
from ragu.common.llm import RemoteLLM, LocalLLM
from ragu.graph.graph_rag import GraphRag
from ragu.common.settings import Settings


def main(args) -> None:
    json_dataset = RAGDataset(args.qa, args.docs)
    chunker_params, triplet_params, reranker_params, generator_params = (
        get_parameters(args.config)
    )
    
    client = LocalLLM(
        model_name=args.model
    )

    graph_rag = GraphRag(
        chunker_params,
        triplet_params,
        reranker_params,
        generator_params
    ).build(json_dataset.get_documents(), client)

    checkpoint_dir = create_dir(source=args.output_name)
    print("Checkpoint directory:", checkpoint_dir)

    # Сохранение результатов
    graph_rag.save_graph(checkpoint_dir / "graph.gml")
    graph_rag.save_community_summary(
        checkpoint_dir / "community_summary.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GraphRag with custom settings.")
    parser.add_argument(
        "--qa", type=str, default="./data/fairy-tales/qa.json", help="Path to the qa data file."
    )
    parser.add_argument(
        "--docs", type=str, default="./data/fairy-tales/documents.json", help="Path to the documents data file."
    )
    parser.add_argument(
        "--output_name", type=str, default="fairy-tales", help="Output name."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml", help="Path to the system config file."
    )
    parser.add_argument(
        "--model", type=str, help="Name of HuggingFace model.", required=True
    )
    
    
    
    args = parser.parse_args()
    main(args)
