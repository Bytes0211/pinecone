

```mermaid
flowchart TD

    A[records.txt<br/>(_id, chunk_text, metadata)] --> B[load_records()<br/>Parse via AST<br/>Extract records list]

    B --> C[Extract Fields<br/>texts[], ids[], metadata[]]

    C --> D[embed_texts()<br/>Pinecone Inference API<br/>MODEL: llama-text-embed-v2]

    D --> E[Determine Dimension<br/>dimension = len(embedding)]

    E --> F[ensure_index()<br/>Describe index<br/>If dim mismatch → delete<br/>Else create serverless index]

    F --> G[Prepare Upsert Payload<br/>[{id, values, metadata}]]

    G --> H[index.upsert()<br/>Write vectors to Pinecone]

    H --> I[index.fetch()<br/>Retrieve vector by ID<br/>Validate ingestion]

    I --> J[Done]
    ```
