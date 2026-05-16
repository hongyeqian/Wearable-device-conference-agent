# Summary Level Summary

## M010 — Metadata & Retrieval Pipeline Discussions

- **[S010-T01] Topic: Reference IDs and metadata–summary linkage**
  The discussion reviews how reference identifiers are assigned and used to bridge metadata, summaries, and datasets. Hongye explains that trunk metadata stores extracted reference indexes and that she manually ensures uniqueness across meetings. The conversation further clarifies how summary IDs and topic IDs are structured and how metadata first identifies the correct dataset before linking to summary chunks. The small size of the dataset results in only one summary being selected from multiple potential matches.
  Reference: M010-T01

- **[S010-T02] Topic: Metadata embedding fields and inverted index design**
  This topic outlines how metadata embeddings incorporate concise summary briefs and action descriptions. Ankit requests clearer visual diagrams to represent the retrieval flow. They discuss naming conventions, the fields used for metadata-level matching, and how many metadata entries can map to a single dataset via an inverted indexing approach.
  Reference: M010-T02

- **[S010-T03] Topic: Building metadata embeddings and selecting top metadata candidates**
  The conversation focuses on how metadata embeddings are constructed by combining brief summaries and descriptions. Hongye shows embedding dimensions and clarifies differences between metadata-level and summary-level embeddings. Ankit examines how cosine similarity is applied and how metadata candidates are selected, noting that Hongye currently uses top-1 selection due to the dataset’s small scale. They acknowledge that this may need adjustment as the dataset grows.
  Reference: M010-T03

- **[S010-T04] Topic: Segment handling and mapping summary chunks to meeting embeddings**
  They review how long segments are managed and how repeated bullet points maintain continuity across segments. Ankit provides descriptive pipeline text for Hongye to incorporate into her slides. The topic also covers how summary-level cosine similarity outputs reference IDs that are then used to locate relevant meeting-level embeddings for deeper comparison.
  Reference: M010-T04

- **[S010-T05] Topic: Meeting-level embeddings, topic chunks, and FAISS index usage**
  This topic examines the differences between topic-level and summary-level chunks, the inclusion of IDs inside embeddings, and how meeting text is split for meeting-level comparisons. They discuss restricting meeting-level comparisons using metadata results, the role of FAISS indexing, and how vector search complements BM25 during retrieval.
  Reference: M010-T05

- **[S010-T06] Topic: BM25 on summarized text and risk of missing query keywords**
  Ankit highlights the limitation that BM25 is applied on summaries rather than full text, which can cause keyword mismatches when summaries omit important terms. They review examples where compressed summaries remove specific expressions, leading to zero BM25 scores even when the original text contained relevant information.
  Reference: M010-T06

- **[S010-T07] Topic: Hybrid score computation and current retrieval quality**
  The discussion explores how vector and BM25 scores are combined through a hybrid scoring formula. Hongye explains how search results are structured, while Ankit emphasizes verifying the formula. They evaluate current retrieval quality, noting correct retrieval but identifying structural concerns due to the small dataset and the C-structure design.
  Reference: M010-T07

- **[S010-T08] Topic: Alternative retrieval paths and testing with LLM**
  They consider alternative retrieval paths that bypass metadata, such as directly using summary-level and meeting-level embeddings. Ankit raises potential inconsistencies when multiple paths yield different top-K results. They agree that extensive testing is needed and discuss integrating an LLM component while exploring retrieval accuracy.
  Reference: M010-T08

- **[S010-T09] Topic: Scalability concerns, DFS vs BFS, and graph-embedding idea**
  This topic addresses how the current design may scale poorly with large numbers of meetings. Ankit compares the current approach to a depth-first strategy and suggests a breadth-first alternative to reduce risk. He introduces the idea of a graph-embedding layer that captures connections between meetings, participants, dates, and keywords, potentially improving retrieval flexibility and robustness.
  Reference: M010-T09

- **[S010-T10] Topic: Use of Gemini transcripts and meeting recordings for validation**
  The final topic reviews how transcription is handled by Gemini and how Hongye manually validates summaries. They reaffirm the value of using their own meeting transcripts for evaluation and testing, and conclude with plans for another meeting after Hongye’s deadline.
  Reference: M010-T10

- **[S010-A01] Action: Hongye Qian — Create PPT diagram of retrieval flow**
  Hongye will prepare a more detailed PPT diagram illustrating the metadata, summary, and meeting-level steps of the retrieval flow, following Ankit’s request for clearer visualization.
  Reference: M010-A01

- **[S010-A02] Action: Ankit — Integrate LLM component and test retrieval pipeline**
  Ankit will add the LLM component into the retrieval pipeline and perform extensive testing, as discussed during the evaluation of alternative retrieval paths.
  Reference: M010-A02

- **[S010-A03] Action: Hongye Qian & Ankit — Use meeting transcripts for validation**
  Both will incorporate their own meeting transcripts into the test dataset and plan a follow-up session on Monday to review progress and retrieval performance.
  Reference: M010-A03
