# Summary Level Summary

## M013 — Summary Architecture Discussion

- **[S013-T01] Topic: Summary Architecture Discussion**
  The discussion centered on the necessity of a summary architecture, highlighting the importance of storing information from raw data and addressing user queries. Concerns were raised regarding the trade-off between computational costs and information retention, alongside suggestions for a user query approach to metadata and summaries. The need for testing, adjustments in logic, and a demo to evaluate system capabilities were also emphasized, along with the requirement for a detailed system diagram and a basic chat interface for testing purposes. 
  Reference: M013-T01

- **[S013-T02] Topic: Metadata and Summarization**
  The conversation focused on metadata, with clarifications that topics and summaries are integral components. The selection of multiple meetings was discussed as a strategy to enhance results, and the necessity of summarization prior to user queries was questioned. A separate summarization module's role was explained, indicating its significance in the overall process. 
  Reference: M013-T02

- **[S013-T03] Topic: Semantic Normalization**
  The introduction of a semantic module for user queries was presented, emphasizing the normalization of user questions. Key elements extraction from queries and the identification of significant words were highlighted as critical components of the process. It was suggested that normalization should occur before metadata filtering to improve query handling. 
  Reference: M013-T03

- **[S013-T04] Topic: Architecture Design and Implementation**
  The architecture for summarizing meeting levels was discussed, with an explanation of the need for varying summary levels based on conversation length. A logic to determine the level of user questions was proposed, alongside a new architecture design for simultaneous summarization. Collaboration between summary and mid-level architectures was described, with a focus on the necessity for LLM to identify conversation boundaries. 
  Reference: M013-T04

- **[S013-A01] Action: Hongye Qian to Begin with Second Architecture Design**
  Hongye Qian is tasked with initiating the second architecture design as discussed during the architecture discussion. 
  Reference: M013-A01

- **[S013-A02] Action: Ankit to Prepare Detailed System Diagram**
  Ankit is responsible for preparing a detailed system diagram for presentations, as suggested during the architecture discussion. 
  Reference: M013-A02

- **[S013-A03] Action: Ankit to Develop Basic Chat Interface**
  Ankit is tasked with developing a basic chat interface for testing, which was discussed during the architecture design session. 
  Reference: M013-A03