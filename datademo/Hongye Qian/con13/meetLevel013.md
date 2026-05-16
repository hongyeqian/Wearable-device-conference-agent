# Meeting Level Summary

## Key Topics

- **Topic Title:** Summary Architecture Discussion
  - Topic id: M013-T01
  - Reference: P001–P013
  - Summary:
      - [00:00] Ankit: Discussing the need for a summary architecture.
      - [00:07] Hongye Qian: Emphasizing the importance of storing information from raw data.
      - [01:25] Hongye Qian: Concerned about user queries and the architecture's ability to handle them.
      - [02:27] Hongye Qian: Noting the trade-off between computation cost and information retention.
      - [03:02] Ankit: Suggesting a user query approach to metadata and summaries.
      - [03:36] Hongye Qian: Mentioning the need for testing and adjustments in logic.
      - [04:44] Ankit: Proposing a demo to evaluate the system's capabilities.
      - [05:28] Hongye Qian: Confirming the start with the second architecture.
      - [06:17] Ankit: Requesting a detailed system diagram for presentations.
      - [07:18] Ankit: Discussing the need for a basic chat interface for testing.
      - [08:10] Hongye Qian: Acknowledging the usefulness of a course on system design.
  
- **Topic Title:** Metadata and Summarization
  - Topic id: M013-T02
  - Reference: P042–P070
  - Summary:
      - [00:00] Ankit: Asking about the screen sharing for metadata discussion.
      - [00:35] Ankit: Requesting to see Hongye's metadata.
      - [01:07] Hongye Qian: Clarifying that topics and summaries are part of metadata.
      - [01:39] Ankit: Discussing the selection of multiple meetings for better results.
      - [02:15] Hongye Qian: Suggesting to reduce search space by selecting multiple meetings.
      - [03:10] Ankit: Inquiring about the need for summarization before user queries.
      - [03:32] Hongye Qian: Explaining the role of a separate summarization module.

- **Topic Title:** Semantic Normalization
  - Topic id: M013-T03
  - Reference: P071–P096
  - Summary:
      - [03:58] Ankit: Introducing the concept of a semantic module for user queries.
      - [04:50] Ankit: Discussing the importance of normalizing user questions.
      - [07:35] Hongye Qian: Highlighting the need to extract key elements from queries.
      - [09:01] Hongye Qian: Emphasizing the importance of identifying significant words in queries.
      - [09:40] Ankit: Suggesting that normalization should precede metadata filtering.

- **Topic Title:** Architecture Design and Implementation
  - Topic id: M013-T04
  - Reference: P097–P140
  - Summary:
      - [09:52] Ankit: Discussing the architecture for summarizing meeting levels.
      - [10:51] Ankit: Explaining the need for different summary levels based on conversation length.
      - [12:58] Ankit: Proposing a logic to determine the level of user questions.
      - [14:17] Ankit: Presenting a new architecture design for simultaneous summarization.
      - [21:52] Ankit: Describing the collaboration between summary and mid-level architectures.
      - [27:21] Hongye Qian: Discussing the need for LLM to identify conversation boundaries.

## Action Items

- **Responsible Person:** Hongye Qian
  - Topic id: M013-A01
  - Reference: P026–P027
  - Task: Begin with the second architecture design.
  - Context: Discussed during the architecture discussion.
  - Duration: [05:28–05:40]
  - Deadline (if any): TBD

- **Responsible Person:** Ankit
  - Topic id: M013-A02
  - Reference: P027–P028
  - Task: Prepare a detailed system diagram for presentations.
  - Context: Suggested during the architecture discussion.
  - Duration: [06:17–06:21]
  - Deadline (if any): TBD

- **Responsible Person:** Ankit
  - Topic id: M013-A03
  - Reference: P032–P033
  - Task: Develop a basic chat interface for testing.
  - Context: Discussed during the architecture design.
  - Duration: [07:18–07:32]
  - Deadline (if any): TBD