"""
RAG-specific prompt templates for LangChain integration.
"""

class RAGPrompts:
    """RAG-specific prompt templates for different use cases."""
    
    @staticmethod
    def get_rag_prompt_template() -> str:
        """
        Main RAG prompt template for generating answers based on context.
        
        Returns:
            str: Formatted prompt template string
        """
        return """
        You are a helpful AI assistant that answers questions based on provided context.
        
        Context: {context}
        
        Question: {question}
        
        Instructions:
        - Answer the question using only the provided context
        - If the context doesn't contain relevant information, say "I don't have enough context to answer this question"
        - Keep answers concise and to the point
        - Do not make up information not present in the context
        - If you cannot answer, be honest about it
        - Provide specific details from the context when available
        - Structure your response clearly and logically
        """
    
    @staticmethod
    def get_context_summary_prompt() -> str:
        """
        Prompt template for summarizing retrieved context.
        
        Returns:
            str: Formatted prompt template string
        """
        return """
        Summarize the following context in 2-3 concise sentences, focusing on the most relevant information:
        
        Context: {context}
        
        Summary:
        """
    
    @staticmethod
    def get_context_validation_prompt() -> str:
        """
        Prompt template for validating if context is relevant to the question.
        
        Returns:
            str: Formatted prompt template string
        """
        return """
        Determine if the following context is relevant to answering the question.
        Respond with ONLY "RELEVANT" or "NOT_RELEVANT".
        
        Context: {context}
        
        Question: {question}
        
        Relevance:
        """
    
    @staticmethod
    def get_answer_refinement_prompt() -> str:
        """
        Prompt template for refining an initial answer based on context.
        
        Returns:
            str: Formatted prompt template string
        """
        return """
        Refine the following answer based on the provided context. 
        Make it more accurate, complete, and context-specific.
        
        Context: {context}
        
        Initial Answer: {initial_answer}
        
        Refined Answer:
        """
    
    @staticmethod
    def get_context_extraction_prompt() -> str:
        """
        Prompt template for extracting key information from context.
        
        Returns:
            str: Formatted prompt template string
        """
        return """
        Extract the most relevant information from the context that would help answer the question.
        Return only the extracted information, nothing else.
        
        Context: {context}
        
        Question: {question}
        
        Relevant Information:
        """
    
    @staticmethod
    def get_fallback_prompt() -> str:
        """
        Prompt template for when no relevant context is found.
        
        Returns:
            str: Formatted prompt template string
        """
        return """
        You are a helpful AI assistant. 
        Unfortunately, I don't have access to relevant context to answer your question.
        
        Question: {question}
        
        Response:
        I don't have enough context to answer this question. Please provide more information or check if the relevant documents have been uploaded to the system.
        """
